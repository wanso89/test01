import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import os, re, asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    UnstructuredFileLoader,  # 범용 파일 로더 (내부적으로 적절한 로더 선택 시도)
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader,  # txt는 TextLoader 유지 또는 UnstructuredFileLoader 사용 가능
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from elasticsearch.helpers import bulk
import traceback
from datetime import datetime

ES_INDEX_NAME = "rag_documents_kure_v1"

IMAGE_DIR = "static/document_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

LOADER_MAPPING = {
    ".pdf": (
        UnstructuredPDFLoader,
        {"mode": "paged", "strategy": "hi_res"},
    ),  # hi_res는 레이아웃 분석에 더 유리할 수 있음 (선택적)
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".xlsx": (
        UnstructuredExcelLoader,
        {"mode": "paged"},
    ),  # 또는 "single" 모드 선택 가능
    ".xls": (UnstructuredExcelLoader, {"mode": "paged"}),
    ".txt": (
        TextLoader,
        {"encoding": "utf-8"},
    ),  # 또는 UnstructuredFileLoader 사용 가능
    # 필요에 따라 다른 형식(pptx 등) 추가 가능
    # ".pptx": (UnstructuredPowerPointLoader, {}),
}


def load_document(file_path: str) -> List[Document]:
    """파일 로드 및 한국어 문서 전처리 강화."""
    ext = Path(file_path).suffix.lower()
    loader_info = LOADER_MAPPING.get(ext)
    if not loader_info:
        print(
            f"지원하지 않는 파일 형식({ext}), UnstructuredFileLoader로 시도: {file_path}"
        )
        loader = UnstructuredFileLoader(file_path, mode="paged")
    else:
        loader_class, loader_kwargs = loader_info
        loader = loader_class(file_path, **loader_kwargs)

    try:
        docs = loader.load()
        processed_docs = []
        for doc in docs:
            # 한국어 문서 전처리: 반복 머리글/바닥글 제거
            cleaned_content = re.sub(
                r"Cloudera 운영자메뉴얼|Version \d+\.\d+|Page \d+/\d+|네오오토|취업규칙",
                "",
                doc.page_content,
            )
            # 연속 공백/줄바꿈 제거
            cleaned_content = re.sub(r"\s+", " ", cleaned_content).strip()
            # 불필요한 접미사 제거 (예: "-습니다", "-합니다")
            cleaned_content = re.sub(r"(습니다|합니다|입니다)\s*", " ", cleaned_content)

            if cleaned_content:
                doc.page_content = cleaned_content
                doc.metadata["source"] = os.path.basename(file_path)
                doc.metadata["loaded_at"] = datetime.now().isoformat()
                if "page_number" in doc.metadata and "page" not in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page_number"]
                processed_docs.append(doc)

        return processed_docs
    except Exception as e:
        print(f"파일 로딩 중 오류 발생 ({file_path}): {e}")
        traceback.print_exc()
        return []


def split_text(
    documents: List[Document],
    chunk_size: int = 768,
    chunk_overlap: int = 128,
    adaptive: bool = True,
) -> List[Document]:
    """의미 단위로 문서를 청크로 분할하며, 문서 특성에 따라 크기와 오버랩을 조정."""
    if not documents:
        return []

    # 문서 길이와 구조에 따라 청킹 크기 동적 조정
    total_length = sum(len(doc.page_content) for doc in documents)
    avg_doc_length = total_length / len(documents) if documents else 0
    if adaptive:
        if avg_doc_length < 500:  # 짧은 문서
            chunk_size = max(256, chunk_size // 2)
            chunk_overlap = max(64, chunk_overlap // 2)
        elif avg_doc_length > 2000:  # 긴 문서
            chunk_size = min(1024, chunk_size * 2)
            chunk_overlap = min(256, chunk_overlap * 2)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        keep_separator=True,
        is_separator_regex=False,
    )

    try:
        chunks = text_splitter.split_documents(documents)
        enhanced_chunks = []
        doc_title = ""

        for i, chunk in enumerate(chunks):
            if i == 0 and "\n" in chunk.page_content:
                first_line = chunk.page_content.split("\n")[0].strip()
                if len(first_line) < 100 and not first_line.endswith("."):
                    doc_title = first_line

            enhanced_content = chunk.page_content
            if doc_title and doc_title not in enhanced_content:
                enhanced_content = f"{doc_title}\n\n{enhanced_content}"

            metadata = chunk.metadata.copy()
            metadata["chunk_id"] = i
            metadata["total_chunks"] = len(chunks)

            enhanced_chunks.append(
                Document(page_content=enhanced_content, metadata=metadata)
            )

        return enhanced_chunks
    except Exception as e:
        print(f"텍스트 분할 중 오류 발생: {e}")
        traceback.print_exc()
        return []


async def index_chunks_to_elasticsearch(
    es_client: Any, embedding_function: Any, chunks: List[Document], category: str
):
    """비동기 배치 처리로 Elasticsearch 인덱싱 속도 최적화."""
    if not chunks:
        print("인덱싱할 청크가 없습니다.")
        return False

    batch_size = 500
    success_count = 0
    failure_count = 0

    async def process_batch(batch_chunks, batch_index):
        nonlocal success_count, failure_count
        valid_chunks = [chunk for chunk in batch_chunks if chunk.page_content.strip()]
        chunk_texts = [chunk.page_content for chunk in valid_chunks]

        if not chunk_texts:
            return

        try:
            embeddings = await asyncio.to_thread(embedding_function, chunk_texts)
            actions = []
            for j, (chunk, embedding) in enumerate(zip(valid_chunks, embeddings)):
                page_number = chunk.metadata.get("page", batch_index + j + 1)
                source_file = chunk.metadata.get("source", "unknown")
                doc_id = f"{source_file.replace('.', '_')}_{page_number}_{j}"
                action = {
                    "_index": ES_INDEX_NAME,
                    "_id": doc_id,
                    "_source": {
                        "text": chunk.page_content,
                        "embedding": embedding,
                        "source": source_file,
                        "page": page_number,
                        "category": category,
                        "chunk_id": chunk.metadata.get("chunk_id", j),
                        "total_chunks": chunk.metadata.get("total_chunks", 0),
                        "indexed_at": datetime.now().isoformat(),
                    },
                }
                actions.append(action)

            if actions:
                success, failed = await asyncio.to_thread(
                    bulk,
                    es_client,
                    actions,
                    chunk_size=batch_size,
                    request_timeout=180,
                    max_retries=5,
                )
                success_count += success
                failure_count += len(failed) if failed else 0
        except Exception as e:
            print(f"배치 {batch_index//batch_size + 1} 처리 중 오류 발생: {e}")
            failure_count += len(batch_chunks)

    tasks = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        tasks.append(process_batch(batch_chunks, i))

    await asyncio.gather(*tasks)
    print(
        f"인덱싱 완료: 총 {len(chunks)} 청크 중 {success_count}개 성공, {failure_count}개 실패"
    )
    return success_count > 0


async def process_and_index_file(
    es_client: Any,
    embedding_function: Any,
    file_path: str,
    category: str,
    chunk_size: int = 768,
    chunk_overlap: int = 128,
) -> bool:
    """단일 파일을 로드, 분할, 임베딩 및 인덱싱하는 전체 프로세스를 실행합니다. 문서 유형별 처리와 상세 로깅 포함."""
    print(f"파일 처리 시작: {file_path}, 카테고리: {category}")
    failed_chunks_log = []

    if not es_client or not embedding_function:
        print("Elasticsearch 클라이언트 또는 임베딩 함수가 유효하지 않습니다.")
        return False

    # 문서 로드
    documents = load_document(file_path)
    if not documents:
        print(f"문서 로드 실패 또는 내용 없음: {file_path}")
        return False
    print(f"문서 로드 완료: {len(documents)} 페이지/섹션")

    original_filename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source"] = original_filename

    # 문서 유형 추정 (간단한 키워드 기반)
    first_doc_content = documents[0].page_content.lower()
    if "faq" in first_doc_content or "질문" in first_doc_content:
        doc_type = "faq"
        chunk_size = 512  # FAQ는 짧은 청크로
        chunk_overlap = 64
    elif "규칙" in first_doc_content or "규정" in first_doc_content:
        doc_type = "regulation"
        chunk_size = 1024  # 규정집은 긴 문맥 유지
        chunk_overlap = 256
    else:
        doc_type = "manual"
        chunk_size = 768  # 기본 매뉴얼
        chunk_overlap = 128

    print(
        f"문서 유형 추정: {doc_type}, 청크 크기: {chunk_size}, 오버랩: {chunk_overlap}"
    )

    # 텍스트 분할
    chunks = split_text(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        print(f"텍스트 분할 실패 또는 청크 없음: {file_path}")
        return False
    print(f"텍스트 분할 완료: {len(chunks)} 청크 생성")

    # 인덱싱
    success = await index_chunks_to_elasticsearch(
        es_client, embedding_function, chunks, category
    )
    if not success:
        print(f"파일 인덱싱 실패: {file_path}")
        # 실패 로그 저장
        with open("failed_indexing.log", "a") as f:
            f.write(
                f"[{datetime.now().isoformat()}] Failed to index: {file_path}, Category: {category}\n"
            )

    return success
