import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import io
import os, re
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    UnstructuredFileLoader, # 범용 파일 로더 (내부적으로 적절한 로더 선택 시도)
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader # txt는 TextLoader 유지 또는 UnstructuredFileLoader 사용 가능
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
    ".pdf": (UnstructuredPDFLoader, {"mode": "paged", "strategy": "hi_res"}), # hi_res는 레이아웃 분석에 더 유리할 수 있음 (선택적)
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {"mode": "paged"}), # 또는 "single" 모드 선택 가능
    ".xls": (UnstructuredExcelLoader, {"mode": "paged"}),
    ".txt": (TextLoader, {"encoding": "utf-8"}), # 또는 UnstructuredFileLoader 사용 가능
    # 필요에 따라 다른 형식(pptx 등) 추가 가능
    # ".pptx": (UnstructuredPowerPointLoader, {}),
}

def load_document(file_path: str) -> List[Document]:
    """파일 경로를 받아 확장자에 맞는 로더를 사용하여 문서를 로드합니다. (unstructured 사용 강화)"""
    ext = Path(file_path).suffix.lower()
    loader_info = LOADER_MAPPING.get(ext)

    if not loader_info:
        print(f"지원하지 않는 파일 형식({ext}), UnstructuredFileLoader로 시도: {file_path}")
        # 지원 목록에 없으면 범용 로더 시도 (선택적)
        loader = UnstructuredFileLoader(file_path, mode="paged")
        # 또는 여기서 에러 처리하고 빈 리스트 반환
        # print(f"지원하지 않는 파일 형식 건너뛰기: {ext} ({file_path})")
        # return []
    else:
        loader_class, loader_kwargs = loader_info
        loader = loader_class(file_path, **loader_kwargs)

    try:
        docs = loader.load()
        # --- 간단한 전처리 단계 추가 ---
        processed_docs = []
        for doc in docs:
            # 예시: 연속된 공백/줄바꿈 제거, 앞뒤 공백 제거
            cleaned_content = re.sub(r'\s+', ' ', doc.page_content).strip()
            # 예시: 특정 머리글/바닥글 패턴 제거 (필요시 정규식 사용)
            # cleaned_content = re.sub(r'페이지 \d+ / \d+', '', cleaned_content)

            if cleaned_content: # 내용이 있는 경우에만 추가
                doc.page_content = cleaned_content
                doc.metadata["source"] = os.path.basename(file_path)
                doc.metadata["loaded_at"] = datetime.now().isoformat()
                # unstructured 로더는 자체적으로 페이지 번호(metadata['page_number']) 등을 잘 넣어주는 경우가 많음
                # 기존 'page' 메타데이터와의 일관성 확인 필요
                if 'page_number' in doc.metadata and 'page' not in doc.metadata:
                     doc.metadata['page'] = doc.metadata['page_number']

                processed_docs.append(doc)

        if not processed_docs:
             print(f"문서 로드 후 처리 결과 내용 없음: {file_path}")
             return []

        return processed_docs

    except Exception as e:
        print(f"파일 로딩 중 오류 발생 ({file_path}): {e}")
        traceback.print_exc()
        return []

def split_text(documents: List[Document], chunk_size: int = 768, chunk_overlap: int = 128) -> List[Document]:
    """의미 단위로 문서를 청크로 분할합니다."""
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        keep_separator=True,  # 구분자 유지
        is_separator_regex=False
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
            
            enhanced_chunks.append(Document(page_content=enhanced_content, metadata=metadata))
        
        return enhanced_chunks
    except Exception as e:
        print(f"텍스트 분할 중 오류 발생: {e}")
        traceback.print_exc()
        return []

def index_chunks_to_elasticsearch(es_client: Any, embedding_function: Any, chunks: List[Document], category: str):
    """분할된 텍스트 청크를 임베딩하여 Elasticsearch에 인덱싱합니다."""
    if not es_client:
        print("Elasticsearch 클라이언트가 제공되지 않아 인덱싱을 건너뜁니다.")
        return False
    if not embedding_function:
        print("임베딩 함수가 제공되지 않아 인덱싱을 건너뜁니다.")
        return False
    if not chunks:
        print("인덱싱할 청크가 없습니다.")
        return False
    
    batch_size = 100  # 한 번에 처리할 청크 수
    success_count = 0
    failure_count = 0
    
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        valid_chunks = [chunk for chunk in batch_chunks if chunk.page_content.strip()]
        chunk_texts = [chunk.page_content for chunk in valid_chunks]
        
        if not chunk_texts:
            continue
        
        try:
            embeddings = embedding_function(chunk_texts)
            
            if len(embeddings) != len(chunk_texts):
                print(f"오류: 생성된 임베딩 수({len(embeddings)})가 청크 수({len(chunk_texts)})와 일치하지 않습니다.")
                continue
            
            actions = []
            for j, (chunk, text, embedding) in enumerate(zip(valid_chunks, chunk_texts, embeddings)):
                if not text.strip():
                    continue
                
                page_number = chunk.metadata.get('page', i + j + 1)
                source_file = chunk.metadata.get('source', 'unknown')
                doc_id = f"{source_file.replace('.', '_')}_{page_number}_{j}"

                image_path = chunk.metadata.get('image_path', '')
                
                action = {
                    "_index": ES_INDEX_NAME,
                    "_id": doc_id,
                    "_source": {
                        "text": text,
                        "embedding": embedding,
                        "source": source_file,
                        "page": page_number,
                        "category": category,
                        "chunk_id": chunk.metadata.get("chunk_id", j),
                        "total_chunks": chunk.metadata.get("total_chunks", 0),
                        "indexed_at": datetime.now().isoformat(),
                        "image_path": image_path  # 이미지 경로 추가
                    }
                }
                actions.append(action)
            
            if actions:
                success, failed = bulk(
                    es_client, actions, chunk_size=batch_size,
                    request_timeout=180,max_retries=5,initial_backoff=2,
                    max_backoff=60,raise_on_error=False
                    )
                success_count += success
                failure_count += len(failed) if failed else 0
                
                print(f"배치 {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} 완료: 성공={success}, 실패={len(failed) if failed else 0}")
                
                if failed:
                    for item in failed:
                        print(f"인덱싱 실패 항목: {item}")
        
        except Exception as e:
            print(f"배치 {i//batch_size + 1} 처리 중 오류 발생: {e}")
            traceback.print_exc()
            failure_count += len(batch_chunks)
    
    print(f"인덱싱 완료: 총 {len(chunks)} 청크 중 {success_count}개 성공, {failure_count}개 실패")
    return success_count > 0

def process_and_index_file(es_client: Any, embedding_function: Any, file_path: str, category: str, chunk_size: int = 768, chunk_overlap: int = 128) -> bool:
    """단일 파일을 로드, 분할, 임베딩 및 인덱싱하는 전체 프로세스를 실행합니다."""
    print(f"파일 처리 시작: {file_path}, 카테고리: {category}")
    
    if not es_client or not embedding_function:
        print("Elasticsearch 클라이언트 또는 임베딩 함수가 유효하지 않습니다.")
        return False
      
    
    documents = load_document(file_path)
    if not documents:
        print(f"문서 로드 실패 또는 내용 없음: {file_path}")
        return False
    print(f"문서 로드 완료: {len(documents)} 페이지/섹션")
    
    original_filename = os.path.basename(file_path)
    for doc in documents:
        doc.metadata["source"] = original_filename
   
    
    chunks = split_text(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        print(f"텍스트 분할 실패 또는 청크 없음: {file_path}")
        return False
    print(f"텍스트 분할 완료: {len(chunks)} 청크 생성")
    
    success = index_chunks_to_elasticsearch(es_client, embedding_function, chunks, category)
    if success:
        print(f"파일 인덱싱 성공: {file_path}")
    else:
        print(f"파일 인덱싱 실패: {file_path}")
    
    return success

