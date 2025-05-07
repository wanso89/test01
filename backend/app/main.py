import os
import asyncio
import uuid
import time
import hashlib
import json
import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from elasticsearch import Elasticsearch
from app.utils.indexing_utils import process_and_index_file, ES_INDEX_NAME


# 모델 임포트
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import traceback

traceback.print_exc()

# 설정 상수
# LLM_MODEL_NAME = r"/home/root/ko-gemma-v1"
# EMBEDDING_MODEL_NAME = r"/home/root/KURE-v1"
# RERANKER_MODEL_NAME = r"/home/root/ko-reranker"
LLM_MODEL_NAME = r"/home/root/Gukbap-Gemma2-9B"
EMBEDDING_MODEL_NAME = r"/home/root/kpf-sbert-v1.1"
RERANKER_MODEL_NAME = r"/home/root/ko-reranker"
ES_HOST = "http://172.10.2.70:9200"
STATIC_DIR = "app/static"
IMAGE_DIR = os.path.join(STATIC_DIR, "document_images")
os.makedirs(IMAGE_DIR, exist_ok=True)


# 질문 요청 모델
class QuestionRequest(BaseModel):
    question: str
    category: Optional[str] = "메뉴얼"
    history: Optional[List[Dict[str, Any]]] = []


# FastAPI 앱 초기화
app = FastAPI(title="RAG Chatbot API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 설정
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# 모델 로드 함수들
def get_elasticsearch_client():
    """Elasticsearch 클라이언트를 로드합니다."""
    print("Initializing Elasticsearch client...")
    try:
        client = Elasticsearch(
            ES_HOST,
            request_timeout=60,
            retry_on_timeout=True,
            max_retries=3,
            verify_certs=False,
        )

        if not client.ping():
            print("Elasticsearch 서버에 연결할 수 없습니다. 서버 상태를 확인하세요.")
            return None

        print("Elasticsearch client connected successfully.")

        # 인덱스 존재 확인 및 생성
        if not client.indices.exists(index=ES_INDEX_NAME):
            INDEX_SETTINGS = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "30s",
                    "analysis": {
                        "analyzer": {
                            "korean": {
                                "type": "custom",
                                "tokenizer": "nori_tokenizer",
                                "filter": ["lowercase", "nori_part_of_speech"],
                            }
                        }
                    },
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "korean",
                            "search_analyzer": "korean",
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 768,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type": "hnsw",
                                "m": 16,
                                "ef_construction": 100,
                            },
                        },
                        "source": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "category": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "total_chunks": {"type": "integer"},
                        "indexed_at": {"type": "date"},
                        "image_path": {"type": "keyword"},
                    }
                },
            }

            try:
                client.indices.create(index=ES_INDEX_NAME, body=INDEX_SETTINGS)
                print(f"Elasticsearch 인덱스 '{ES_INDEX_NAME}' 생성 완료.")
            except Exception as e:
                print(f"Elasticsearch 인덱스 생성 실패: {e}")
                return None

        return client
    except Exception as e:
        print(f"Elasticsearch 클라이언트 초기화 중 오류 발생: {e}")
        traceback.print_exc()
        return None


def get_embedding_function():
    """임베딩 모델 함수를 로드합니다."""
    print("Loading embedding model...")
    try:
        # LangchainEmbeddingFunction 클래스 정의
        class LangchainEmbeddingFunction:
            def __init__(self, model_name: str):
                # HuggingFaceEmbeddings 직접 사용
                self.model = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    },
                )

            def __call__(self, texts: list[str]) -> List[List[float]]:
                # embed_documents 메서드 사용
                return self.model.embed_documents(texts)

        embedding_func = LangchainEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)

        # 테스트 임베딩 실행
        try:
            test_result = embedding_func(["테스트"])
            print(
                f"Embedding model loaded successfully. Vector dimension: {len(test_result[0])}"
            )
            return embedding_func
        except Exception as e:
            print(f"임베딩 모델 테스트 중 오류 발생: {e}")
            traceback.print_exc()
            return None
    except Exception as e:
        print(f"임베딩 모델 로딩 중 오류 발생: {e}")
        traceback.print_exc()
        return None


def get_llm_model_and_tokenizer():
    print("Loading LLM model and tokenizer...")
    try:
        torch.cuda.empty_cache()  # 메모리 정리
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto",
            revision="main",
        )
        print("LLM model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"LLM 모델 또는 토크나이저 로딩 중 오류 발생: {e}")
        traceback.print_exc()
        return None, None


def get_reranker_model():
    """Reranker 모델을 로드합니다."""
    print("Loading reranker model...")
    try:
        reranker = CrossEncoder(
            RERANKER_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Reranker model loaded successfully.")
        return reranker
    except Exception as e:
        print(f"Reranker 모델 로딩 중 오류 발생: {e}")
        traceback.print_exc()
        return None


# ElasticsearchRetriever 클래스 정의
class ElasticsearchRetriever:
    def __init__(self, es_client: Any, embedding_function: Any, category: str, k=20):
        self.es_client = es_client
        self.index_name = ES_INDEX_NAME
        self.embedding_function = embedding_function
        self.k = k
        self.category = category

    def get_relevant_documents(self, query: str) -> List[Document]:
        if not self.es_client or not self.embedding_function:
            print("Elasticsearch 클라이언트 또는 임베딩 함수가 초기화되지 않았습니다.")
            return []

        try:
            # 임베딩 생성
            query_embedding = self.embedding_function([query])[0]

            hybrid_query = {
                "size": self.k,
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "bool": {
                        "should": [
                            # 1. 정확한 문구 검색
                            {
                                "match_phrase": {
                                    "text": {"query": query, "boost": 3.0, "slop": 2}
                                }
                            },
                            # 2. BM25 키워드 검색
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": 2.0,
                                        "operator": "OR",
                                        "minimum_should_match": "50%",
                                    }
                                }
                            },
                            # 3. 벡터 검색 (script_score)
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding},
                                    },
                                    "boost": 1.5,
                                }
                            },
                        ],
                        "filter": [{"term": {"category": self.category}}],
                        "minimum_should_match": 1,
                    }
                },
            }

            # 검색 실행
            response = self.es_client.search(index=self.index_name, body=hybrid_query)

            # 결과 처리
            docs = []
            for hit in response["hits"]["hits"]:
                # 메타데이터 추출 (embedding 필드 제외)
                metadata = {}
                for k, v in hit["_source"].items():
                    if k != "text" and k != "embedding":
                        metadata[k] = v

                # 점수 정규화 (0~1 사이로)
                raw_score = hit["_score"]
                # Elasticsearch 스코어는 범위가 다양하므로 정규화
                normalized_score = min(
                    max(raw_score / 10, 0), 1
                )  # 10으로 나누어 0~1 범위로 조정

                metadata["relevance_score"] = hit["_score"]
                metadata["source"] = hit["_source"].get("source", "unknown")
                metadata["page"] = hit["_source"].get("page", 1)

                # Document 객체 생성
                docs.append(
                    Document(
                        page_content=hit["_source"].get("text", ""), metadata=metadata
                    )
                )

            print(f"검색 완료: {len(docs)} 문서 검색됨")
            return docs

        except Exception as e:
            print(f"Elasticsearch 검색 오류: {e}")
            traceback.print_exc()
            return []


# 향상된 리랭커 클래스 정의
class EnhancedLocalReranker:
    def __init__(self, reranker_model: Any, top_n=15):
        self.reranker = reranker_model
        self.top_n = top_n

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs or not self.reranker:
            return []

        try:
            # 상위 10개 문서만 리랭킹
            docs_to_rerank = docs[:10]
            pairs = [(query, doc.page_content) for doc in docs_to_rerank]
            scores = self.reranker.predict(pairs)

            for doc, score in zip(docs_to_rerank, scores):
                normalized_score = min(max((score + 1) / 2, 0), 1)
                doc.metadata["rerank_score"] = float(normalized_score)

            # 리랭킹된 문서와 나머지 문서 결합
            sorted_docs = sorted(
                docs_to_rerank,
                key=lambda x: x.metadata.get("rerank_score", 0.0),
                reverse=True,
            )
            sorted_docs.extend([doc for doc in docs[10:] if doc not in docs_to_rerank])

            threshold = 0.6
            filtered_docs = [
                doc
                for doc in sorted_docs
                if doc.metadata.get("rerank_score", 0.0) >= threshold
            ]

            if not filtered_docs and sorted_docs:
                filtered_docs = [sorted_docs[0]]

            return filtered_docs[: self.top_n]
        except Exception as e:
            print(f"Reranking 중 오류 발생: {e}")
            traceback.print_exc()
            return docs


# LLM 답변 생성 함수
async def generate_llm_response(
    llm_model: Any,
    tokenizer: Any,
    question: str,
    top_docs: List[Document],
    temperature: float = 0.2,
    conversation_history=None,
) -> str:
    if not llm_model or not tokenizer:
        return "LLM 모델 또는 토크나이저가 로드되지 않았습니다."
    if not top_docs:
        return "관련 문서를 찾지 못해 답변을 생성할 수 없습니다."

    # 대화 기록이 없으면 빈 리스트로 초기화
    if conversation_history is None:
        conversation_history = []

    # 문서 컨텍스트 준비
    context_parts = []
    for i, doc in enumerate(top_docs[:5]):  # 상위 5개 문서 사용
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[문서 {i+1}] (출처: {source}, 페이지: {page})\n{doc.page_content.strip()}"
        )

    combined_context = "\n\n".join(context_parts)

    # 이전 대화 기록 포맷팅
    conversation_context = ""
    if conversation_history:
        conversation_parts = []
        # 대화 기록의 최대 길이 제한 (예: 최근 5개 턴)
        recent_history = (
            conversation_history[-5:]
            if len(conversation_history) > 5
            else conversation_history
        )

        for msg in recent_history:
            role = "사용자" if msg["role"] == "user" else "시스템"
            conversation_parts.append(f"{role}: {msg['content']}")

        conversation_context = "대화 기록:\n" + "\n".join(conversation_parts) + "\n\n"

    # 간결하고 명확한 프롬프트
    prompt = f"""다음 질문에 제공된 문서 정보만을 기반으로 정확하게 답변하세요:

{conversation_context}현재 질문: {question}

참고 문서:
{combined_context}

지침:
1. 문서 정보를 기반으로 정확하고 간결한 답변을 작성하세요.
2. 문서에 없는 정보는 '정보 없음'으로 명시하세요.
3. 전문 용어는 정확히 사용하고, 불필요한 설명은 생략하세요.
4. HTML 태그는 <b>, <ul>, <li>만 사용하세요.

답변:"""

    try:
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(llm_model.device)

            # 생성 파라미터 최적화
            outputs = await asyncio.wait_for(
                asyncio.to_thread(
                    llm_model.generate,
                    **inputs,
                    max_new_tokens=500,
                    repetition_penalty=1.2,
                    temperature=0.05,
                    do_sample=True,
                    top_p=0.95,
                    top_k=40,
                ),
                timeout=30.0,  # 30초 타임아웃
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 응답 추출
            if "답변:" in generated_text:
                answer = generated_text.split("답변:")[-1].strip()
            else:
                answer = generated_text.replace(prompt, "").strip()

            return answer
    except Exception as e:
        print(f"LLM 답변 생성 중 오류 발생: {e}")
        traceback.print_exc()
        return "답변 생성 중 오류가 발생했습니다."


# 검색 및 결합 함수
async def search_and_combine(
    es_client: Any,
    embedding_function: Any,
    reranker_model: Any,
    llm_model: Any,
    tokenizer: Any,
    query: str,
    category: str,
    conversation_history: List[Dict] = None,
) -> Dict[str, Any]:
    """Elasticsearch 검색, Reranking, LLM 답변 생성을 수행합니다."""

    # 대화 기록 초기화
    if conversation_history is None:
        conversation_history = []

    # 대화 기록 최적화 함수 정의
    def optimize_conversation_history(
        history: List[Dict], max_turns: int = 5
    ) -> List[Dict]:
        """대화 기록 최적화: 토큰 수 제한을 위해 최근 대화만 유지"""
        if len(history) > max_turns * 2:  # 사용자+봇 메시지를 한 턴으로 계산
            # 최근 대화만 유지
            return history[-max_turns * 2 :]
        return history

    # 대화 기록 최적화 적용
    if conversation_history:
        conversation_history = optimize_conversation_history(conversation_history)

    # 쿼리가 None인 경우 처리
    if query is None:
        return {"answer": "질문을 입력해주세요.", "sources": []}

    # 디버그 출력 추가
    print(f"--- DEBUG: search_and_combine ---")
    print(f"Original query (stripped): '{query}'")
    print(f"Is query empty? : {not query}")

    if not query:  # query.strip() 이후 query가 비어있는지 확인
        print("--- DEBUG: Query is empty, returning early. ---")  # 확인용 출력
        return {
            "answer": "질문 내용을 입력해주세요. 공백만으로는 검색할 수 없습니다.",
            "sources": [],
        }

    # 쿼리 전처리 및 정규화
    query = query.strip()

    if query is None:
        return {"answer": "질문을 입력해주세요.", "sources": []}

    # 쿼리 해시 생성 (캐싱용)
    query_hash = hashlib.md5(f"{query.lower()}:{category}".encode()).hexdigest()

    # 성능 측정 시작
    start_time = time.time()

    # 1. 검색
    retrieval_start = time.time()
    retriever = ElasticsearchRetriever(es_client, embedding_function, category=category)
    docs = retriever.get_relevant_documents(query)
    retrieval_time = time.time() - retrieval_start

    # 2. Reranking
    rerank_start = time.time()
    reranker = EnhancedLocalReranker(reranker_model)
    reranked_docs = reranker.rerank(query, docs)
    rerank_time = time.time() - rerank_start

    if not reranked_docs:
        return {
            "answer": "검색된 관련 문서가 없습니다. 다른 질문을 시도해 보세요.",
            "sources": [],
        }

    # 3. LLM 답변 생성
    llm_start = time.time()
    final_docs = reranked_docs[:5]  # LLM에 전달할 최종 문서 수

    print(f"--- DEBUG: Docs passed to LLM ---")
    for idx, doc in enumerate(final_docs):
        print(
            f"Doc {idx+1} (Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}) Content:\n{doc.page_content[:500]}..."
        )  # 앞부분 500자 정도만 출력
    print(f"---------------------------------")

    answer = await generate_llm_response(
        llm_model,
        tokenizer,
        query,
        final_docs,
        conversation_history=conversation_history,
    )
    llm_time = time.time() - llm_start

    # 출처 정보 생성
    sources = []
    source_contents = {}  # 원본 문서 내용 저장

    for i, doc in enumerate(final_docs):
        source_path = os.path.basename(doc.metadata.get("source", "N/A"))
        page_num = doc.metadata.get("page", 1)

        # 출처 정보
        source_info = {
            "path": source_path,
            "page": page_num,
            "similarity": doc.metadata.get("relevance_score", 0.0),
            "rerank_score": doc.metadata.get("rerank_score", 0.0),
            "rank": doc.metadata.get("rank", i + 1),
        }
        sources.append(source_info)

        # 원본 문서 내용 저장 (모달 표시용)
        source_key = f"{source_path}_{page_num}"
        source_contents[source_key] = doc.page_content

    # 총 처리 시간 계산
    total_time = time.time() - start_time

    return {
        "answer": answer,
        "sources": sources,
        "source_contents": source_contents,  # 원본 문서 내용 추가
    }


# 모델 초기화
es_client = get_elasticsearch_client()
embedding_function = get_embedding_function()
llm_model, tokenizer = get_llm_model_and_tokenizer()
reranker_model = get_reranker_model()


class FeedbackRequest(BaseModel):
    messageId: str
    feedbackType: str
    rating: int
    content: str


# 피드백 저장 디렉토리 설정
FEEDBACK_DIR = "app/feedback"
os.makedirs(FEEDBACK_DIR, exist_ok=True)
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.json")


# 피드백 저장 함수
def save_feedback(feedback_data: Dict[str, Any]):
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                feedbacks = json.load(f)
        else:
            feedbacks = []
        feedbacks.append(feedback_data)
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(feedbacks, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"피드백 저장 중 오류 발생: {e}")
        return False


class ConversationRequest(BaseModel):
    userId: str  # 사용자 식별자 (임시로 문자열 사용, 실제로는 인증 기반 ID)
    conversationId: str  # 대화 식별자
    messages: List[Dict[str, Any]]  # 대화 메시지 목록


class ConversationLoadRequest(BaseModel):
    userId: str
    conversationId: str


# 대화 저장 디렉토리 설정
CONVERSATION_DIR = "app/conversations"
os.makedirs(CONVERSATION_DIR, exist_ok=True)


# 대화 저장 함수
def save_conversation(
    user_id: str, conversation_id: str, messages: List[Dict[str, Any]]
):
    try:
        file_path = os.path.join(CONVERSATION_DIR, f"{user_id}_{conversation_id}.json")
        conversation_data = {
            "userId": user_id,
            "conversationId": conversation_id,
            "messages": messages,
            "timestamp": datetime.now().isoformat(),
        }
        with open(file_path, "w") as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"대화 저장 중 오류 발생: {e}")
        return False


# 대화 불러오기 함수
def load_conversation(user_id: str, conversation_id: str):
    try:
        file_path = os.path.join(CONVERSATION_DIR, f"{user_id}_{conversation_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"대화 불러오기 중 오류 발생: {e}")
        return None


# 대화 저장 엔드포인트
@app.post("/api/conversations/save")
async def save_conversation_endpoint(request: ConversationRequest = Body(...)):
    try:
        success = save_conversation(
            request.userId, request.conversationId, request.messages
        )
        if success:
            return {"status": "success", "message": "대화가 저장되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="대화 저장에 실패했습니다.")
    except Exception as e:
        print(f"대화 저장 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"대화 저장 중 오류 발생: {str(e)}")


# 대화 불러오기 엔드포인트
@app.post("/api/conversations/load")
async def load_conversation_endpoint(request: ConversationLoadRequest = Body(...)):
    try:
        conversation = load_conversation(request.userId, request.conversationId)
        if conversation:
            return {"status": "success", "conversation": conversation}
        else:
            raise HTTPException(status_code=404, detail="대화를 찾을 수 없습니다.")
    except Exception as e:
        print(f"대화 불러오기 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"대화 불러오기 중 오류 발생: {str(e)}"
        )


class UserSettingsRequest(BaseModel):
    userId: str  # 사용자 식별자 (임시로 문자열 사용, 실제로는 인증 기반 ID)
    settings: Dict[str, Any]  # 사용자 설정 데이터


class UserSettingsLoadRequest(BaseModel):
    userId: str


# 사용자 설정 저장 디렉토리 설정
SETTINGS_DIR = "app/settings"
os.makedirs(SETTINGS_DIR, exist_ok=True)


# 사용자 설정 저장 함수
def save_user_settings(user_id: str, settings: Dict[str, Any]):
    try:
        file_path = os.path.join(SETTINGS_DIR, f"{user_id}_settings.json")
        settings_data = {
            "userId": user_id,
            "settings": settings,
            "timestamp": datetime.now().isoformat(),
        }
        with open(file_path, "w") as f:
            json.dump(settings_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"사용자 설정 저장 중 오류 발생: {e}")
        return False


# 사용자 설정 불러오기 함수
def load_user_settings(user_id: str):
    try:
        file_path = os.path.join(SETTINGS_DIR, f"{user_id}_settings.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"사용자 설정 불러오기 중 오류 발생: {e}")
        return None


# 사용자 설정 저장 엔드포인트
@app.post("/api/settings/save")
async def save_user_settings_endpoint(request: UserSettingsRequest = Body(...)):
    try:
        success = save_user_settings(request.userId, request.settings)
        if success:
            return {"status": "success", "message": "사용자 설정이 저장되었습니다."}
        else:
            raise HTTPException(
                status_code=500, detail="사용자 설정 저장에 실패했습니다."
            )
    except Exception as e:
        print(f"사용자 설정 저장 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"사용자 설정 저장 중 오류 발생: {str(e)}"
        )


# 사용자 설정 불러오기 엔드포인트
@app.post("/api/settings/load")
async def load_user_settings_endpoint(request: UserSettingsLoadRequest = Body(...)):
    try:
        settings_data = load_user_settings(request.userId)
        if settings_data:
            return {"status": "success", "settings": settings_data["settings"]}
        else:
            return {
                "status": "not_found",
                "message": "사용자 설정을 찾을 수 없습니다.",
                "settings": {},
            }  # 파일이 없어도 에러 대신 빈 설정 반환
    except Exception as e:
        print(f"사용자 설정 불러오기 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"사용자 설정 불러오기 중 오류 발생: {str(e)}",
            "settings": {},
        }  # 500 대신 에러 메시지 반환


class SourcePreviewRequest(BaseModel):
    path: str  # 문서 경로 (예: tmpmn1xccd8.pdf)
    page: int  # 페이지 번호


# 참고 문서 미리보기 엔드포인트
@app.post("/api/source-preview")
async def source_preview_endpoint(request: SourcePreviewRequest = Body(...)):
    try:
        # Elasticsearch에서 해당 문서 내용 검색
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"source": request.path}},
                        {"term": {"page": request.page}},
                    ]
                }
            },
            "size": 1,
            "_source": ["text", "source", "page"],
        }
        response = es_client.search(index=ES_INDEX_NAME, body=search_body)
        hits = response["hits"]["hits"]
        if hits:
            hit = hits[0]
            content = hit["_source"].get("text", "내용을 찾을 수 없습니다.")
            return {
                "status": "success",
                "content": content,
                "source": hit["_source"].get("source", "N/A"),
                "page": hit["_source"].get("page", 0),
            }
        else:
            return {
                "status": "not_found",
                "message": "해당 문서를 찾을 수 없습니다.",
                "content": "",
                "source": request.path,
                "page": request.page,
            }
    except Exception as e:
        print(f"참고 문서 미리보기 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"참고 문서 미리보기 중 오류 발생: {str(e)}",
            "content": "",
            "source": request.path,
            "page": request.page,
        }


class StatsRequest(BaseModel):
    userId: str  # 사용자 식별자
    action: str  # 수행한 행동 (예: question, feedback, view_source)
    details: Dict[str, Any] = {}  # 추가 세부 정보 (예: 카테고리, 피드백 유형)


class StatsQueryRequest(BaseModel):
    userId: str = ""  # 특정 사용자 조회 (빈 문자열이면 전체 조회)
    startDate: str = ""  # 시작 날짜 (형식: YYYY-MM-DD)
    endDate: str = ""  # 종료 날짜 (형식: YYYY-MM-DD)


# 통계 저장 디렉토리 설정
STATS_DIR = "app/stats"
os.makedirs(STATS_DIR, exist_ok=True)
STATS_FILE = os.path.join(STATS_DIR, "stats.json")


# 통계 저장 함수
def save_stat(stat_data: Dict[str, Any]):
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                stats_list = json.load(f)
        else:
            stats_list = []
        stats_list.append(stat_data)
        with open(STATS_FILE, "w") as f:
            json.dump(stats_list, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"통계 저장 중 오류 발생: {e}")
        return False


# 통계 조회 함수
def query_stats(user_id: str = "", start_date: str = "", end_date: str = ""):
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE, "r") as f:
                stats_list = json.load(f)
        else:
            return []

        filtered_stats = stats_list
        if user_id:
            filtered_stats = [s for s in filtered_stats if s["userId"] == user_id]
        if start_date:
            filtered_stats = [s for s in filtered_stats if s["timestamp"] >= start_date]
        if end_date:
            filtered_stats = [
                s
                for s in filtered_stats
                if s["timestamp"] <= end_date + "T23:59:59.999999"
            ]
        return filtered_stats
    except Exception as e:
        print(f"통계 조회 중 오류 발생: {e}")
        return []


# 통계 저장 엔드포인트
@app.post("/api/stats/save")
async def save_stats_endpoint(request: StatsRequest = Body(...)):
    try:
        stat_data = {
            "userId": request.userId,
            "action": request.action,
            "details": request.details,
            "timestamp": datetime.now().isoformat(),
        }
        success = save_stat(stat_data)
        if success:
            return {"status": "success", "message": "통계가 저장되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="통계 저장에 실패했습니다.")
    except Exception as e:
        print(f"통계 저장 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"통계 저장 중 오류 발생: {str(e)}")


# 통계 조회 엔드포인트
@app.post("/api/stats/query")
async def query_stats_endpoint(request: StatsQueryRequest = Body(...)):
    try:
        stats = query_stats(request.userId, request.startDate, request.endDate)
        return {"status": "success", "stats": stats}
    except Exception as e:
        print(f"통계 조회 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류 발생: {str(e)}")


# 서버 시작 시 인덱스 확인
@app.on_event("startup")
async def startup_event():
    if not all([es_client, embedding_function, llm_model, tokenizer, reranker_model]):
        print("필수 리소스 로딩에 실패했습니다. 서버 로그를 확인하세요.")


# 파일 업로드 및 인덱싱 엔드포인트
@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),  # 다중 파일 지원
    category: str = Form("메뉴얼"),
):
    results = []
    for file in files:
        print(f"파일 업로드 요청 수신: {file.filename}, 카테고리: {category}")
        # 임시 파일 저장
        file_path = f"app/static/uploads/{uuid.uuid4()}_{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print(f"임시 파일 저장 완료: {file_path}")

        # 파일 처리 및 인덱싱
        try:
            print(f"파일 인덱싱 시작: {file.filename}")
            success = await process_and_index_file(
                es_client, embedding_function, file_path, category
            )
            if success:
                print(f"파일 인덱싱 성공: {file.filename}")
                results.append(
                    {
                        "filename": file.filename,
                        "status": "success",
                        "message": f"파일 '{file.filename}' 인덱싱 완료",
                    }
                )
            else:
                print(f"파일 인덱싱 실패: {file.filename}")
                results.append(
                    {
                        "filename": file.filename,
                        "status": "error",
                        "message": "파일 인덱싱 실패",
                    }
                )
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {file.filename}, 오류: {str(e)}")
            results.append(
                {
                    "filename": file.filename,
                    "status": "error",
                    "message": f"오류 발생: {str(e)}",
                }
            )
        finally:
            # 임시 파일 삭제
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"임시 파일 삭제 완료: {file_path}")

    print(f"업로드 결과: {len(results)}개 파일 처리 완료")
    return {"results": results}


# 질문-응답 엔드포인트
@app.post("/api/chat")
async def chat(request: QuestionRequest = Body(...)):
    try:
        result = await search_and_combine(
            es_client,
            embedding_function,
            reranker_model,
            llm_model,
            tokenizer,
            query=request.question,
            category=request.category,
            conversation_history=request.history,
        )

        return {"bot_response": result["answer"], "sources": result["sources"]}

    except Exception as e:
        print(f"챗봇 응답 생성 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"챗봇 응답 생성 중 오류 발생: {str(e)}"
        )


@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),  # 다중 파일 지원
    category: str = Form("메뉴얼"),
):
    results = []
    for file in files:
        print(f"파일 업로드 요청 수신: {file.filename}, 카테고리: {category}")
        # 임시 파일 저장
        file_path = f"app/static/uploads/{uuid.uuid4()}_{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print(f"임시 파일 저장 완료: {file_path}")

        # 파일 처리 및 인덱싱
        try:
            print(f"파일 인덱싱 시작: {file.filename}")
            success = await process_and_index_file(
                es_client, embedding_function, file_path, category
            )
            if success:
                print(f"파일 인덱싱 성공: {file.filename}")
                results.append(
                    {
                        "filename": file.filename,
                        "status": "success",
                        "message": f"파일 '{file.filename}' 인덱싱 완료",
                    }
                )
            else:
                print(f"파일 인덱싱 실패: {file.filename}")
                results.append(
                    {
                        "filename": file.filename,
                        "status": "error",
                        "message": "파일 인덱싱 실패",
                    }
                )
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {file.filename}, 오류: {str(e)}")
            results.append(
                {
                    "filename": file.filename,
                    "status": "error",
                    "message": f"오류 발생: {str(e)}",
                }
            )
        finally:
            # 임시 파일 삭제
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"임시 파일 삭제 완료: {file_path}")

    print(f"업로드 결과: {len(results)}개 파일 처리 완료")
    return {"results": results}


# 카테고리 목록 조회 엔드포인트
@app.get("/api/categories")
async def get_categories():
    try:
        query = {
            "size": 0,
            "aggs": {"categories": {"terms": {"field": "category", "size": 100}}},
        }

        result = es_client.search(index=ES_INDEX_NAME, body=query)
        categories = [
            bucket["key"] for bucket in result["aggregations"]["categories"]["buckets"]
        ]

        # 카테고리가 없으면 기본값 추가
        if not categories:
            categories = ["메뉴얼", "장애보고서"]

        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"카테고리 조회 실패: {str(e)}")


# 기본 라우트
@app.get("/")
async def root():
    return {"message": "RAG Chatbot API 서버가 실행 중입니다."}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    if exc.status_code == 405:
        print(
            f"405 Method Not Allowed: {request.method} 요청이 {request.url}로 수신됨",
            flush=True,
        )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


# 피드백 저장 엔드포인트
@app.post("/api/feedback")
async def save_feedback_endpoint(request: FeedbackRequest = Body(...)):
    try:
        feedback_data = {
            "messageId": request.messageId,
            "feedbackType": request.feedbackType,
            "rating": request.rating,
            "content": request.content,
            "timestamp": datetime.now().isoformat(),
        }
        success = save_feedback(feedback_data)
        if success:
            return {"status": "success", "message": "피드백이 저장되었습니다."}
        else:
            raise HTTPException(status_code=500, detail="피드백 저장에 실패했습니다.")
    except Exception as e:
        print(f"피드백 저장 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"피드백 저장 중 오류 발생: {str(e)}"
        )
