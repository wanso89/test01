from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import asyncio
import uuid
import time
import hashlib
from datetime import datetime
from elasticsearch import Elasticsearch
from app.utils.indexing_utils import process_and_index_file, ES_INDEX_NAME
import traceback

# 모델 임포트 (streamlit_chatbot.py에서 가져옴)
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import traceback; traceback.print_exc()

# 설정 상수
LLM_MODEL_NAME = r"/home/root/ko-gemma-v1"
EMBEDDING_MODEL_NAME = r"/home/root/KURE-v1"
RERANKER_MODEL_NAME = r"/home/root/ko-reranker"
ES_HOST = "http://localhost:9200"
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
            verify_certs=False
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
                                "filter": ["lowercase", "nori_part_of_speech"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "analyzer": "korean",
                            "search_analyzer": "korean"
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1024,
                            "index": True,
                            "similarity": "cosine",
                            "index_options": {
                                "type":"hnsw",
                                "m": 16,
                                "ef_construction": 100
                            }
                        },
                        "source": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "category": {"type": "keyword"},
                        "chunk_id": {"type": "keyword"},
                        "total_chunks": {"type": "integer"},
                        "indexed_at": {"type": "date"},
                        "image_path" : {"type" : "keyword"}
                    }
                }
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
                    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
                )
            
            def __call__(self, texts: list[str]) -> List[List[float]]:
                # embed_documents 메서드 사용
                return self.model.embed_documents(texts)
        
        embedding_func = LangchainEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        
        # 테스트 임베딩 실행
        try:
            test_result = embedding_func(["테스트"])
            print(f"Embedding model loaded successfully. Vector dimension: {len(test_result[0])}")
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
            RERANKER_MODEL_NAME, 
            device='cuda' if torch.cuda.is_available() else 'cpu'
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
                                    "text": {
                                        "query": query,
                                        "boost": 3.0,
                                        "slop": 2
                                    }
                                }
                            },
                            # 2. BM25 키워드 검색
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": 2.0,
                                        "operator": "OR",
                                        "minimum_should_match": "50%"
                                    }
                                }
                            },
                            # 3. 벡터 검색 (script_score)
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding}
                                    },
                                    "boost": 1.5
                                }
                            }
                        ],
                        "filter": [
                            {"term": {"category": self.category}}
                        ],
                        "minimum_should_match": 1
                    }
                }
            }
            
            # 검색 실행
            response = self.es_client.search(
                index=self.index_name,
                body=hybrid_query
            )
            
            # 결과 처리
            docs = []
            for hit in response['hits']['hits']:
                # 메타데이터 추출 (embedding 필드 제외)
                metadata = {}
                for k, v in hit['_source'].items():
                    if k != 'text' and k != 'embedding':
                        metadata[k] = v

                # 점수 정규화 (0~1 사이로)
                raw_score = hit['_score']
                # Elasticsearch 스코어는 범위가 다양하므로 정규화
                normalized_score = min(max(raw_score / 10, 0), 1)  # 10으로 나누어 0~1 범위로 조정
              
                metadata["relevance_score"] = hit['_score']
                metadata["source"] = hit['_source'].get('source', 'unknown')
                metadata["page"] = hit['_source'].get('page', 1)
                
                # Document 객체 생성
                docs.append(Document(
                    page_content=hit['_source'].get('text', ''),
                    metadata=metadata
                ))
            
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
            sorted_docs = sorted(docs_to_rerank, key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
            sorted_docs.extend([doc for doc in docs[10:] if doc not in docs_to_rerank])
            
            threshold = 0.6
            filtered_docs = [doc for doc in sorted_docs if doc.metadata.get("rerank_score", 0.0) >= threshold]
            
            if not filtered_docs and sorted_docs:
                filtered_docs = [sorted_docs[0]]
            
            return filtered_docs[:self.top_n]
        except Exception as e:
            print(f"Reranking 중 오류 발생: {e}")
            traceback.print_exc()
            return docs

# LLM 답변 생성 함수
async def generate_llm_response(llm_model: Any, tokenizer: Any, question: str, top_docs: List[Document], temperature: float = 0.2, conversation_history=None) -> str:
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
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', 'N/A')
        context_parts.append(f"[문서 {i+1}] (출처: {source}, 페이지: {page})\n{doc.page_content.strip()}")
    
    combined_context = "\n\n".join(context_parts)

    # 이전 대화 기록 포맷팅
    conversation_context = ""
    if conversation_history:
        conversation_parts = []
        # 대화 기록의 최대 길이 제한 (예: 최근 5개 턴)
        recent_history = conversation_history[-5:] if len(conversation_history) > 5 else conversation_history
        
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
                    top_k=40
                ),
                timeout=30.0  # 30초 타임아웃
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
async def search_and_combine(es_client: Any, embedding_function: Any, reranker_model: Any, llm_model: Any, tokenizer: Any, query: str, category: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
    """Elasticsearch 검색, Reranking, LLM 답변 생성을 수행합니다."""

    # 대화 기록 초기화
    if conversation_history is None:
        conversation_history = []

    # 대화 기록 최적화 함수 정의
    def optimize_conversation_history(history: List[Dict], max_turns: int = 5) -> List[Dict]:
        """대화 기록 최적화: 토큰 수 제한을 위해 최근 대화만 유지"""
        if len(history) > max_turns * 2:  # 사용자+봇 메시지를 한 턴으로 계산
            # 최근 대화만 유지
            return history[-max_turns * 2:]
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

    if not query: # query.strip() 이후 query가 비어있는지 확인
        print("--- DEBUG: Query is empty, returning early. ---") # 확인용 출력
        return {"answer": "질문 내용을 입력해주세요. 공백만으로는 검색할 수 없습니다.", "sources": []}

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
        return {"answer": "검색된 관련 문서가 없습니다. 다른 질문을 시도해 보세요.", "sources": []}
    
    # 3. LLM 답변 생성
    llm_start = time.time()
    final_docs = reranked_docs[:5]  # LLM에 전달할 최종 문서 수

    print(f"--- DEBUG: Docs passed to LLM ---")
    for idx, doc in enumerate(final_docs):
        print(f"Doc {idx+1} (Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}) Content:\n{doc.page_content[:500]}...") # 앞부분 500자 정도만 출력
    print(f"---------------------------------")

    answer = await generate_llm_response(llm_model, tokenizer, query, final_docs, conversation_history=conversation_history)
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
            "rank": doc.metadata.get("rank", i+1)
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
        "source_contents": source_contents  # 원본 문서 내용 추가
    }

# 모델 초기화
es_client = get_elasticsearch_client()
embedding_function = get_embedding_function()
llm_model, tokenizer = get_llm_model_and_tokenizer()
reranker_model = get_reranker_model()

# 서버 시작 시 인덱스 확인
@app.on_event("startup")
async def startup_event():
    if not all([es_client, embedding_function, llm_model, tokenizer, reranker_model]):
        print("필수 리소스 로딩에 실패했습니다. 서버 로그를 확인하세요.")

# 파일 업로드 및 인덱싱 엔드포인트
@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form("메뉴얼")
):
    # 임시 파일 저장
    file_path = f"app/static/uploads/{uuid.uuid4()}_{file.filename}"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # 파일 처리 및 인덱싱
    try:
        success = process_and_index_file(
            es_client, 
            embedding_function, 
            file_path, 
            category
        )
        
        if success:
            return {"status": "success", "message": f"파일 '{file.filename}' 인덱싱 완료"}
        else:
            raise HTTPException(status_code=500, detail="파일 인덱싱 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")
    finally:
        # 임시 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

# 질문-응답 엔드포인트
@app.post("/api/chat")
async def chat(request: QuestionRequest):
    print("API 호출됨 - 요청 수신 시작작", flush=True)
    try:
        print("요청 내용:", request, flush=True)
        question = request.question
        category = request.category
        history = request.history
        print(f"question: {question}, category: {category}", flush=True)

        conversation_history = [{"role": msg["role"], "content": msg["content"]} for msg in history]

        # 캐시 대신 항상 직접 검색
        result = await asyncio.wait_for(
            search_and_combine(
                es_client,
                embedding_function,
                reranker_model,
                llm_model,
                tokenizer,
                query=question,
                category=category,
                conversation_history=conversation_history
            ),
            timeout=60.0  # 전체 요청에 60초 타임아웃 설정
        )
        print("search_and_combine 결과:", result, flush=True)
        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except asyncio.TimeoutError:
        print("요청 처리 시간이 초과되었습니다.", flush=True)
        raise HTTPException(status_code=504, detail="요청 처리 시간이 초과되었습니다.")
    except Exception as e:
        print("API 오류:", e, flush=True)
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"질문 처리 중 오류 발생: {str(e)}")


# 카테고리 목록 조회 엔드포인트
@app.get("/api/categories")
async def get_categories():
    try:
        query = {
            "size": 0,
            "aggs": {
                "categories": {
                    "terms": {"field": "category", "size": 100}
                }
            }
        }
        
        result = es_client.search(index=ES_INDEX_NAME, body=query)
        categories = [bucket["key"] for bucket in result["aggregations"]["categories"]["buckets"]]
        
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
        print(f"405 Method Not Allowed: {request.method} 요청이 {request.url}로 수신됨", flush=True)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )
