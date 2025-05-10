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
from app.utils.indexing_utils import process_and_index_file, ES_INDEX_NAME, check_file_exists, format_file_size
from fastapi.responses import FileResponse, StreamingResponse
import mimetypes  # 파일 타입 감지용


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
        
        # 토크나이저 로드 최적화: 병렬 처리 옵션 활성화
        tokenizer = AutoTokenizer.from_pretrained(
            LLM_MODEL_NAME,
            use_fast=True,  # 빠른 토크나이저 사용
            padding_side="left",  # 왼쪽 패딩 (생성 모델에 적합)
            use_auth_token=None,  # 인증 토큰 불필요 시 명시적으로 None
            trust_remote_code=True,  # 원격 코드 신뢰 (일부 모델에 필요)
        )
        
        # 특수 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 양자화 설정 고급 옵션 (4비트 정밀도 + offload)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True  # CPU offload 활성화
        )
        
        # 모델 로드 최적화 설정
        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16,
            "device_map": "auto",  # 자동 장치 맵핑
            "revision": "main",
            "low_cpu_mem_usage": True,  # 낮은 CPU 메모리 사용
            "attn_implementation": "flash_attention_2",  # Flash Attention 2 사용 (지원 시)
            "use_cache": True,  # KV 캐시 활성화
            "trust_remote_code": True,  # 원격 코드 신뢰
        }
        
        # 모델 로드 시도
        try:
            # 먼저 Flash Attention으로 로드 시도
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                **model_kwargs
            )
            print("LLM model loaded successfully with Flash Attention.")
        except Exception as flash_att_error:
            print(f"Flash Attention 로드 실패, 표준 방식으로 재시도: {flash_att_error}")
            # Flash Attention 실패 시 일반 방식으로 로드
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_NAME,
                **model_kwargs
            )
            print("LLM model loaded successfully with standard attention.")
        
        # 모델 최적화 설정 (추론 전용)
        model.eval()  # 평가 모드 설정
        
        # 모델 메모리 사용 정보 출력 (옵션)
        if torch.cuda.is_available():
            print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
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
    def __init__(self, es_client: Any, embedding_function: Any, category: str, k=25):
        self.es_client = es_client
        self.index_name = ES_INDEX_NAME
        self.embedding_function = embedding_function
        self.k = k
        self.category = category
        # 성능 최적화를 위한 캐시 추가
        self._cache = {}
        self._cache_size = 150  # 최대 캐시 항목 수 증가 (100 → 150)
        self._cache_ttl = 7200  # 캐시 유효 시간 증가 (3600 → 7200초, 2시간)

    def get_relevant_documents(self, query: str) -> List[Document]:
        if not self.es_client or not self.embedding_function:
            print("Elasticsearch 클라이언트 또는 임베딩 함수가 초기화되지 않았습니다.")
            return []

        # 캐시 키 생성 (쿼리와 카테고리 조합)
        query_normalized = query.lower().strip()
        cache_key = f"{query_normalized}:{self.category}"
        
        # 캐시에서 결과 확인
        current_time = time.time()
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if current_time - cache_entry["timestamp"] < self._cache_ttl:
                print(f"캐시에서 검색 결과 반환: '{query[:30]}...'")
                return cache_entry["results"]
        
        # 캐시 정리 (필요시) - LRU 방식 최적화
        if len(self._cache) >= self._cache_size:
            # 가장 오래된 항목부터 삭제 (LRU)
            oldest_keys = sorted(
                self._cache.keys(), 
                key=lambda k: self._cache[k]["timestamp"]
            )[:len(self._cache) // 3]  # 1/3 정도 삭제 (기존 1/4에서 증가)
            for old_key in oldest_keys:
                del self._cache[old_key]

        try:
            # 임베딩 생성
            query_embedding = self.embedding_function([query_normalized])[0]

            # 하이브리드 쿼리 구성 (BM25 + 벡터 검색) - 가중치 최적화
            hybrid_query = {
                "size": self.k,
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "bool": {
                        "should": [
                            # 1. 정확한 문구 검색 (가중치 상향)
                            {
                                "match_phrase": {
                                    "text": {"query": query, "boost": 3.5, "slop": 3}  # 3.0 → 3.5
                                }
                            },
                            # 2. BM25 키워드 검색 (가중치 상향)
                            {
                                "match": {
                                    "text": {
                                        "query": query,
                                        "boost": 2.5,  # 2.0 → 2.5
                                        "operator": "OR",
                                        "minimum_should_match": "60%",  # 50%에서 60%로 상향
                                    }
                                }
                            },
                            # 3. 벡터 검색 (가중치 상향)
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding},
                                    },
                                    "boost": 2.2,  # 가중치 상향 (1.5 → 2.2)
                                }
                            },
                        ],
                        "filter": [{"term": {"category": self.category}}],
                        "minimum_should_match": 1,
                    }
                },
            }

            # 검색 실행 (타임아웃 설정)
            response = self.es_client.search(
                index=self.index_name, 
                body=hybrid_query,
                request_timeout=30  # 30초 타임아웃
            )

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

                metadata["relevance_score"] = raw_score  # 원본 점수 유지
                metadata["source"] = hit["_source"].get("source", "unknown")
                metadata["page"] = hit["_source"].get("page", 1)
                
                # chunk_id가 정수형이면 문자열로 변환 (type 오류 방지)
                chunk_id = hit["_source"].get("chunk_id")
                if chunk_id is not None:
                    metadata["chunk_id"] = str(chunk_id)  # 명시적 문자열 변환

                # Document 객체 생성
                docs.append(
                    Document(
                        page_content=hit["_source"].get("text", ""), metadata=metadata
                    )
                )

            print(f"검색 완료: {len(docs)} 문서 검색됨")
            
            # 결과를 캐시에 저장
            self._cache[cache_key] = {
                "results": docs,
                "timestamp": current_time
            }
            
            return docs

        except Exception as e:
            print(f"Elasticsearch 검색 오류: {e}")
            traceback.print_exc()
            return []


# 향상된 리랭커 클래스 정의
class EnhancedLocalReranker:
    def __init__(self, reranker_model: Any, top_n=18):  # top_n 증가 (15 → 18)
        self.reranker = reranker_model
        self.top_n = top_n
        # 성능 최적화를 위한 캐시 추가
        self._cache = {}
        self._cache_size = 150  # 캐시 크기 증가 (100 → 150)
        self._cache_ttl = 7200  # 캐시 유효 시간 증가 (1시간 → 2시간)
        # 배치 처리 최적화
        self.batch_size = 24  # 배치 크기 증가 (16 → 24)

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs or not self.reranker:
            return []

        # 캐시 키 생성 (쿼리와 문서 ID 조합)
        # chunk_id를 명시적으로 문자열로 변환하여 에러 방지
        query_normalized = query.lower().strip()
        cache_key = f"{query_normalized}:{','.join([str(d.metadata.get('chunk_id', i)) for i, d in enumerate(docs[:10])])}"
        
        # 캐시에서 결과 확인
        current_time = time.time()
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if current_time - cache_entry["timestamp"] < self._cache_ttl:
                print(f"리랭킹 캐시 적중: '{query[:30]}...'")
                return cache_entry["results"]
                
        # 캐시 정리 (필요시) - LRU 방식 최적화
        if len(self._cache) >= self._cache_size:
            oldest_keys = sorted(
                self._cache.keys(), 
                key=lambda k: self._cache[k]["timestamp"]
            )[:len(self._cache) // 3]  # 1/3 정도 삭제 (1/4에서 증가)
            for old_key in oldest_keys:
                del self._cache[old_key]

        try:
            # 메모리 최적화를 위한 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 상위 12개 문서 리랭킹 (원래 10개에서 상향) → 12개 그대로 유지
            docs_to_rerank = docs[:12]
            pairs = [(query_normalized, doc.page_content) for doc in docs_to_rerank]
            
            # 배치 처리로 성능 최적화
            scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i:i + self.batch_size]
                # torch CUDA 설정으로 성능 최적화
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                    batch_scores = self.reranker.predict(batch_pairs)
                    scores.extend(batch_scores)

            # 메타데이터에 점수 추가 및 정규화
            for doc, score in zip(docs_to_rerank, scores):
                # 점수 범위를 0~1로 정규화 (-1~1 범위에서)
                normalized_score = min(max((score + 1) / 2, 0), 1)
                doc.metadata["rerank_score"] = float(normalized_score)
                doc.metadata["raw_rerank_score"] = float(score)  # 원본 점수도 저장

            # 리랭킹된 문서와 나머지 문서 결합
            sorted_docs = sorted(
                docs_to_rerank,
                key=lambda x: x.metadata.get("rerank_score", 0.0),
                reverse=True,
            )
            
            # 나머지 문서 추가 (이미 포함된 문서 제외)
            remaining_docs = [doc for doc in docs[12:] if doc not in docs_to_rerank]
            sorted_docs.extend(remaining_docs)

            # 임계값 필터링 - 점수가 낮은 문서 제외 (임계값 하향으로 더 많은 문서 포함)
            threshold = 0.52  # 임계값 하향 (0.6 → 0.52)
            filtered_docs = [
                doc
                for doc in sorted_docs
                if doc.metadata.get("rerank_score", 0.0) >= threshold
            ]

            # 필터링 결과가 최소 개수 미만이면 상위 문서 추가
            min_docs = 3  # 최소 3개 문서 보장
            if len(filtered_docs) < min_docs and sorted_docs:
                additional_docs = [
                    doc for doc in sorted_docs 
                    if doc not in filtered_docs
                ][:min_docs - len(filtered_docs)]
                filtered_docs.extend(additional_docs)

            # 결과 캐싱
            result_docs = filtered_docs[:self.top_n]
            self._cache[cache_key] = {
                "results": result_docs,
                "timestamp": current_time
            }
            
            return result_docs
            
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

    # 문서 컨텍스트 준비 - 최적화: 상위 7개 문서 사용 (기존 5개에서 증가)
    context_parts = []
    for i, doc in enumerate(top_docs[:7]):  # 상위 7개 문서 사용 (증가)
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page", "N/A")
        context_parts.append(
            f"[문서 {i+1}] (출처: {source}, 페이지: {page})\n{doc.page_content.strip()}"
        )

    combined_context = "\n\n".join(context_parts)

    # 이전 대화 기록 포맷팅 - 최적화: 최근 3개 턴으로 제한 (기존 5개에서 감소)
    conversation_context = ""
    if conversation_history:
        conversation_parts = []
        # 대화 기록의 최대 길이 제한 (최근 3개 턴)
        recent_history = (
            conversation_history[-3:]
            if len(conversation_history) > 3
            else conversation_history
        )

        for msg in recent_history:
            role = "사용자" if msg["role"] == "user" else "시스템"
            conversation_parts.append(f"{role}: {msg['content']}")

        conversation_context = "대화 기록:\n" + "\n".join(conversation_parts) + "\n\n"

    # 간결하고 명확한 프롬프트 - 최적화: 프롬프트 단순화
    prompt = f"""<start_of_turn>user
당신은 사용자 질문에 대해 주어진 참고 문서를 기반으로 답변하는 AI 어시스턴트입니다.
다음 지침을 따라주세요:
1. 답변은 반드시 제공된 '참고 문서' 섹션의 내용에 근거해야 합니다.
2. 문서에 질문과 관련된 정보가 없다면, "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 명확히 답변하세요.
3. 답변은 명확하고 간결하게 작성해주세요.
5. HTML 태그는 <b>, <ul>, <li>만 사용할 수 있습니다.

{conversation_context}
현재 질문: {question}

참고 문서:
{combined_context}
<end_of_turn>
<start_of_turn>model

답변:"""

    try:
        # 메모리 최적화를 위한 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):  # Mixed precision 활성화
            # 입력 인코딩 및 CUDA 장치로 이동
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(llm_model.device) for k, v in inputs.items()}

            # 생성 파라미터 최적화 - 안정성 및 속도 개선
            outputs = await asyncio.wait_for(
                asyncio.to_thread(
                    llm_model.generate,
                    **inputs,
                    max_new_tokens=512,  # 약간 증가 (500 → 512)
                    repetition_penalty=1.2,
                    temperature=0.01,  # 더 낮은 온도로 설정 - 더 결정적인 응답 (0.05 → 0.01)
                    do_sample=False,  # 결정적인 디코딩 사용
                    top_p=0.95,
                    top_k=40,
                    pad_token_id=tokenizer.eos_token_id,  # 패딩 토큰 명시적 설정
                    num_beams=1,  # 빔 서치 없이 빠른 생성
                ),
                timeout=60.0,  # 타임아웃 60초로 증가 (기존 45초)
            )

            # 효율적인 텍스트 디코딩 (skip_special_tokens=True)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 응답 추출 로직 최적화 - 정규식 사용
            if "<start_of_turn>model" in generated_text:
                answer = generated_text.split("<start_of_turn>model")[-1].strip()
                if answer.startswith("답변:"):
                    answer = answer[3:].strip()  # "답변:" 제거
            elif "답변:" in generated_text:
                answer = generated_text.split("답변:")[-1].strip()
            else:
                # 프롬프트에 응답 추가 부분 제거
                prompt_parts = prompt.split("<start_of_turn>model")[0]
                if prompt_parts in generated_text:
                    answer = generated_text.replace(prompt_parts, "").strip()
                else:
                    answer = generated_text

            # 빈 응답 처리
            if not answer.strip():
                answer = "죄송합니다, 답변을 생성하는 데 문제가 발생했습니다. 다시 질문해 주세요."
                
            return answer
    except asyncio.TimeoutError:
        print(f"LLM 답변 생성 시간 초과 (질문: {question[:50]}...)")
        return "답변 생성 시간이 초과되었습니다. 더 짧은 질문으로 다시 시도해 주세요."
    except Exception as e:
        print(f"LLM 답변 생성 중 오류 발생: {e}")
        traceback.print_exc()
        return "답변 생성 중 오류가 발생했습니다. 다시 시도해 주세요."


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

    if conversation_history is None:
        conversation_history = []

    # 대화 기록 최적화 (최근 5턴으로 제한)
    def optimize_conversation_history(
        history: List[Dict], max_turns: int = 3  # 최대 턴 수 감소 (5→3)
    ) -> List[Dict]:
        if len(history) > max_turns * 2:
            return history[-max_turns * 2 :]
        return history

    if conversation_history:
        conversation_history = optimize_conversation_history(conversation_history)

    # 기본 유효성 검사
    if query is None:
        return {"answer": "질문을 입력해주세요.", "sources": []}

    query = query.strip()
    if not query:
        print("--- DEBUG: Query is empty after strip, returning early. ---")
        return {
            "answer": "질문 내용을 입력해주세요. 공백만으로는 검색할 수 없습니다.",
            "sources": [],
        }

    print(f"--- DEBUG: search_and_combine ---")
    print(f"Processed query: '{query}'")

    start_time = time.time()

    try:
        # 메모리 최적화를 위한 토치 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 1. 검색 (비동기 처리)
        retrieval_start = time.time()
        # ElasticsearchRetriever는 k=25 (성능 최적화 설정)
        retriever = ElasticsearchRetriever(
            es_client, embedding_function, category=category, k=25
        )
        
        docs = retriever.get_relevant_documents(query)
        retrieval_time = time.time() - retrieval_start
        print(f"Retrieval time: {retrieval_time:.2f}s, Found {len(docs)} docs from ES.")

        # 검색 결과가 없을 경우 조기 반환
        if not docs:
            print("--- DEBUG: No documents found from retrieval. ---")
            return {
                "answer": "검색된 관련 문서가 없습니다. 다른 질문을 시도해 보세요.",
                "sources": [],
            }

        # 메모리 최적화를 위한 토치 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 2. Reranking (최적화 - 비동기 처리)
        rerank_start = time.time()
        # EnhancedLocalReranker는 top_n=18 (성능 최적화 설정)
        reranker = EnhancedLocalReranker(reranker_model, top_n=18)
        
        try:
            reranked_docs = reranker.rerank(query, docs)
            rerank_time = time.time() - rerank_start
            print(f"Reranking time: {rerank_time:.2f}s, Reranked to {len(reranked_docs)} docs.")
        except Exception as rerank_error:
            print(f"Reranking 중 오류 발생, 원본 문서 사용: {rerank_error}")
            traceback.print_exc()
            # 리랭킹 실패 시 원본 문서 사용
            reranked_docs = docs[:15]  # 상위 15개만 사용

        if not reranked_docs:
            print("--- DEBUG: No documents after reranking. ---")
            return {
                "answer": "검색된 관련 문서가 없습니다. 다른 질문을 시도해 보세요.",
                "sources": [],
            }

        # 메모리 최적화를 위한 토치 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 3. LLM 답변 생성
        llm_start = time.time()
        # final_docs 문서 수 증가 (5 → 7)
        final_docs = reranked_docs[:7]  # 더 많은 컨텍스트 제공
        print(f"--- DEBUG: Passing {len(final_docs)} docs to LLM. ---")
        for idx, doc_to_llm in enumerate(final_docs):
            print(
                f"  Doc for LLM {idx+1} - Source: {doc_to_llm.metadata.get('source')}, Page: {doc_to_llm.metadata.get('page')}, ChunkID: {doc_to_llm.metadata.get('chunk_id')}"
            )

        answer = await generate_llm_response(
            llm_model,
            tokenizer,
            query,
            final_docs,
            conversation_history=conversation_history,
        )
        llm_time = time.time() - llm_start
        print(f"LLM generation time: {llm_time:.2f}s")

        # --- 출처 정보 생성 ---
        sources = []
        print(
            f"--- DEBUG: Creating sources for frontend from {len(final_docs)} final_docs ---"
        )
        for i, doc_for_source in enumerate(final_docs):
            try:
                source_path = os.path.basename(doc_for_source.metadata.get("source", "unknown"))

                # 페이지 번호 처리 (기존 방식 유지)
                page_num_from_meta = doc_for_source.metadata.get("page")
                page_num_to_send = 1  # 기본값
                if page_num_from_meta is not None:
                    try:
                        page_num_to_send = int(page_num_from_meta)
                        if page_num_to_send <= 0:
                            print(
                                f"  WARNING (sources creation): Invalid page number {page_num_from_meta} for {source_path}. Defaulting to 1."
                            )
                            page_num_to_send = 1
                    except ValueError:
                        print(
                            f"  WARNING (sources creation): Page number {page_num_from_meta} is not an int for {source_path}. Defaulting to 1."
                        )
                        page_num_to_send = 1
                else:
                    print(
                        f"  WARNING (sources creation): 'page' key not found in metadata for {source_path}. Defaulting to page 1."
                    )
                    page_num_to_send = 1

                # 청크 ID 처리 - 명시적으로 문자열로 변환
                chunk_id_raw = doc_for_source.metadata.get("chunk_id", f"chunk_idx_{i}")
                chunk_id_to_send = str(chunk_id_raw)  # 명시적으로 문자열로 변환

                # 점수 정규화
                similarity = doc_for_source.metadata.get("relevance_score", 0.0)
                rerank_score = doc_for_source.metadata.get("rerank_score", 0.0)

                # 소스 정보 생성
                current_source_info = {
                    "path": source_path,
                    "page": page_num_to_send,
                    "chunk_id": chunk_id_to_send,
                    "content_full": doc_for_source.page_content,
                    "similarity": similarity,
                    "rerank_score": rerank_score,
                    "rank": i + 1,
                }
                sources.append(current_source_info)
            except Exception as source_error:
                print(f"소스 정보 생성 중 오류 발생 (문서 {i+1}): {source_error}")
                # 에러 발생해도 계속 진행

        total_time = time.time() - start_time
        print(
            f"Total search_and_combine time: {total_time:.2f}s, Returning {len(sources)} sources."
        )

        return {
            "answer": answer,
            "sources": sources,
        }
    
    except Exception as e:
        print(f"search_and_combine 중 예외 발생: {str(e)}")
        traceback.print_exc()
        return {
            "answer": "검색 및 답변 생성 중 오류가 발생했습니다. 다시 시도해 주세요.",
            "sources": [],
            "error": str(e),
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
    path: str
    page: int  # 페이지 정보는 여전히 유용할 수 있음 (UI 표시용)
    chunk_id: Any  # 문자열 또는 정수일 수 있으므로 Any (ES 저장 방식에 따라)


# 참고 문서 미리보기 엔드포인트
@app.post("/api/source-preview")
async def source_preview_endpoint(request: SourcePreviewRequest = Body(...)):
    try:
        print(
            f"Source preview 요청: path={request.path}, page={request.page}, chunk_id={request.chunk_id}"
        )

        # chunk_id 전처리 - 명시적 문자열 변환
        chunk_id_query = str(request.chunk_id)  # 모든 경우 문자열로 처리

        # 다양한 쿼리 시도 (우선순위에 따라)
        search_attempts = []
        
        # 1. source, page, chunk_id(문자열)로 검색 (1순위)
        search_attempts.append({
            "query": {
                "bool": {
                    "must": [
                        {"term": {"source": request.path}},
                        {"term": {"page": request.page}},
                        {"term": {"chunk_id": chunk_id_query}}
                    ]
                }
            }
        })
        
        # 2. chunk_id가 정수로 저장되었을 가능성 (2순위)
        try:
            chunk_id_int = int(request.chunk_id)
            search_attempts.append({
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"source": request.path}},
                            {"term": {"page": request.page}},
                            {"term": {"chunk_id": chunk_id_int}}
                        ]
                    }
                }
            })
        except (ValueError, TypeError):
            # 정수 변환 불가 시 이 시도는 건너뜀
            pass
            
        # 3. source와 page로만 검색 (3순위)
        search_attempts.append({
            "query": {
                "bool": {
                    "must": [
                        {"term": {"source": request.path}},
                        {"term": {"page": request.page}}
                    ]
                }
            },
            "size": 1,
            "sort": [{"chunk_id": "asc"}]  # 첫 번째 청크 선택
        })
        
        # 4. source_path만으로 검색 (4순위 - 마지막 시도)
        search_attempts.append({
            "query": {
                "term": {"source": request.path}}
            ,
            "size": 1
        })
        
        # 순차적으로 검색 시도
        for i, search_body in enumerate(search_attempts):
            # 기본 _source 필드 추가
            if "_source" not in search_body:
                search_body["_source"] = ["text", "source", "page", "chunk_id"]
                
            # 크기 지정이 없으면 기본값 설정
            if "size" not in search_body:
                search_body["size"] = 1
                
            response = es_client.search(index=ES_INDEX_NAME, body=search_body)
            hits = response["hits"]["hits"]
            
            if hits:
                hit = hits[0]["_source"]
                result_msg = f"검색 성공 (시도 {i+1}/{len(search_attempts)})"
                
                # 모든 필드가 문자열로 변환되도록 보장
                result = {
                    "status": "success",
                    "content": hit.get("text", "내용을 찾을 수 없습니다."),
                    "source": os.path.basename(hit.get("source", "N/A")),
                    "page": hit.get("page", 0),
                    "chunk_id": str(hit.get("chunk_id", "N/A")),  # 명시적 문자열 변환
                    "message": result_msg
                }
                
                # 첫 번째 시도가 아니라면 알림 메시지 추가
                if i > 0:
                    result["message"] = f"원본 chunk_id로 찾지 못해 대체 문서를 표시합니다. ({result_msg})"
                    
                return result

        # 모든 시도 실패
        return {
            "status": "not_found",
            "message": "해당 문서 내용을 찾을 수 없습니다.",
            "content": "",
            "source": os.path.basename(request.path),
            "page": request.page,
            "chunk_id": str(request.chunk_id),  # 명시적 문자열 변환
        }
    except Exception as e:
        print(f"참고 문서 미리보기 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"참고 문서 미리보기 중 오류 발생: {str(e)}",
            "content": "",
            "source": os.path.basename(request.path),
            "page": request.page,
            "chunk_id": str(request.chunk_id),  # 명시적 문자열 변환
        }


@app.get("/api/indexed-files")
async def get_indexed_files():
    """Elasticsearch에 인덱싱된 고유한 파일명 목록을 반환합니다."""
    if not es_client:
        raise HTTPException(status_code=503, detail="Elasticsearch is not connected")

    try:
        # Elasticsearch Terms Aggregation 쿼리
        # source 필드의 고유한 값을 모두 가져오기 위해 size를 충분히 크게 설정
        query = {
            "size": 0,  # 실제 문서는 가져오지 않음
            "aggs": {
                "unique_sources": {
                    "terms": {
                        "field": "source",  # source 필드 기준
                        "size": 10000,  # 충분히 큰 값 (예상되는 고유 파일 수 이상)
                    }
                }
            },
        }

        response = es_client.search(index=ES_INDEX_NAME, body=query)

        # Aggregation 결과에서 파일명 추출
        buckets = (
            response.get("aggregations", {})
            .get("unique_sources", {})
            .get("buckets", [])
        )
        file_list = [
            bucket.get("key") for bucket in buckets if bucket.get("key")
        ]  # key가 파일명

        print(
            f"Indexed file list requested. Found {len(file_list)} unique files."
        )  # 로그 추가
        return {"status": "success", "files": file_list}

    except Exception as e:
        print(f"인덱싱된 파일 목록 조회 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"파일 목록 조회 중 오류 발생: {str(e)}"
        )


# --- API 엔드포인트 추가 끝 ---


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


@app.get("/api/file-viewer/{filename}")
async def get_file_for_viewer(filename: str):
    # UUID가 포함된 전체 파일명을 사용한다고 가정
    # 보안: filename에 ../ 등이 포함되어 상위 디렉토리 접근 시도 방지
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # 파일이 저장된 실제 경로 (main.py 기준 상대 경로 또는 절대 경로)
    # /api/upload에서 저장한 경로와 동일해야 함
    file_path = os.path.join(STATIC_DIR, "uploads", filename)

    print(f"File view request for: {filename}, Path: {file_path}")

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    # 파일 타입(MIME) 추측
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # 기본값

    # Content-Disposition을 inline으로 설정하여 브라우저에서 바로 열도록 시도
    # (브라우저 설정에 따라 다운로드될 수도 있음)
    # 실제 파일명을 표시하려면 cleanFilename 로직을 여기서도 사용 가능
    # clean_name = filename[filename.find('_')+1:] if '_' in filename else filename
    # headers = {'Content-Disposition': f'inline; filename="{clean_name}"'}

    # FileResponse 사용하여 파일 내용 반환
    # return FileResponse(path=file_path, media_type=mime_type, headers=headers)
    # 간단히 파일 내용만 반환 (브라우저가 타입에 맞게 처리)
    return FileResponse(path=file_path, media_type=mime_type)


# 파일 업로드 및 인덱싱 엔드포인트
@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),  # 다중 파일 지원
    category: str = Form("메뉴얼"),  # 기본값을 메뉴얼로 설정
):
    results = []
    start_time = time.time()  # 전체 처리 시작 시간
    
    print(f"파일 업로드 요청 수신: {len(files)}개 파일, 카테고리: {category}")
    
    # 메모리 정리 - 초기 상태
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"업로드 처리 시작 - 초기 GPU 메모리 사용량: {initial_memory:.2f} MB")
    
    for file_index, file in enumerate(files):
        print(f"[{file_index+1}/{len(files)}] 파일 처리 중: {file.filename}, 카테고리: {category}")
        
        # 임시 파일 저장
        unique_id = uuid.uuid4()
        file_path = f"app/static/uploads/{unique_id}_{file.filename}"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # 파일 정보 및 초기 상태
            file_size = os.path.getsize(file_path)
            file_result = {
                "filename": file.filename,
                "unique_id": str(unique_id),
                "status": "processing",
                "message": f"파일 '{file.filename}' 처리 중...",
                "size": file_size,
                "start_time": time.time(),
                "progress": 0,  # 진행률 추가
                "file_info": {
                    "size_formatted": format_file_size(file_size),
                    "extension": os.path.splitext(file.filename)[1].lower(),
                    "index": file_index + 1,
                    "total": len(files)
                }
            }
            
            # 파일 처리 및 인덱싱
            try:
                print(f"파일 인덱싱 시작: {file.filename}")
                # 파일 중복 체크
                file_exists, file_hash = await check_file_exists(es_client, file_path)
                
                # 해시값 저장
                file_result["file_hash"] = file_hash[:8] + "..." if file_hash else None
                
                if file_exists:
                    # 중복 파일인 경우
                    file_result.update({
                        "status": "skipped",
                        "message": f"파일 '{file.filename}'은(는) 이미 인덱싱되어 있습니다.",
                        "processing_time": round(time.time() - file_result["start_time"], 2),
                        "duplicate": True,
                        "progress": 100
                    })
                else:
                    # 새 파일 처리 - 메모리 관리 강화
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        pre_process_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                        print(f"파일 처리 전 GPU 메모리: {pre_process_memory:.2f} MB")
                    
                    # 파일 처리 진행률 업데이트 (실제로는 비동기 처리가 필요할 수 있음)
                    file_result["progress"] = 30
                    
                    success = await process_and_index_file(
                        es_client, embedding_function, file_path, category
                    )
                    
                    # 메모리 사용량 확인
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        post_process_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                        memory_used = post_process_memory - pre_process_memory
                        print(f"파일 처리 후 GPU 메모리: {post_process_memory:.2f} MB (변화: {memory_used:.2f} MB)")
                    
                    if success:
                        print(f"파일 인덱싱 성공: {file.filename}")
                        file_result.update({
                            "status": "success",
                            "message": f"파일 '{file.filename}' 인덱싱 완료",
                            "processing_time": round(time.time() - file_result["start_time"], 2),
                            "progress": 100
                        })
                    else:
                        print(f"파일 인덱싱 실패: {file.filename}")
                        file_result.update({
                            "status": "error",
                            "message": "파일 인덱싱 실패",
                            "processing_time": round(time.time() - file_result["start_time"], 2),
                            "progress": 100
                        })
            except Exception as e:
                print(f"파일 처리 중 오류 발생: {file.filename}, 오류: {str(e)}")
                traceback.print_exc()
                file_result.update({
                    "status": "error",
                    "message": f"오류 발생: {str(e)}",
                    "processing_time": round(time.time() - file_result["start_time"], 2),
                    "error_details": str(e),
                    "progress": 100
                })
            
            results.append(file_result)
        except Exception as e:
            print(f"파일 저장 중 오류 발생: {file.filename}, 오류: {str(e)}")
            traceback.print_exc()
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"파일 저장 중 오류 발생: {str(e)}",
                "error_details": str(e),
                "progress": 100
            })
    
    # 최종 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"전체 처리 완료 - 최종 GPU 메모리 사용량: {final_memory:.2f} MB")
    
    # 전체 요약 통계 추가
    total_time = round(time.time() - start_time, 2)
    summary = {
        "total_files": len(results),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
        "skipped_count": sum(1 for r in results if r["status"] == "skipped"),
        "total_size": sum(r["size"] for r in results),
        "total_size_formatted": format_file_size(sum(r["size"] for r in results)),
        "total_processing_time": total_time,
        "average_file_time": round(total_time / len(results), 2) if results else 0
    }

    return JSONResponse(
        content={
            "status": "success" if summary["error_count"] == 0 else "partial_success",
            "message": f"{summary['total_files']}개 파일 처리 완료. {summary['success_count']}개 성공, {summary['error_count']}개 실패, {summary['skipped_count']}개 건너뜀",
            "results": results,
            "summary": summary,
        }
    )

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

        return {
            "bot_response": result["answer"],
            "sources": result["sources"],
            "questionContext": request.question,
        }

    except Exception as e:
        print(f"챗봇 응답 생성 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"챗봇 응답 생성 중 오류 발생: {str(e)}"
        )


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
