import os
import asyncio
import uuid
import time
import hashlib
import json
import traceback
import difflib  # 유사도 비교를 위한 표준 라이브러리

# CUDA 메모리 관리 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Depends, Request, Query, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import logging
import random
from enum import Enum
from elasticsearch import Elasticsearch
from app.utils.indexing_utils import process_and_index_file, ES_INDEX_NAME, check_file_exists, format_file_size
from fastapi.responses import FileResponse, StreamingResponse
import mimetypes  # 파일 타입 감지용
from collections import Counter  # 피드백 통계용
import re  # 정규식 사용을 위한 모듈
from pathlib import Path  # 경로 처리용

# 검색 개선 모듈 import
from app.utils.search_enhancer import EnhancedSearchPipeline
# 피드백 분석 모듈 import
from app.utils.feedback_analyzer import FeedbackAnalyzer, SearchQualityOptimizer
# 파일 관리 모듈 import
from app.utils.file_manager import delete_indexed_file, find_file_by_name

# 모델 임포트
import torch
from torch.cuda.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import traceback
from threading import Thread

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 파일 핸들러
if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
    fh = logging.FileHandler("app.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

# 스트림 핸들러
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

# traceback.print_exc() # 애플리케이션 시작 시 불필요한 traceback 제거

# 설정 상수
# LLM_MODEL_NAME = r"/home/root/ko-gemma-v1"
#EMBEDDING_MODEL_NAME = r"/home/root/KURE-v1"
# RERANKER_MODEL_NAME = r"/home/root/ko-reranker"
#LLM_MODEL_NAME = r"/home/root/Qwen3-8B"
LLM_MODEL_NAME = r"/home/root/Gukbap-Qwen2.5-7B"
EMBEDDING_MODEL_NAME = r"/home/root/kpf-sbert-v1.1"
#RERANKER_MODEL_NAME = r"/home/root/bge-reranker-large"
RERANKER_MODEL_NAME = r"/home/root/bge-reranker-v2-m3"
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
    """임베딩 기능을 제공하는 함수를 반환합니다."""
    print("Loading embedding function...")
    try:
        # HuggingFace 임베딩 커스텀 래퍼 클래스
        from app.utils.cache_utils import cache_embeddings

        class LangchainEmbeddingFunction:
            def __init__(self, model_name: str):
                # HuggingFaceEmbeddings 직접 사용
                try:
                    # 빠른 토크나이저 및 장치 최적화 설정 추가
                    self.embeddings_model = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
                        encode_kwargs={"normalize_embeddings": True, "batch_size": 8},  # 배치 크기 제한
                        cache_folder="./.cache",  # 명시적 캐시 폴더 설정
                        multi_process=False  # 임베딩 병렬 처리 비활성화
                    )
                    print(f"임베딩 모델 로드 성공: {model_name}")
                except Exception as e:
                    print(f"임베딩 모델 로드 실패, CPU 버전으로 재시도: {e}")
                    self.embeddings_model = HuggingFaceEmbeddings(
                        model_name=model_name, 
                        model_kwargs={"device": "cpu"}
                    )
            
            # 캐싱 데코레이터 제거 - 올바른 위치로 이동
            def __call__(self, texts: list[str]) -> List[List[float]]:
                # embed_documents 메서드 사용
                embeddings = []
                try:
                    # 수정: 임베딩 생성 전에 타입 확인
                    if not isinstance(texts, list):
                        print(f"경고: texts가 리스트가 아님 - 타입: {type(texts)}")
                        if isinstance(texts, str):
                            texts = [texts]
                        else:
                            return [[0.0] * 768]  # 타입 오류 시 기본값 반환
                            
                    # 빈 입력 처리
                    if not texts:
                        return []
                        
                    # 임베딩 생성 전 각 텍스트 항목이 문자열인지 확인
                    for i, text in enumerate(texts):
                        if not isinstance(text, str):
                            print(f"경고: texts[{i}]가 문자열이 아님 - 타입: {type(text)}")
                            texts[i] = str(text)
                    
                    # 메모리 효율적인 임베딩 생성 (배치 처리로 변경)
                    import numpy as np
                    batch_size = 8  # 작은 배치 크기로 설정
                    
                    # 배치 단위로 처리
                    embeddings = []
                    for i in range(0, len(texts), batch_size):
                        batch_texts = texts[i:i+batch_size]
                        if torch.cuda.is_available():
                            # 메모리 정리
                            torch.cuda.empty_cache()
                        batch_embeddings = self.embeddings_model.embed_documents(batch_texts)
                        embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"임베딩 생성 오류: {e}")
                    traceback.print_exc()
                    # 오류 시 빈 임베딩 반환 (각 768차원)
                    embeddings = [[0.0] * 768 for _ in range(len(texts))]
                return embeddings

        # 클래스 인스턴스 생성 및 반환
        return LangchainEmbeddingFunction(EMBEDDING_MODEL_NAME)
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

        # Qwen2.5 모델에 최적화된 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # float16 -> bfloat16로 변경 (Qwen 최적화)
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=False  # A6000 환경에서는 비활성화하는 것이 더 효율적
        )

        # 모델 로드 최적화 설정
        model_kwargs = {
            "quantization_config": quantization_config,
            "torch_dtype": torch.bfloat16,  # float16 -> bfloat16로 변경 (Qwen 최적화)
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
        # 피드백 기반 검색 최적화 도구 초기화
        self.search_optimizer = SearchQualityOptimizer()

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
            query_normalized = query.lower().strip()
            try:
                # 임베딩 함수 호출 시 오류 방지를 위한 타입 확인
                if callable(self.embedding_function):
                    query_embedding = self.embedding_function([query_normalized])[0]
                else:
                    print("경고: 임베딩 함수가 호출 가능하지 않음")
                    return []
            except Exception as embed_error:
                print(f"임베딩 생성 중 오류 발생: {embed_error}")
                traceback.print_exc()
                return []  # 임베딩 실패 시 빈 결과 반환

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

            # 피드백 기반 쿼리 최적화 적용
            try:
                optimized_query = self.search_optimizer.apply_optimizations_to_query(
                    query_normalized, hybrid_query
                )
                if optimized_query != hybrid_query:
                    print(f"피드백 기반 쿼리 최적화 적용됨: '{query[:30]}...'")
                    hybrid_query = optimized_query
            except Exception as optimize_error:
                print(f"쿼리 최적화 적용 중 오류: {optimize_error}")
                # 최적화 오류 시 원본 쿼리 사용

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

            # 결과 캐싱
            self._cache[cache_key] = {
                "results": docs,
                "timestamp": current_time,
            }

            return docs
        except Exception as e:
            print(f"Elasticsearch 검색 중 오류 발생: {e}")
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
    tokenizer_for_template_application: Any,
    question: str,
    top_docs: List[Document],
    temperature: float = 0.2,
    conversation_history=None,
) -> dict:  # 반환 타입을 str에서 dict로 변경
    logger.debug("Generating final prompt for LLM.")
    # 1. 대화 기록과 현재 질문, 검색된 문서를 바탕으로 프롬프트 구성
    context_str = "\\n\\n".join([f"문서 {i+1}: {doc.page_content}" for i, doc in enumerate(top_docs)])
    
    # Qwen ChatML 형식에 맞게 프롬프트 생성
    # 실제 tokenizer.apply_chat_template 사용을 권장하며, 아래는 그 예시입니다.
    # messages 구성 (conversation_history가 None일 경우 빈 리스트로 초기화)
    messages = []
    if conversation_history:
        for entry in conversation_history:
            # role과 content 키가 있는지 확인
            if isinstance(entry, dict) and "role" in entry and "content" in entry:
                messages.append({"role": entry["role"], "content": entry["content"]})
            else:
                logger.warning(f"Invalid entry in conversation_history: {entry}")

    # 시스템 메시지 추가
    system_message = f"""You are a helpful AI assistant. Answer the questions based on the provided documents.
If the information is not in the documents, say that you cannot answer.
Provided documents:
{context_str}"""
    messages.insert(0, {"role": "system", "content": system_message})
    
    # 사용자 질문 추가
    messages.append({"role": "user", "content": question})

    try:
        # tokenizer_for_template_application (원래 tokenizer)를 사용하여 프롬프트 템플릿 적용
        final_prompt_text = tokenizer_for_template_application.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # assistant 응답을 유도
        )
    except Exception as e:
        logger.error(f"Error applying chat template: {e}")
        # 템플릿 적용 실패 시 기본 폴백 프롬프트 (매우 단순화된 버전)
        final_prompt_text = f"System: {system_message}\\nUser: {question}\\nAssistant:"

    logger.debug(f"Generated final_prompt_text (first 100 chars): {final_prompt_text[:100]}")
    
    # 문서 출처 정보 추출
    source_metadata = []
    for i, doc in enumerate(top_docs):
        # 소스 정보 URL 인코딩
        source_path = doc.metadata.get("source", "unknown")
        page_num = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", i)

        # 파일명에서 UUID 제거 (UUID_파일명.확장자 형식 가정)
        clean_filename = os.path.basename(source_path)
        # UUID_ 패턴 감지 (UUID는 일반적으로 8-4-4-4-12 형식의 16진수 문자)
        if '_' in clean_filename:
            uuid_parts = clean_filename.split('_', 1)
            if len(uuid_parts) > 1 and len(uuid_parts[0]) >= 8:  # UUID로 추정되는 부분이 있으면 제거
                clean_filename = uuid_parts[1]

        source_metadata.append({
            "path": source_path,
            "display_name": clean_filename,  # 화면 표시용 정제된 파일명 추가
            "page": page_num,
            "chunk_id": chunk_id,
            "score": doc.metadata.get("relevance_score", 0),
        })
    
    # 프롬프트 텍스트와 소스 메타데이터를 함께 반환
    return {
        "prompt_text": final_prompt_text,
        "source_metadata": source_metadata,
        "top_docs": top_docs  # 문서 전체 내용도 함께 반환 (인용 감지용)
    }


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
    
    # Redis 캐싱 적용: 동일한 쿼리의 중복 처리 방지 - 성능 대폭 개선
    from app.utils.cache_utils import RedisCache, CacheKeys, CACHE_TTL_SEARCH
    
    # 캐시 키 생성 (질문 + 카테고리 기반)
    cache_key = RedisCache.generate_key(
        CacheKeys.CHAT, 
        {"query": query, "category": category}
    )
    
    # 캐시에서 결과 확인
    cached_result = RedisCache.get(cache_key)
    if cached_result:
        print(f"캐시된 결과 사용: {cache_key}")
        # 캐시된 결과에 성능 메트릭 추가
        cached_result["from_cache"] = True
        cached_result["processing_time"]["cache_hit"] = round(time.time() - start_time, 3)
        
        # 캐시된 결과를 스트리밍 방식으로 반환
        async def cached_response_stream():
            # 캐시된 텍스트 응답을 작은 청크로 나누어 스트리밍
            answer_text = cached_result.get("answer", "")
            sources = cached_result.get("sources", [])
            cited_sources = cached_result.get("cited_sources", [])
            
            # 텍스트를 토큰 단위로 스트리밍 (여기서는 간단히 문자 단위로 나눔)
            chunk_size = 4  # 4자씩 스트리밍 (실제 토큰 크기와 유사하게)
            for i in range(0, len(answer_text), chunk_size):
                chunk = answer_text[i:i+chunk_size]
                yield f"data: {json.dumps({'token': chunk})}\n\n"
                await asyncio.sleep(0.01)  # 실제 스트리밍 효과를 위한 짧은 지연
            
            # 소스 정보 전송
            yield f"data: {json.dumps({'event': 'sources', 'sources': sources, 'cited_sources': cited_sources})}\n\n"
            
            # 캐시 사용 정보 전송 (클라이언트에서 캐시 사용 여부 표시 가능)
            yield f"data: {json.dumps({'event': 'cache_info', 'from_cache': True})}\n\n"
            
            # 스트림 종료 이벤트
            yield f"data: {json.dumps({'event': 'eos', 'message': 'Stream ended (from cache).'})}\n\n"
        
        # 캐시된 응답을 스트리밍 형태로 반환
        return StreamingResponse(
            cached_response_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )
        
        # 기존 코드 (한 번에 반환)
        # return cached_result

    # 검색 개선 파이프라인 초기화
    search_pipeline = EnhancedSearchPipeline()

    try:
        # 메모리 최적화를 위한 토치 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 1. 검색 (비동기 처리) - 검색 결과 수 최적화
        retrieval_start = time.time()
        # ElasticsearchRetriever 검색 결과 수 감소 (25 → 10): 정확도는 유지하면서 처리 속도 향상
        retriever = ElasticsearchRetriever(
            es_client, embedding_function, category=category, k=10
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

        # 1.5. 검색 개선 파이프라인 적용 (쿼리 확장, 점수 보정, 컨텍스트 리랭킹)
        enhance_start = time.time()
        # 검색 개선 파이프라인 실행
        try:
            query_info, enhanced_docs = search_pipeline.process(query, docs)

            if enhanced_docs:
                docs = enhanced_docs
                print(f"Enhanced search with query variants: {', '.join(query_info['variants'])}")
            else:
                print("Enhanced search returned no results, using original docs")
        except Exception as enhance_error:
            print(f"검색 개선 적용 중 오류 발생: {enhance_error}")
            traceback.print_exc()
            # 개선 실패 시 원본 문서 사용

        enhance_time = time.time() - enhance_start
        print(f"Search enhancement time: {enhance_time:.2f}s")

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

        # 최종 토큰 수 제한
        # 컨텍스트 크기를 최대 3,500 토큰으로 제한
        max_tokens = 3500
        token_count = 0
        final_docs = []

        for doc in reranked_docs:
            if not doc.page_content:
                continue

            # 입력에 대해 예상 토큰 수 계산 (한국어 토큰화는 복잡하므로 근사치 사용)
            approx_tokens = len(doc.page_content) / 3
            if token_count + approx_tokens > max_tokens:
                break

            token_count += approx_tokens
            final_docs.append(doc)

        # LLM 입력 형식으로 변환
        # 리랭킹 결과 문서들을 하나의 컨텍스트로 결합
        context_chunks = []
        source_metadata = []

        for i, doc in enumerate(final_docs):
            # 소스 정보 URL 인코딩
            source_path = doc.metadata.get("source", "unknown")
            page_num = doc.metadata.get("page", 1)
            chunk_id = doc.metadata.get("chunk_id", i)

            # 텍스트에서 불필요한 공백과 개행 정리
            chunk_text = doc.page_content.strip()
            chunk_text = " ".join(chunk_text.split())

            # 파일명에서 UUID 제거 (UUID_파일명.확장자 형식 가정)
            clean_filename = os.path.basename(source_path)
            # UUID_ 패턴 감지 (UUID는 일반적으로 8-4-4-4-12 형식의 16진수 문자)
            if '_' in clean_filename:
                uuid_parts = clean_filename.split('_', 1)
                if len(uuid_parts) > 1 and len(uuid_parts[0]) >= 8:  # UUID로 추정되는 부분이 있으면 제거
                    clean_filename = uuid_parts[1]

            if chunk_text:
                context_chunks.append(f"[문서 {i+1}] {chunk_text}")
                source_metadata.append({
                        "path": source_path,
                    "display_name": clean_filename,  # 화면 표시용 정제된 파일명 추가
                        "page": page_num,
                        "chunk_id": chunk_id,
                        "score": doc.metadata.get("relevance_score", 0),
                })

        full_context = "\n\n".join(context_chunks)
        print(f"Combined context length: {len(full_context)} characters.")

        # LLM으로 답변 생성
        llm_start = time.time()
        answer = await generate_llm_response(
            tokenizer,
            query,
            final_docs,
            0.2,
            conversation_history
        )
        
        # 응답이 None인 경우 대체 응답 사용 (방어 코드)
        if answer is None:
            print("LLM 응답이 None입니다. 대체 응답을 사용합니다.")
            answer = "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다. 다시 질문해 주세요."
        
        # 응답 정제 - Qwen의 ChatML 포맷 처리
        def clean_response(resp):
            """응답 텍스트에서 불필요한 시스템 메시지, 사용자 메시지 등을 제거합니다."""
            if not resp:
                return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
                
            # 1. 시스템 메시지 제거
            if resp.startswith("system\n"):
                parts = resp.split("user\n")
                if len(parts) > 1:
                    resp = parts[1]
                    parts = resp.split("assistant\n")
                    if len(parts) > 1:
                        resp = parts[1].strip()
                        return resp
            
            # 2. assistant 접두사 제거
            if "assistant\n" in resp:
                parts = resp.split("assistant\n")
                if len(parts) > 1:
                    resp = parts[-1].strip()
                    return resp
            
            # 3. 그 외의 경우
            # 불필요한 태그 제거
            for tag in ["system\n", "user\n", "assistant\n", "system:", "user:", "assistant:", "system", "user", "assistant"]:
                if resp.startswith(tag):
                    resp = resp[len(tag):].strip()
                    
            return resp.strip()
            
        # 응답 정제 적용
        cleaned_answer = clean_response(answer)
        print(f"원본 응답 시작 부분: {answer[:50]}...")
        print(f"정제된 응답 시작 부분: {cleaned_answer[:50]}...")
        
        # 소스 텍스트가 LLM 출력에 직접 인용된 경우를 확인
        cited_sources = []
        
        # 응답 처리 - 정제된 응답 사용
        if not cleaned_answer or cleaned_answer.strip() == "":
            print("정제된 응답이 비어있어 원본 응답을 사용합니다.")
            cleaned_answer = "안녕하세요! 어떻게 도와드릴까요?"
        
        # 인용 감지를 위한 간단한 키워드 추출
        def extract_keywords(text, min_length=3, max_keywords=20):
            # None이나 비문자열 체크
            if text is None or not isinstance(text, str):
                print(f"extract_keywords: 유효하지 않은 입력 타입 - {type(text)}")
                return []
                
            # 특수문자, 공백 등을 기준으로 단어 분리
            words = re.findall(r'\b[가-힣a-zA-Z0-9]+\b', text)
            # 길이가 min_length 이상인 단어만 필터링
            filtered_words = [w for w in words if len(w) >= min_length]
            # 중복 제거 및 최대 개수 제한
            unique_words = list(set(filtered_words))[:max_keywords]
            return unique_words
            
        # 응답에서 키워드 추출
        answer_keywords = extract_keywords(cleaned_answer)
        
        # answer가 None이 아닌 경우만 인용 처리 진행 
        if cleaned_answer and isinstance(cleaned_answer, str):
            for i, meta in enumerate(source_metadata):
                cited = False
                source_text = context_chunks[i] if i < len(context_chunks) else ""
    
                # 1. 기존 방식: 연속된 텍스트 일치 여부 확인 (최소 30자)
                if len(source_text) > 50:
                    for j in range(0, len(source_text) - 30, 10):
                        snippet = source_text[j:j+30]
                        if snippet in cleaned_answer:
                            cited = True
                            break
    
                # 2. 개선된 방식: 키워드 기반 매칭 (기존 방식으로 감지되지 않은 경우)
                if not cited and source_text:
                    # 소스에서 키워드 추출
                    source_keywords = extract_keywords(source_text)
                    # 키워드 일치율 계산
                    if source_keywords:
                        matches = [k for k in source_keywords if k in cleaned_answer]
                        match_ratio = len(matches) / len(source_keywords)
                        # 키워드의 30% 이상이 응답에 포함되어 있으면 인용으로 간주
                        if match_ratio > 0.3:
                            cited = True
                
                # 메타데이터에 인용 여부 저장
                meta["is_cited"] = cited
                if cited:
                    cited_sources.append(meta)
        else:
            print("유효한 응답이 없어 인용 처리를 건너뜁니다.")
            # 모든 메타데이터에 is_cited = False 설정
            for meta in source_metadata:
                meta["is_cited"] = False

        llm_time = time.time() - llm_start
        print(f"LLM generation time: {llm_time:.2f}s")
        print(f"응답에 포함된 출처 수: {len(cited_sources)}")

        # 최종 응답 생성
        final_result = {
            "answer": cleaned_answer,  # 정제된 응답 사용
            "sources": source_metadata,
            "cited_sources": cited_sources,
            "processing_time": {
                "retrieval": round(retrieval_time, 2),
                "enhancement": round(enhance_time, 2),
                "reranking": round(rerank_time, 2),
                "llm_generation": round(llm_time, 2),
                "total": round(time.time() - start_time, 2),
            },
        }
        
        # 결과 캐싱 (재사용을 위해)
        try:
            RedisCache.set(cache_key, final_result, CACHE_TTL_SEARCH)
            print(f"응답 결과 캐싱 완료: {cache_key}")
        except Exception as cache_error:
            print(f"캐싱 중 오류 발생 (무시됨): {cache_error}")
        
        return final_result

    except Exception as e:
        print(f"검색 및 응답 생성 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "answer": f"요청을 처리하는 중 오류가 발생했습니다: {str(e)}",
            "sources": [],
            "cited_sources": [],  # 빈 cited_sources 추가
            "error": str(e),  # 오류 정보 추가
            "processing_time": {  # 일관된 구조 유지를 위한 처리 시간 정보 추가
                "total": round(time.time() - start_time, 2),
                "retrieval": 0,
                "enhancement": 0,
                "reranking": 0,
                "llm_generation": 0
            }
        }


# SQLCoder 모델 로드 함수 추가
def get_sqlcoder_model():
    """SQLCoder 모델을 로드합니다."""
    print("SQLCoder 모델 로딩 중...")
    try:
        from app.utils.sqlcoder_utils import load_sqlcoder_model
        model, tokenizer = load_sqlcoder_model()
        
        if model is None or tokenizer is None:
            print("SQLCoder 모델 로드 실패")
            return None, None
            
        print("SQLCoder 모델 로드 성공")
        return model, tokenizer
    except Exception as e:
        print(f"SQLCoder 모델 로딩 중 오류 발생: {e}")
        traceback.print_exc()
        return None, None

# 모델 초기화
es_client = get_elasticsearch_client()
embedding_function = get_embedding_function()
llm_model, tokenizer = get_llm_model_and_tokenizer()
reranker_model = get_reranker_model()
sqlcoder_model, sqlcoder_tokenizer = get_sqlcoder_model()


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
    keywords: Optional[List[str]] = None  # 프론트엔드에서 전달하는 하이라이트 키워드 (선택)
    answer_text: Optional[str] = None  # 챗봇 응답 전체 텍스트 (선택)


# 참고 문서 미리보기 엔드포인트
@app.post("/api/source-preview")
async def source_preview_endpoint(request: SourcePreviewRequest = Body(...)):
    try:
        # 요청 데이터 유효성 검사
        if not request.path:
            print(f"소스 프리뷰 오류: 경로 필드가 비어있습니다")
            return {
                "status": "error",
                "message": "요청에 파일 경로가 없습니다",
                "content": None,
            }
            
        print(
            f"Source preview 요청: path={request.path}, page={request.page}, chunk_id={request.chunk_id}"
        )

        # ES 클라이언트 확인
        if not es_client:
            print("소스 프리뷰 오류: Elasticsearch 클라이언트가 초기화되지 않았습니다")
            return {
                "status": "error",
                "message": "검색 서비스를 사용할 수 없습니다. 관리자에게 문의하세요.",
                "content": None,
            }

        # 먼저 해당 파일이 인덱스에 존재하는지 검증
        file_exists_query = {
            "size": 0,
            "query": {
                "term": {"source": request.path}
            },
            "aggs": {
                "path_exists": {
                    "value_count": {
                        "field": "source"
                    }
                }
            }
        }
        
        try:
            # 먼저 파일 존재 여부를 확인
            verify_response = es_client.search(index=ES_INDEX_NAME, body=file_exists_query)
            doc_count = verify_response.get("aggregations", {}).get("path_exists", {}).get("value", 0)
            
            if doc_count == 0:
                print(f"파일이 인덱스에 존재하지 않음: {request.path}")
                
                # 파일 목록 조회 및 유사한 파일 찾기
                indexed_files_resp = es_client.search(
                    index=ES_INDEX_NAME, 
                    body={"size": 0, "aggs": {"unique_files": {"terms": {"field": "source", "size": 30}}}}
                )
                
                files = [bucket["key"] for bucket in indexed_files_resp.get("aggregations", {}).get("unique_files", {}).get("buckets", [])]
                
                # 비슷한 파일명 찾기 (간단한 유사도)
                similar_files = []
                if files:
                    request_filename = request.path.split("_", 1)[1] if "_" in request.path else request.path
                    for file in files:
                        file_name = file.split("_", 1)[1] if "_" in file else file
                        if any(part in file_name for part in request_filename.split("_") if len(part) > 3):
                            similar_files.append(file)
                
                return {
                    "status": "error",
                    "message": f"요청한 문서를 찾을 수 없습니다.",
                    "content": None,
                    "debug_info": {
                        "request_path": request.path,
                        "indexed_files_count": len(files),
                        "similar_files": similar_files[:3] if similar_files else []
                    }
                }
        except Exception as e:
            print(f"파일 존재 확인 중 오류: {e}")
            # 오류가 발생하더라도 계속 진행
            pass

        # chunk_id 전처리 - 명시적 문자열 변환
        chunk_id_query = str(request.chunk_id) if request.chunk_id is not None else ""

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
            print(f"chunk_id({request.chunk_id})를 정수로 변환할 수 없어 두 번째 쿼리 시도 건너뜀")
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
        
        # 시도 목록을 순회하며 검색 실행
        doc = None
        failed_attempts = []
        
        for i, query in enumerate(search_attempts):
            try:
                response = es_client.search(index=ES_INDEX_NAME, body=query)
                hits = response.get("hits", {}).get("hits", [])
                
                if hits:
                    doc = hits[0]
                    print(f"검색 시도 {i+1}번째 성공: {hits[0].get('_id')}")
                    break
                else:
                    print(f"검색 시도 {i+1}번째 실패")
                    failed_attempts.append({"query": query, "error": "결과 없음"})
            except Exception as query_error:
                print(f"검색 시도 {i+1}번째 오류: {str(query_error)}")
                failed_attempts.append({"query": query, "error": str(query_error)})
        
        # 문서를 찾지 못한 경우
        if not doc:
            error_msg = "요청한 문서를 인덱스에서 찾을 수 없습니다."
            
            # 디버깅 정보 로깅
            print(f"소스 프리뷰 실패: {error_msg}")
            print(f"요청 정보: path={request.path}, page={request.page}, chunk_id={request.chunk_id}")
            print(f"모든 검색 시도 실패: {json.dumps(failed_attempts, ensure_ascii=False)}")
            
            # 비슷한 페이지나 청크 찾기 위한 쿼리
            try:
                similar_query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"source": request.path}}
                            ]
                        }
                    },
                    "size": 1
                }
                
                similar_response = es_client.search(index=ES_INDEX_NAME, body=similar_query)
                similar_hits = similar_response.get("hits", {}).get("hits", [])
                
                if similar_hits:
                    similar_doc = similar_hits[0]["_source"]
                    similar_page = similar_doc.get("page")
                    similar_chunk = similar_doc.get("chunk_id")
                    
                    suggestions = []
                    if similar_page and similar_page != request.page:
                        suggestions.append(f"페이지 {similar_page}에서 내용을 찾을 수 있습니다.")
                    
                    if suggestions:
                        error_msg += f" {suggestions[0]}"
            except Exception as e:
                print(f"유사 문서 검색 중 오류: {e}")
            
            return {
                "status": "error",
                "message": error_msg,
                "content": None,
                "debug_info": {
                    "request_params": {
                        "path": request.path,
                        "page": request.page,
                        "chunk_id": request.chunk_id
                    },
                    "failed_attempts": failed_attempts
                } if os.environ.get("DEBUG_MODE") == "true" else None
            }
        
        # 이미지 경로 확인
        image_path = doc["_source"].get("image_path")
        if image_path:
            print(f"이미지 경로 발견: {image_path}")
            # 이미지 URL 구성
            image_url = f"/static/document_images/{os.path.basename(image_path)}"
            return {
                "status": "success",
                "message": "이미지 콘텐츠를 찾았습니다.",
                "content_type": "image/jpeg",  # 기본값, 실제로는 확장자에 따라 달라질 수 있음
                "image_url": image_url,
            }
        
        # 텍스트 콘텐츠 가져오기
        content = doc["_source"].get("text", "")
        if not content:
            return {
                "status": "error",
                "message": "문서 내용을 찾을 수 없습니다.",
                "content": None,
            }
        
        # 원본 콘텐츠 포맷팅
        formatted_content = format_content(content)
        
        # 프론트엔드에서 전달한 키워드 또는 자동 추출
        use_keywords = []
        if request.keywords and isinstance(request.keywords, list):
            use_keywords = request.keywords
        else:
            # 자동으로 키워드 추출
            use_keywords = extract_keywords_from_text(formatted_content)
        
        # 하이라이트 적용 및 관련 문단 추출
        highlighted_content, highlighted_keywords = highlight_keywords(
            formatted_content, 
            use_keywords,
            request.answer_text  # 챗봇 응답 텍스트 전달
        )
        
        # 결과 반환
        return {
            "status": "success",
            "message": "문서 콘텐츠를 찾았습니다.",
            "content": highlighted_content,
            "original_content": formatted_content,
            "keywords": highlighted_keywords,  # 하이라이트된 키워드
            "source_metadata": {
                "filename": os.path.basename(request.path),
                "page": request.page,
                "chunk_id": request.chunk_id,
            },
        }
        
    except Exception as e:
        print(f"문서 미리보기 처리 중 오류 발생: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"문서 미리보기 처리 중 오류가 발생했습니다: {str(e)}",
            "content": None,
            "error_details": str(e) if os.environ.get("DEBUG_MODE") == "true" else None
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
    # 필수 리소스 확인
    if not all([es_client, embedding_function, llm_model, tokenizer, reranker_model]):
        print("필수 리소스 로딩에 실패했습니다. 서버 로그를 확인하세요.")
    
    # SQLCoder 초기화
    print("SQLCoder 모듈 초기화 중...")
    try:
        # sql_sqlcoder_init.py 에서 초기화 함수 임포트
        from app.sql_sqlcoder_init import initialize_sqlcoder
        
        # SQLCoder 초기화
        success, message = initialize_sqlcoder()
        
        if success:
            print(f"SQLCoder 초기화 성공: {message}")
            # 앱 상태에 모델 저장 (API에서 사용)
            app.state.sqlcoder_model = sqlcoder_model
            app.state.sqlcoder_tokenizer = sqlcoder_tokenizer
        else:
            print(f"SQLCoder 초기화 실패: {message}")
    except Exception as e:
        print(f"SQLCoder 초기화 중 예외 발생: {str(e)}")
        traceback.print_exc()
    
    # 모델 상태 확인
    app.state.llm_model = llm_model
    app.state.tokenizer = tokenizer


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


# 파일 삭제 엔드포인트
@app.delete("/api/delete-file")
async def delete_file(filename: str):
    """특정 파일을 Elasticsearch와 디스크에서 삭제합니다."""
    if not es_client:
        raise HTTPException(status_code=503, detail="Elasticsearch is not connected")
    
    if not filename:
        raise HTTPException(status_code=400, detail="Filename parameter is required")

    # 보안: filename에 ../ 등이 포함되어 상위 디렉토리 접근 시도 방지
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    try:
        print(f"파일 삭제 요청: {filename}")
        
        # file_manager.py의 delete_indexed_file 함수 호출
        result = delete_indexed_file(
            es_client=es_client,
            filename=filename,
            index_name=ES_INDEX_NAME,
            uploads_dir=os.path.join(STATIC_DIR, "uploads")
        )
        
        if result["status"] == "success":
            print(f"파일 삭제 성공: {filename}")
            return {"status": "success", "message": result["message"]}
        else:
            print(f"파일 삭제 실패: {filename}, 이유: {result['message']}")
            return {"status": result["status"], "message": result["message"]}
    
    except Exception as e:
        print(f"파일 삭제 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"파일 삭제 중 오류가 발생했습니다: {str(e)}")


# 파일 전체 삭제 엔드포인트 추가
@app.delete("/api/delete-all-files")
async def delete_all_files():
    """
    인덱싱된 모든 파일을 삭제합니다.
    파일 시스템에서 파일을 삭제하고 Elasticsearch에서 관련 문서를 모두 제거합니다.
    """
    try:
        es_client = get_elasticsearch_client()
        if not es_client:
            raise HTTPException(status_code=500, detail="Elasticsearch 연결 실패")
            
        # 업로드 디렉토리 경로
        uploads_dir = "app/static/uploads"
        
        # 1. 모든 파일 목록 가져오기
        try:
            files = os.listdir(uploads_dir)
        except Exception as e:
            logger.error(f"업로드 디렉토리 읽기 실패: {e}")
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": f"업로드 디렉토리 읽기 실패: {str(e)}"}
            )
            
        # 2. 각 파일에 대해 삭제 작업 수행
        deleted_count = 0
        failed_count = 0
        
        for filename in files:
            file_path = os.path.join(uploads_dir, filename)
            if os.path.isfile(file_path):
                try:
                    # Elasticsearch에서 문서 삭제
                    delete_query = {
                        "query": {
                            "term": {
                                "source": filename
                            }
                        }
                    }
                    
                    es_client.delete_by_query(
                        index=ES_INDEX_NAME, 
                        body=delete_query,
                        refresh=True
                    )
                    
                    # 파일 시스템에서 삭제
                    os.remove(file_path)
                    deleted_count += 1
                    logger.info(f"파일 삭제 성공: {filename}")
                except Exception as e:
                    failed_count += 1
                    logger.error(f"파일 삭제 실패 ({filename}): {e}")
        
        # 3. 결과 반환
        if failed_count == 0:
            return JSONResponse(
                content={
                    "status": "success", 
                    "message": f"모든 파일이 성공적으로 삭제되었습니다. (총 {deleted_count}개)"
                }
            )
        else:
            return JSONResponse(
                content={
                    "status": "partial_success",
                    "message": f"{deleted_count}개 파일 삭제 성공, {failed_count}개 파일 삭제 실패"
                }
            )
            
    except Exception as e:
        logger.error(f"전체 파일 삭제 중 오류 발생: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"전체 파일 삭제 중 오류 발생: {str(e)}"}
        )


# 파일 업로드 및 인덱싱 엔드포인트
@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),  # 다중 파일 지원
    category: str = Form("메뉴얼"),  # 기본값을 메뉴얼로 설정
):
    results = []
    start_time = time.time()  # 전체 처리 시작 시간

    logger.info(f"파일 업로드 요청 수신: {len(files)}개 파일, 카테고리: {category}")

    # 지원하는 이미지 확장자 목록 (OCR 처리 가능)
    ocr_supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    # 메모리 정리 - 초기 상태
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.info(f"업로드 처리 시작 - 초기 GPU 메모리 사용량: {initial_memory:.2f} MB")

    for file_index, file in enumerate(files):
        logger.info(f"[{file_index+1}/{len(files)}] 파일 처리 중: {file.filename}, 카테고리: {category}")
        
        # 파일 확장자 확인
        file_extension = Path(file.filename).suffix.lower()
        is_ocr_candidate = file_extension in ocr_supported_extensions or file_extension == '.pdf'
        
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
                    "extension": file_extension,
                    "index": file_index + 1,
                    "total": len(files),
                    "ocr_supported": is_ocr_candidate  # OCR 지원 여부 표시
                }
            }

            # 파일 처리 및 인덱싱
            try:
                logger.info(f"파일 인덱싱 시작: {file.filename}")
                
                if is_ocr_candidate:
                    logger.info(f"OCR 지원 파일 감지: {file.filename} - OCR 처리가 시도될 수 있습니다.")
                
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
                        logger.info(f"파일 처리 전 GPU 메모리: {pre_process_memory:.2f} MB")

                    # 파일 처리 진행률 업데이트 (실제로는 비동기 처리가 필요할 수 있음)
                    file_result["progress"] = 30
                    
                    processing_start = time.time()
                    if is_ocr_candidate:
                        logger.info(f"OCR 처리 시작: {file.filename}")
                    
                    success = await process_and_index_file(
                        es_client, embedding_function, file_path, category
                    )
                    
                    processing_time = time.time() - processing_start
                    logger.info(f"파일 처리 소요 시간: {processing_time:.2f}초")

                    # 메모리 사용량 확인
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        post_process_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                        memory_used = post_process_memory - pre_process_memory
                        logger.info(f"파일 처리 후 GPU 메모리: {post_process_memory:.2f} MB (변화: {memory_used:.2f} MB)")

                    if success:
                        logger.info(f"파일 인덱싱 성공: {file.filename}")
                        # 성공 메시지에 OCR 정보 포함
                        if is_ocr_candidate:
                            success_message = f"파일 '{file.filename}' 인덱싱 완료 (OCR 처리 적용)"
                        else:
                            success_message = f"파일 '{file.filename}' 인덱싱 완료"
                            
                        file_result.update({
                                "status": "success",
                                "message": success_message,
                            "processing_time": round(time.time() - file_result["start_time"], 2),
                            "ocr_processed": is_ocr_candidate,
                            "progress": 100
                        })
                    else:
                        logger.error(f"파일 인덱싱 실패: {file.filename}")
                        file_result.update({
                                "status": "error",
                                "message": "파일 인덱싱 실패",
                            "processing_time": round(time.time() - file_result["start_time"], 2),
                            "progress": 100
                        })
            except Exception as e:
                logger.error(f"파일 처리 중 오류 발생: {file.filename}, 오류: {str(e)}")
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
            logger.error(f"파일 저장 중 오류 발생: {file.filename}, 오류: {str(e)}")
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
        logger.info(f"전체 처리 완료 - 최종 GPU 메모리 사용량: {final_memory:.2f} MB")

    # 전체 요약 통계 추가
    total_time = round(time.time() - start_time, 2)
    ocr_files_count = sum(1 for r in results if r.get("file_info", {}).get("ocr_supported", False))
    ocr_processed_count = sum(1 for r in results if r.get("ocr_processed", False) and r["status"] == "success")
    
    summary = {
        "total_files": len(results),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "error_count": sum(1 for r in results if r["status"] == "error"),
        "skipped_count": sum(1 for r in results if r["status"] == "skipped"),
        "ocr_supported_count": ocr_files_count,
        "ocr_processed_count": ocr_processed_count,
        "total_size": sum(r["size"] for r in results),
        "total_size_formatted": format_file_size(sum(r["size"] for r in results)),
        "total_processing_time": total_time,
        "average_file_time": round(total_time / len(results), 2) if results else 0
    }

    return JSONResponse(
        content={
            "status": "success" if summary["error_count"] == 0 else "partial_success",
            "message": f"{summary['total_files']}개 파일 처리 완료. {summary['success_count']}개 성공, {summary['error_count']}개 실패, {summary['skipped_count']}개 건너뜀 (OCR 처리: {summary['ocr_processed_count']}개)",
            "results": results,
            "summary": summary,
        }
    )

# 질문-응답 엔드포인트
@app.post("/api/chat")
async def chat(request: QuestionRequest = Body(...)):
    logger.info(f"Received chat request: '{request.question}', Category: '{request.category}', History items: {len(request.history) if request.history else 0}")
    request_start_time = time.time()

    # 의존성 주입 또는 전역 변수를 통해 모델/클라이언트 가져오기
    # 이 예제에서는 전역 변수(es_client, embedding_function, reranker_model, llm_model, tokenizer)를 사용한다고 가정합니다.
    # 실제 프로덕션 코드에서는 FastAPI의 Depends 시스템을 사용하는 것이 좋습니다.
    # FastAPI 애플리케이션 시작 시 (예: @app.on_event("startup")) 이 변수들이 초기화되어야 합니다.
    global es_client, embedding_function, reranker_model, llm_model, tokenizer

    if not all([es_client, embedding_function, reranker_model, llm_model, tokenizer]):
        logger.error("Critical components (ES, models, tokenizer) not initialized.")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "챗봇 시스템이 준비되지 않았습니다. 관리자에게 문의하세요."}
        )

    try:
        # 1. Elasticsearch에서 문서 검색, 검색 결과 개선, 리랭킹
        search_pipeline_start_time = time.time()

        # 1a. Elasticsearch Retriever를 사용하여 초기 문서 검색
        retriever = ElasticsearchRetriever(
            es_client=es_client,
            embedding_function=embedding_function,
            category=request.category,
            k=10 # 초기 검색 문서 수 (search_and_combine 함수 참고)
        )
        retrieved_docs_initial = await asyncio.to_thread(retriever.get_relevant_documents, request.question)
        logger.info(f"Initial document retrieval completed in {time.time() - search_pipeline_start_time:.4f} seconds. Found {len(retrieved_docs_initial)} docs.")

        if not retrieved_docs_initial:
            logger.info("No relevant documents found for the query from initial retrieval.")
            async def empty_response_stream():
                yield f"data: {json.dumps({'token': '관련 문서를 찾을 수 없습니다. 다른 질문을 시도해 주세요.'})}\n\n"
                yield f"data: {json.dumps({'event': 'eos'})}\n\n"
            return StreamingResponse(empty_response_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        # 1b. EnhancedSearchPipeline을 사용하여 검색 결과 개선 (TypeError 수정)
        enhance_start_time = time.time()
        enhancer = EnhancedSearchPipeline() # 인자 없이 초기화
        # process 메소드는 (query, docs)를 인자로 받음
        _query_info, enhanced_docs = await asyncio.to_thread(enhancer.process, request.question, retrieved_docs_initial)
        logger.info(f"Search enhancement completed in {time.time() - enhance_start_time:.4f} seconds.")
        
        # 개선된 문서가 있으면 사용, 없으면 초기 검색 결과 사용
        docs_for_reranking = enhanced_docs if enhanced_docs else retrieved_docs_initial

        # 1c. Reranking
        rerank_start_time = time.time()
        local_reranker = EnhancedLocalReranker(reranker_model=reranker_model)
        reranked_docs = await asyncio.to_thread(local_reranker.rerank, request.question, docs_for_reranking)
        logger.info(f"Document reranking completed in {time.time() - rerank_start_time:.4f} seconds. Reranked to {len(reranked_docs)} docs.")
        
        # 실제 LLM에 전달할 문서 수 제한 (예: 상위 5-10개)
        # 너무 많은 문서는 컨텍스트 길이 초과 또는 노이즈 증가 유발 가능
        # 이 값은 실험을 통해 최적화 필요
        NUM_DOCS_FOR_LLM = 7 # 예시 값
        top_docs = reranked_docs[:NUM_DOCS_FOR_LLM]
        logger.info(f"Using top {len(top_docs)} docs for LLM context.")

        # 2. LLM에 전달할 최종 프롬프트 생성
        prompt_generation_start_time = time.time()
        prompt_data = await generate_llm_response(
            tokenizer,
            request.question,
            top_docs,
            0.1,
            request.history
        )
        final_prompt_text = prompt_data["prompt_text"]
        source_metadata = prompt_data["source_metadata"]
        top_docs_content = prompt_data["top_docs"]
        logger.info(f"Prompt generation completed in {time.time() - prompt_generation_start_time:.4f} seconds.")
        # logger.debug(f"Final prompt for LLM (first 200 chars): {final_prompt_text[:200]}")

        # 3. TextIteratorStreamer 및 StreamingResponse 설정
        # streamer는 현재 토큰나이저를 사용
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        # 모델 추론을 위한 입력 준비 (토큰화)
        # 주의: final_prompt_text가 이미 ChatML 템플릿이 적용된 완전한 문자열이어야 함
        # (즉, tokenizer.apply_chat_template의 결과물)
        try:
            inputs = tokenizer(final_prompt_text, return_tensors="pt", padding=False, truncation=False).to(llm_model.device)
        except Exception as e:
            logger.error(f"Error tokenizing final prompt: {e}. Prompt (first 100 chars): {final_prompt_text[:100]}")
            raise # 토큰화 실패는 심각한 문제이므로 예외를 다시 발생시켜 처리 중단

        # LLM 생성 파라미터 설정
        # 실제 모델 및 사용 사례에 맞게 이 값들을 조정해야 합니다.
        generation_temperature = 0.1 # 예시: 약간의 창의성 허용, 너무 높으면 일관성 저하
        generation_max_new_tokens = 2048 # 답변 최대 길이

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=generation_max_new_tokens,
            temperature=generation_temperature,
            pad_token_id=tokenizer.eos_token_id, # 매우 중요
            #eos_token_id=tokenizer.eos_token_id, # 필요시 명시 (Qwen은 여러 eos_token_id를 가질 수 있음)
            streamer=streamer,
        )
        # temperature > 0 이면 do_sample=True가 기본이나, 명시적으로 설정 가능
        if generation_temperature > 0.0:
             generation_kwargs["do_sample"] = True
        else: # temperature가 0이면 greedy decoding
             generation_kwargs["do_sample"] = False
        
        # Qwen 모델은 <|im_end|>를 eos_token으로 사용할 수 있음.
        # 또는 tokenizer.eos_token_id가 이를 가리키도록 설정되어 있어야 함.
        # eos_token_ids = [tokenizer.eos_token_id]
        # if tokenizer.im_end_id: # Qwen의 특수 토큰 ID 확인
        #    eos_token_ids.append(tokenizer.im_end_id)
        # generation_kwargs["eos_token_id"] = eos_token_ids # 리스트로 전달 가능

        logger.info(f"Starting LLM generation with params: temp={generation_temperature}, max_tokens={generation_max_new_tokens}")
        llm_generation_start_time = time.time()

        # 별도 스레드에서 모델 생성 실행 (GPU 작업은 GIL의 영향을 덜 받지만, I/O 바운드 작업처럼 처리)
        # autocast 컨텍스트는 generate 함수 내부에서 처리되거나, 스레드 타겟 함수를 래핑하여 적용 가능
        # 현재 llm_model.generate가 autocast를 내부적으로 처리한다고 가정
        thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
        thread.start()

        # 인용 감지 및 소스 처리를 위한 변수
        accumulated_text = ""
        cited_sources = []

        # 비동기 제너레이터 정의
        async def stream_generator():
            nonlocal accumulated_text, cited_sources
            # logger.debug("Stream generator started.")
            generated_text_count = 0
            try:
                for new_text in streamer:
                    if new_text:
                        generated_text_count += len(new_text)
                        accumulated_text += new_text
                        # logger.debug(f"Streaming token: {new_text}")
                        yield f"data: {json.dumps({'token': new_text})}\n\n" # SSE 형식
                    await asyncio.sleep(0.001) # 다른 비동기 작업 실행 기회 부여
                
                # 스트림 종료 시 인용 소스 처리
                if accumulated_text:
                    # 응답 정제 (불필요한 시스템 메시지 등 제거)
                    def clean_response(resp):
                        if not resp:
                            return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."
                            
                        # 1. 시스템 메시지 제거
                        if resp.startswith("system\n"):
                            parts = resp.split("user\n")
                            if len(parts) > 1:
                                resp = parts[1]
                                parts = resp.split("assistant\n")
                                if len(parts) > 1:
                                    resp = parts[1].strip()
                                    return resp
                        
                        # 2. assistant 접두사 제거
                        if "assistant\n" in resp:
                            parts = resp.split("assistant\n")
                            if len(parts) > 1:
                                resp = parts[-1].strip()
                                return resp
                        
                        # 3. 그 외의 경우
                        # 불필요한 태그 제거
                        for tag in ["system\n", "user\n", "assistant\n", "system:", "user:", "assistant:", "system", "user", "assistant"]:
                            if resp.startswith(tag):
                                resp = resp[len(tag):].strip()
                                
                        return resp.strip()
                    
                    # 응답 정제 적용
                    cleaned_text = clean_response(accumulated_text)
                    
                    # 인용 소스 감지
                    def extract_keywords(text, min_length=3, max_keywords=20):
                        if text is None or not isinstance(text, str):
                            return []
                        words = re.findall(r'\b[가-힣a-zA-Z0-9]+\b', text)
                        filtered_words = [w for w in words if len(w) >= min_length]
                        unique_words = list(set(filtered_words))[:max_keywords]
                        return unique_words
                    
                    # 응답에서 키워드 추출
                    answer_keywords = extract_keywords(cleaned_text)
                    
                    # 인용 소스 감지
                    for i, meta in enumerate(source_metadata):
                        cited = False
                        source_text = top_docs_content[i].page_content if i < len(top_docs_content) else ""
                        
                        # 1. 연속된 텍스트 일치 여부 확인 (최소 30자)
                        if len(source_text) > 50:
                            for j in range(0, len(source_text) - 30, 10):
                                snippet = source_text[j:j+30]
                                if snippet in cleaned_text:
                                    cited = True
                                    break
                        
                        # 2. 키워드 기반 매칭
                        if not cited and source_text:
                            source_keywords = extract_keywords(source_text)
                            if source_keywords:
                                matches = [k for k in source_keywords if k in cleaned_text]
                                match_ratio = len(matches) / len(source_keywords)
                                if match_ratio > 0.3:
                                    cited = True
                        
                        # 인용된 소스만 추가
                        if cited:
                            meta["is_cited"] = True
                            cited_sources.append(meta)
                        else:
                            meta["is_cited"] = False
                
                # 스트림 종료 알림 (모든 토큰 생성 완료) - 출처 정보 포함
                logger.info(f"LLM generation stream finished. Total chars: {generated_text_count}. Time: {time.time() - llm_generation_start_time:.4f}s")
                
                # 최종 메시지에 출처 정보 포함
                yield f"data: {json.dumps({'event': 'sources', 'sources': source_metadata, 'cited_sources': cited_sources})}\n\n"
                
                # 스트림 종료 이벤트
                yield f"data: {json.dumps({'event': 'eos', 'message': 'Stream ended successfully.'})}\n\n"
                
                # 스트리밍 응답 완료 후 캐싱 처리
                try:
                    # Redis 캐싱을 위한 데이터 준비
                    from app.utils.cache_utils import RedisCache, CacheKeys, CACHE_TTL_CHAT
                    
                    # 캐시 키 생성 (질문 + 카테고리 기반)
                    cache_key = RedisCache.generate_key(
                        CacheKeys.CHAT, 
                        {"query": request.question, "category": request.category}
                    )
                    
                    # 캐싱할 최종 결과 데이터 구성
                    final_result = {
                        "answer": cleaned_text,
                        "sources": source_metadata,
                        "cited_sources": cited_sources,
                        "processing_time": {
                            "total": round(time.time() - request_start_time, 2),
                            "llm_generation": round(time.time() - llm_generation_start_time, 2)
                        }
                    }
                    
                    # 결과 캐싱
                    cache_success = RedisCache.set(cache_key, final_result, CACHE_TTL_CHAT)
                    if cache_success:
                        logger.info(f"스트리밍 응답 캐싱 완료: {cache_key}")
                    else:
                        logger.warning(f"스트리밍 응답 캐싱 실패: {cache_key}")
                except Exception as cache_error:
                    logger.error(f"스트리밍 응답 캐싱 중 오류 발생: {cache_error}")

            except Exception as e:
                logger.error(f"Error during LLM streaming: {e}", exc_info=True)
                yield f"data: {json.dumps({'error': '스트리밍 중 오류가 발생했습니다.', 'details': str(e)})}\n\n"
            finally:
                if thread.is_alive():
                    thread.join(timeout=5.0) # 스레드 종료 대기 (타임아웃 설정)
                    if thread.is_alive():
                        logger.warning("LLM generation thread did not terminate gracefully.")
                # logger.debug("Stream generator finished.")
                total_request_time = time.time() - request_start_time
                logger.info(f"Total chat request processing time: {total_request_time:.4f} seconds.")
        
        # StreamingResponse 반환
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache", # 클라이언트 및 프록시 캐싱 방지
            "Connection": "keep-alive",  # 연결 유지
            "X-Accel-Buffering": "no",   # Nginx 등 리버스 프록시 버퍼링 비활성화
        }
        return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=headers)

    except HTTPException: # FastAPI의 HTTPException은 그대로 전달
        raise
    except Exception as e:
        logger.error(f"Unhandled error in chat endpoint: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "챗봇 응답 처리 중 심각한 오류가 발생했습니다.", "error_details": str(e)}
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


# 대화 제목 자동 생성을 위한 모델 추가
class GenerateTitleRequest(BaseModel):
    messages: List[Dict[str, Any]]  # 대화 메시지 목록

# 대화 제목 생성 API 엔드포인트
@app.post("/api/generate-title")
async def generate_title_endpoint(request: GenerateTitleRequest = Body(...)):
    try:
        if not request.messages or len(request.messages) == 0:
            return {"title": "새 대화"}

        # 첫 번째 사용자 메시지 추출
        user_messages = [msg for msg in request.messages if msg.get("role") == "user"]
        if not user_messages:
            return {"title": "새 대화"}

        first_user_message = user_messages[0].get("content", "")
        if not first_user_message:
            return {"title": "새 대화"}

        # 간단한 제목 생성 로직 (LLM 사용 없이)
        title = first_user_message[:20]  # 첫 20자 추출

        # 마침표, 물음표, 느낌표로 끝나는 경우 처리
        punctuation_marks = [".", "?", "!", ",", ";", ":", "...", "…"]
        for mark in punctuation_marks:
            if title.endswith(mark):
                title = title[:-len(mark)]
                break

        # 조사로 끝나는 경우 처리
        korean_particles = ["이", "가", "을", "를", "은", "는", "에", "의", "로", "와", "과"]
        for particle in korean_particles:
            if title.endswith(particle):
                title = title[:-len(particle)]
                break

        # 너무 짧은 경우 적당한 접미사 추가
        if len(title) < 5:
            title += "에 대한 대화"
        elif "?" in first_user_message or "질문" in first_user_message:
            title += "에 대한 질문"

        # 50자 이상이면 줄임
        if len(title) > 50:
            title = title[:47] + "..."

        return {"title": title.strip()}
    except Exception as e:
        print(f"대화 제목 생성 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"대화 제목 생성 중 오류 발생: {str(e)}")


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


# 피드백 분석 및 검색 품질 개선 API 추가
@app.get("/api/feedback/stats")
async def get_feedback_statistics():
    """
    피드백 데이터 통계를 반환하는 API 엔드포인트
    """
    try:
        analyzer = FeedbackAnalyzer()
        stats = analyzer.get_feedback_stats()

        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        print(f"피드백 통계 조회 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"피드백 통계 조회 중 오류 발생: {str(e)}"
        )

@app.get("/api/feedback/trends")
async def get_feedback_trends(days: int = 30):
    """
    최근 N일간의 피드백 추세를 분석하는 API 엔드포인트
    """
    try:
        analyzer = FeedbackAnalyzer()
        trends = analyzer.analyze_feedback_trends(days=days)

        return {
            "status": "success",
            "trends": trends
        }
    except Exception as e:
        print(f"피드백 추세 분석 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"피드백 추세 분석 중 오류 발생: {str(e)}"
        )

@app.get("/api/feedback/documents")
async def get_document_quality_scores():
    """
    피드백을 기반으로 한 문서 품질 점수를 반환하는 API 엔드포인트
    """
    try:
        analyzer = FeedbackAnalyzer()
        scores = analyzer.calculate_document_quality_scores()

        # 단순 출력용으로 점수 정렬
        sorted_scores = sorted(
            [{"document": doc, "score": score} for doc, score in scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )

        return {
            "status": "success",
            "document_scores": sorted_scores,
            "count": len(sorted_scores)
        }
    except Exception as e:
        print(f"문서 품질 점수 조회 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"문서 품질 점수 조회 중 오류 발생: {str(e)}"
        )

@app.get("/api/feedback/frequently-asked")
async def get_frequently_asked_questions(min_count: int = 3):
    """
    자주 묻는 질문 패턴을 반환하는 API 엔드포인트
    """
    try:
        analyzer = FeedbackAnalyzer()
        questions = analyzer.extract_frequent_questions(min_count=min_count)

        return {
            "status": "success",
            "questions": questions,
            "count": len(questions)
        }
    except Exception as e:
        print(f"자주 묻는 질문 조회 중 오류 발생: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"자주 묻는 질문 조회 중 오류 발생: {str(e)}"
        )

# SQL 관련 모델 정의
class SQLQueryRequest(BaseModel):
    question: str
    
class SQLAndLLMRequest(BaseModel):
    question: str

# SQL 관련 엔드포인트 추가
@app.get("/api/db-schema")
async def get_db_schema():
    """데이터베이스 스키마 정보를 반환합니다."""
    try:
        from app.utils.get_mariadb_schema import get_schema_for_sqlcoder, test_db_connection
        
        # 먼저 DB 연결 테스트
        if not test_db_connection():
            print("DB 연결 테스트 실패")
            return {
                "status": "error", 
                "schema": "# 데이터베이스 연결 오류\n\n데이터베이스에 연결할 수 없습니다. 관리자에게 문의하세요."
            }
        
        # SQLCoder 형식으로 스키마 가져오기
        schema = get_schema_for_sqlcoder()
        
        # 오류 메시지가 반환된 경우 (ERROR로 시작하는 문자열)
        if isinstance(schema, str) and schema.startswith("ERROR:"):
            print(f"DB 스키마 조회 오류: {schema}")
            # 오류가 발생했지만 200 OK와 함께 오류 메시지 전달
            return {
                "status": "error", 
                "schema": "# 데이터베이스 연결 오류\n\n데이터베이스에 연결할 수 없습니다. 관리자에게 문의하세요.",
                "error": schema
            }
            
        return {"status": "success", "schema": schema}
    except Exception as e:
        print(f"DB 스키마 조회 중 오류 발생: {str(e)}")
        traceback.print_exc()
        # 500 에러가 아닌 200 OK 응답으로 변경하여 클라이언트 오류 처리 개선
        return {
            "status": "error",
            "schema": "# 데이터베이스 스키마 로딩 오류\n\n시스템 오류로 스키마를 불러올 수 없습니다. 관리자에게 문의하세요.",
            "error": str(e)
        }

@app.post("/api/sql-query")
async def process_sql_query(request: SQLQueryRequest = Body(...)):
    """자연어 질문을 SQL로 변환하고 실행 결과를 반환합니다."""
    try:
        # SQLCoder 유틸 사용
        from app.utils.sqlcoder_utils import generate_sql_from_question, run_sql_query
        
        # SQL 생성
        print(f"자연어 질문: {request.question}")
        sql = generate_sql_from_question(request.question)
        
        if not sql:
            return {
                "sql": "SELECT 1;",
                "results": "⚠️ SQL을 생성할 수 없습니다. 질문을 더 구체적으로 작성해주세요."
            }
        
        # SQL 실행
        results = run_sql_query(sql)
        
        # 결과가 에러인 경우
        if isinstance(results, dict) and "error" in results:
            error_msg = results["error"]
            return {
                "sql": sql,
                "results": f"❌ SQL 실행 오류: {error_msg}"
            }
        
        # 결과가 문자열인 경우 (이미 포맷팅된 메시지일 수 있음)
        if isinstance(results, str):
            return {
                "sql": sql,
                "results": results
            }
        
        # 결과가 비어있는 경우
        if not results or len(results) == 0:
            return {
                "sql": sql,
                "results": "⚠️ 쿼리 결과가 없습니다."
            }
        
        # 테이블 형태로 결과 변환
        try:
            # 결과를 마크다운 테이블로 변환
            headers = list(results[0].keys())
            markdown_table = "| " + " | ".join(headers) + " |\n"
            markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            
            for row in results:
                row_values = []
                for header in headers:
                    value = row[header]
                    # None 값 처리
                    if value is None:
                        value = "NULL"
                    # 긴 문자열 처리
                    elif isinstance(value, str) and len(value) > 50:
                        value = value[:47] + "..."
                    row_values.append(str(value))
                markdown_table += "| " + " | ".join(row_values) + " |\n"
                
            return {
                "sql": sql,
                "results": markdown_table
            }
        except Exception as e:
            print(f"결과 포맷팅 오류: {str(e)}")
            # 결과를 문자열로 변환하여 안전하게 반환
            return {
                "sql": sql,
                "results": f"결과 처리 중 오류가 발생했습니다: {str(e)}\n\n원본 결과: {str(results)}"
            }
            
    except Exception as e:
        print(f"SQL 쿼리 처리 중 오류 발생: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SQL 쿼리 처리 중 오류 발생: {str(e)}")

@app.post("/api/sql-and-llm")
async def process_sql_and_llm(request: SQLAndLLMRequest = Body(...)):
    """자연어 질문을 SQL로 변환 실행하고, LLM으로 설명을 추가합니다. 스트리밍 방식으로 응답합니다."""
    try:
        # SQLCoder 유틸 사용
        from app.utils.sqlcoder_utils import generate_sql_from_question, run_sql_query
        
        # SQL 생성 및 실행
        sql_query = generate_sql_from_question(request.question)
        
        # 쿼리가 오류를 포함하고 있는 경우
        if sql_query.startswith("-- SQL 생성 중 오류 발생"):
            return JSONResponse(content={
                "status": "error",
                "sql": sql_query,
                "results": "❌ SQL 생성에 실패했습니다",
                "explanation": f"SQL 생성 중 오류가 발생했습니다: {sql_query}"
            })
        
        # SQL 쿼리 실행 (run_sql_query가 SQL 및 마크다운 형식 결과 반환)
        sql_result = run_sql_query(sql_query)
        
        # SQL 실행 중 오류가 발생했는지 확인
        has_error = isinstance(sql_result, dict) and sql_result.get('error') is not None
        is_empty = False  # 결과가 비어있는지 여부
        
        if has_error:
            # 오류 메시지 가져오기
            error_message = sql_result.get('error', "알 수 없는 오류")
            results_markdown = f"❌ SQL 실행 오류: {error_message}"
            
            # SQL 오류가 복잡한 경우 단순화된 메시지 생성
            explanation = f"SQL 쿼리 실행 중 오류가 발생했습니다. 쿼리 문법이나 존재하지 않는 테이블/컬럼을 참조했을 수 있습니다."
            
            return JSONResponse(content={
                "status": "error",
                "sql": sql_query,
                "results": results_markdown,
                "explanation": explanation
            })
        elif isinstance(sql_result, str) and "레코드를 찾을 수 없습니다" in sql_result:
            # 결과가 없는 경우
            results_markdown = "⚠️ 조건에 맞는 데이터가 없습니다."
            is_empty = True
        else:
            # 정상 실행 결과인 경우 마크다운 테이블 생성
            results_markdown = sql_result
        
        print(f"[SQL+LLM] LLM 모델로 향상된 응답 생성 시작 - 질문: '{request.question}'")
        
        # LLM을 통한 설명 생성을 스트리밍 방식으로 변경
        # 프롬프트 구성
        explanation_prompt = f"""# SQL 쿼리 결과 분석 및 응답 생성
## 질문
{request.question}

## SQL 쿼리
```sql
{sql_query}
```

## 쿼리 결과
```
{results_markdown}
```

## 지시사항
1. 위 SQL 쿼리와 결과를 바탕으로 질문에 대한 답변을 생성해주세요.
2. 간결하게 요점만 설명하되, 핵심 내용을 누락하지 마세요.
3. 가능하다면 데이터의 특징이나 패턴을 언급하세요.
4. 수치가 있다면 중요한 수치를 언급하고 그 의미를 설명하세요.
5. 질문에 직접적인 답변을 하는 형식으로 작성하세요.
6. SQL 코드나 기술적 설명은 포함하지 마세요.
7. 결과가 비어있다면 그 이유와 가능한 원인을 설명하세요.
"""

        print(f"[SQL+LLM] 향상된 프롬프트 길이: {len(explanation_prompt)} 문자")
        
        # TextIteratorStreamer 설정
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 모델 입력 준비
        try:
            inputs = tokenizer(explanation_prompt, return_tensors="pt", padding=False, truncation=False).to(llm_model.device)
        except Exception as e:
            print(f"[SQL+LLM] 프롬프트 토큰화 오류: {e}")
            return JSONResponse(content={
                "status": "error",
                "sql": sql_query,
                "results": results_markdown,
                "explanation": f"설명 생성을 위한 프롬프트 처리 중 오류 발생: {str(e)}"
            })
        
        # 생성 파라미터 설정
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            do_sample=False  # 결정적 생성
        )
        
        # 별도 스레드에서 모델 생성 실행
        thread = Thread(target=llm_model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # 스트리밍 응답을 위한 변수
        accumulated_text = ""
        
        # 비동기 제너레이터 정의
        async def stream_generator():
            nonlocal accumulated_text
            
            # 먼저 SQL 및 결과 정보 전송
            yield f"data: {json.dumps({'event': 'sql', 'sql': sql_query, 'results': results_markdown})}\n\n"
            
            # LLM 응답 스트리밍
            for new_text in streamer:
                if new_text:
                    accumulated_text += new_text
                    yield f"data: {json.dumps({'token': new_text})}\n\n"
                await asyncio.sleep(0.001)  # 다른 비동기 작업 실행 기회 부여
            
            # 중복 제거 로직 적용
            try:
                cleaned_text = deduplicate_markdown_sections_py(accumulated_text)
                
                # 응답이 비었거나 너무 짧은 경우 등의 후처리
                if not cleaned_text.strip() or len(cleaned_text.strip()) < 10 or "오류" in cleaned_text:
                    if is_empty:
                        cleaned_text = "조건에 맞는 데이터가 없습니다. 검색 조건을 변경해 보시거나, 다른 질문을 시도해보세요."
                    elif "오류" in accumulated_text and not cleaned_text.strip():
                        cleaned_text = accumulated_text  # 원본 오류 메시지 사용
                    else:
                        cleaned_text = "SQL 쿼리 결과를 기반으로 한 설명입니다. 위 테이블에서 자세한 정보를 확인하세요."
                
                # 최종 정리된 응답 전송 (필요시)
                if cleaned_text != accumulated_text:
                    yield f"data: {json.dumps({'event': 'cleaned_explanation', 'explanation': cleaned_text})}\n\n"
            except Exception as clean_error:
                print(f"[SQL+LLM] 응답 정리 중 오류: {clean_error}")
                # 오류 시 원본 텍스트 사용
            
            # 스트림 종료 알림
            yield f"data: {json.dumps({'event': 'eos'})}\n\n"
            
            # 스레드 종료 대기
            if thread.is_alive():
                thread.join(timeout=5.0)
                if thread.is_alive():
                    print("[SQL+LLM] 생성 스레드가 정상적으로 종료되지 않았습니다.")
        
        # StreamingResponse 반환
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(stream_generator(), media_type="text/event-stream", headers=headers)
        
    except Exception as e:
        print(f"SQL+LLM 처리 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return JSONResponse(content={
            "status": "error",
            "sql": "-- SQL 생성 실패",
            "results": "❌ 오류 발생",
            "explanation": f"처리 중 오류가 발생했습니다: {str(e)}"
        })

# 소스 미리보기에 필요한 유틸리티 함수
def extract_keywords_from_text(text, max_keywords=12):
    """텍스트에서 주요 키워드 추출"""
    try:
        # 한글, 영문, 숫자 단어 추출 (특수문자 및 공백 기준 분리)
        words = re.findall(r'[가-힣a-zA-Z0-9]{2,}', text)
        
        # 불용어 제거 (한국어 기준)
        stopwords = {
            '이', '그', '저', '것', '수', '등', '및', '에서', '에게', '으로', '로', '을', '를',
            '이다', '있다', '하다', '이런', '저런', '그런', '어떤', '무슨', '어떻게', '왜',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of',
            '있는', '없는', '경우', '때문', '위해', '통해', '따라', '의해', '의한', '때는',
            '있습니다', '없습니다', '합니다', '입니다', '됩니다', '관련', '때문에', '위하여',
            '만약', '그러나', '하지만', '또한', '그리고', '따라서', '이러한', '그러한', 
            '이것', '그것', '저것', '무엇', '어디', '언제', '누구'
        }
        
        # 중복 제거 및 단어 개수 세기
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            if len(word) >= 2 and word_lower not in stopwords:
                # 특수 가중치 적용 - 특정 길이 범위의 단어에 가중치 부여
                weight = 1.0
                if 3 <= len(word) <= 8:  # 보통 의미있는 단어 길이 범위
                    weight = 1.5
                if word[0].isupper() and len(word) > 1 and not word.isupper():  # 대문자로 시작하는 단어 (고유명사 가능성)
                    weight = 2.0
                
                word_counts[word] = word_counts.get(word, 0) + weight
        
        # 빈도수 기준 상위 키워드 추출 (가중치 적용)
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 중복 단어 제거 (소문자화하여 비교)
        unique_keywords = []
        added_lower = set()
        
        for kw, _ in keywords:
            kw_lower = kw.lower()
            if kw_lower not in added_lower and len(kw) >= 2:
                unique_keywords.append(kw)
                added_lower.add(kw_lower)
            
            if len(unique_keywords) >= max_keywords:
                break
        
        return unique_keywords
    except Exception as e:
        print(f"키워드 추출 중 오류: {e}")
        return []

def format_content(content):
    """내용 서식 개선"""
    if not content:
        return ""
    
    # 여러 개의 연속 공백을 하나로 압축
    formatted = re.sub(r'\s+', ' ', content).strip()
    
    # 문장 구분자 후 개행 추가 (문단 구분)
    formatted = re.sub(r'([.!?])\s+', r'\1\n\n', formatted)
    
    # 중복 개행 제거 (3개 이상 → 2개)
    formatted = re.sub(r'\n{3,}', '\n\n', formatted)
    
    return formatted

def highlight_keywords(content, keywords, answer_text=None):
    """내용에서 문장 단위 하이라이트 - 답변과 일치하는 문장 강조"""
    if not content:
        return content, []
        
    # 최종 결과와 관련 문단 저장 변수
    highlighted_paragraphs = []
    relevant_paragraphs = []
    
    # 답변 텍스트가 있으면 문장 단위 매칭 적용
    if answer_text and isinstance(answer_text, str) and len(answer_text.strip()) > 10:
        print(f"응답 텍스트 기반 하이라이트 시작 - 응답 길이: {len(answer_text)}")
        
        # 1. 콘텐츠를 문단으로 분리
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', content)
        
        # 2. 응답 텍스트에서 핵심 키워드 추출
        answer_keywords = extract_keywords_from_text(answer_text, max_keywords=15)
        
        # 3. 각 문단의 관련성 점수 계산 및 하이라이트 적용
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph or len(paragraph) < 15:
                continue
            
            # 문단에서 키워드 추출
            paragraph_keywords = extract_keywords_from_text(paragraph, max_keywords=10)
            
            # 키워드 일치 점수 계산
            matching_keywords = [k for k in paragraph_keywords if k.lower() in [ak.lower() for ak in answer_keywords]]
            keyword_score = len(matching_keywords) / max(1, len(paragraph_keywords))
            
            # 직접 텍스트 매칭 검사 (정확한 인용구 확인)
            direct_match = False
            if len(paragraph) > 30:
                for i in range(len(paragraph) - 30):
                    snippet = paragraph[i:i+30]
                    if snippet in answer_text:
                        direct_match = True
                        break
            
            # 관련성 높은 문단 선택 (키워드 일치 또는 직접 매칭)
            is_relevant = keyword_score > 0.3 or direct_match
            
            # 하이라이트 적용 및 관련 문단 표시
            if is_relevant:
                # 키워드 볼드 처리와 함께 문단 전체에 배경색 적용
                highlighted_paragraph = paragraph
                for keyword in matching_keywords:
                    try:
                        # 정규식 특수문자 이스케이프
                        escaped_keyword = re.escape(keyword)
                        # 키워드에 볼드체 적용
                        highlighted_paragraph = re.sub(
                            f'\\b{escaped_keyword}\\b', 
                            f'**{keyword}**', 
                            highlighted_paragraph, 
                            flags=re.IGNORECASE
                        )
                    except Exception as e:
                        print(f"하이라이트 오류: {e}")
                
                # 관련성에 따라 다른 클래스 적용 (강한 관련성은 진한 노란색, 중간 관련성은 연한 노란색)
                highlight_class = "highlight-strong" if direct_match else "highlight-medium"
                
                # HTML 태그로 감싸서 노란색 배경 적용
                highlighted_paragraph = f'<span class="{highlight_class}">{highlighted_paragraph}</span>'
                
                highlighted_paragraphs.append(highlighted_paragraph)
                relevant_paragraphs.append({
                    'text': paragraph,
                    'relevance': 'high' if direct_match else 'medium',
                    'matching_keywords': matching_keywords
                })
            else:
                # 관련성 낮은 문단은 그대로 추가
                highlighted_paragraphs.append(paragraph)
        
        # 하이라이트된 내용을 다시 결합
        highlighted_content = "\n\n".join(highlighted_paragraphs)
        
        # 하이라이트에 사용된 키워드 목록 (중복 제거)
        all_matching_keywords = []
        for para in relevant_paragraphs:
            all_matching_keywords.extend(para.get('matching_keywords', []))
        unique_keywords = list(set(all_matching_keywords))
        
        return highlighted_content, unique_keywords
    
    # 답변 텍스트가 없는 경우, 기본 키워드 하이라이트만 적용
    elif keywords and isinstance(keywords, list):
        for keyword in keywords:
            try:
                # 정규식 특수문자 이스케이프
                escaped_keyword = re.escape(keyword)
                # 키워드에 볼드체 적용
                content = re.sub(
                    f'\\b{escaped_keyword}\\b', 
                    f'**{keyword}**', 
                    content, 
                    flags=re.IGNORECASE
                )
            except Exception as e:
                print(f"하이라이트 오류: {e}")
        return content, keywords
    
    # 키워드나 답변 텍스트가 없을 경우 원본 콘텐츠 반환
    return content, []

# 상수 선언
ES_INDEX_NAME = "rag_documents_kure_v1"

# 직접 실행을 위한 코드 추가
if __name__ == "__main__":
    import uvicorn
    print("Qwen2.5-7B 모델로 서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 기타 라우터 등록
from app.stats.dashboard_api import router as dashboard_router
app.include_router(dashboard_router, prefix="", tags=["dashboard"])

# 개선된 중복 제거 함수 (Python 버전) - 마크다운 섹션 기반 구조적 중복 제거
def deduplicate_markdown_sections_py(markdown_text: str) -> str:
    if not markdown_text or not isinstance(markdown_text, str):
        return ""
    
    # 1. 마크다운 헤더(##, ### 등)로 섹션 분리
    section_pattern = re.compile(r'(^|\n)(#+ [^\n]+)(?=\n)', re.MULTILINE)
    sections = []
    last_idx = 0
    for match in section_pattern.finditer(markdown_text):
        start = match.start(2)
        if last_idx < start:
            # 헤더 없는 앞부분 블록
            block = markdown_text[last_idx:match.start(1)].strip()
            if block:
                sections.append((None, block))
        header = match.group(2).strip()
        last_idx = start
        # 다음 헤더 전까지가 본문
    # 마지막 헤더 이후 블록
    if last_idx < len(markdown_text):
        block = markdown_text[last_idx:].strip()
        if block:
            # 헤더 추출
            header_match = re.match(r'^(#+ [^\n]+)\n', block)
            if header_match:
                header = header_match.group(1).strip()
                body = block[len(header):].strip()
                sections.append((header, body))
            else:
                sections.append((None, block))

    # 2. 블록별로 중복(유사) 여부 판단
    def is_similar(a, b, threshold=0.8):
        # difflib의 SequenceMatcher로 유사도 측정
        return difflib.SequenceMatcher(None, a, b).ratio() >= threshold

    unique_blocks = []
    seen_bodies = []
    for header, body in sections:
        # 질문 섹션은 무조건 스킵
        if header and ("질문" in header or "question" in header.lower()):
            continue
        # 본문 정규화
        norm_body = re.sub(r'\s+', ' ', body.strip().lower())
        # 이미 유사한 본문이 있는지 확인
        is_dup = False
        for prev_body in seen_bodies:
            if is_similar(norm_body, prev_body):
                is_dup = True
                break
        if not is_dup:
            seen_bodies.append(norm_body)
            unique_blocks.append((header, body))

    # 3. 최종 조합 (헤더 없는 블록은 그대로, 헤더 있는 블록은 헤더+본문)
    result_lines = []
    for header, body in unique_blocks:
        if header:
            result_lines.append(header)
        result_lines.append(body)
    result = '\n\n'.join(result_lines)
    # 연속 빈 줄 정리
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()
