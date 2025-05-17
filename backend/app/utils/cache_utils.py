import json
import time
import hashlib
from redis import Redis
from typing import Any, Dict, List, Optional, Union
import aioredis

# Redis 연결 설정
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_PASSWORD = None  # 필요 시 비밀번호 설정
REDIS_TIMEOUT = 3  # 연결 타임아웃 (초)

# TTL 설정 (초 단위)
CACHE_TTL_DEFAULT = 3600  # 1시간
CACHE_TTL_SEARCH = 1800   # 30분
CACHE_TTL_FREQUENT = 7200  # 2시간
CACHE_TTL_CONVERSATIONS = 86400  # 24시간

# main.py에서 사용하는 이름으로 추가 상수 정의
REDIS_CACHE_EXPIRY = CACHE_TTL_DEFAULT
REDIS_VECTOR_CACHE_EXPIRY = CACHE_TTL_FREQUENT  # 벡터 검색 캐시는 자주 사용하는 것으로 간주
REDIS_CHAT_HISTORY_EXPIRY = CACHE_TTL_CONVERSATIONS

class RedisCache:
    """Redis 캐싱 유틸리티 클래스"""
    
    _instance = None
    _client = None
    
    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴 구현"""
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_client(cls) -> Redis:
        """Redis 클라이언트 인스턴스 반환"""
        if cls._client is None:
            try:
                cls._client = Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD,
                    socket_timeout=REDIS_TIMEOUT,
                    decode_responses=True  # 문자열 응답 자동 디코딩
                )
                # 연결 테스트
                cls._client.ping()
                print(f"Redis 서버 연결 성공 ({REDIS_HOST}:{REDIS_PORT})")
            except Exception as e:
                print(f"Redis 서버 연결 실패: {e}")
                cls._client = None
        return cls._client
    
    @staticmethod
    def generate_key(prefix: str, data: Any) -> str:
        """캐시 키 생성 함수"""
        if isinstance(data, str):
            serialized = data
        else:
            try:
                serialized = json.dumps(data, sort_keys=True)
            except (TypeError, ValueError):
                serialized = str(data)
        
        hash_value = hashlib.md5(serialized.encode('utf-8')).hexdigest()
        return f"{prefix}:{hash_value}"
    
    @classmethod
    def set(cls, key: str, value: Any, ttl: int = CACHE_TTL_DEFAULT) -> bool:
        """데이터 캐싱 (JSON 직렬화)"""
        client = cls.get_client()
        if not client:
            return False
        
        try:
            serialized = json.dumps(value)
            return client.setex(key, ttl, serialized)
        except Exception as e:
            print(f"Redis 캐싱 오류: {e}")
            return False
    
    @classmethod
    def get(cls, key: str) -> Optional[Any]:
        """캐시된 데이터 조회 (JSON 역직렬화)"""
        client = cls.get_client()
        if not client:
            return None
        
        try:
            data = client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"Redis 캐시 조회 오류: {e}")
            return None
    
    @classmethod
    def delete(cls, key: str) -> bool:
        """캐시 항목 삭제"""
        client = cls.get_client()
        if not client:
            return False
        
        try:
            return client.delete(key) > 0
        except Exception as e:
            print(f"Redis 캐시 삭제 오류: {e}")
            return False
    
    @classmethod
    def exists(cls, key: str) -> bool:
        """캐시 키 존재 여부 확인"""
        client = cls.get_client()
        if not client:
            return False
        
        try:
            return client.exists(key) > 0
        except Exception as e:
            print(f"Redis 캐시 확인 오류: {e}")
            return False
    
    @classmethod
    def flush_all(cls) -> bool:
        """모든 캐시 삭제 (주의: 전체 DB 삭제)"""
        client = cls.get_client()
        if not client:
            return False
        
        try:
            return client.flushdb()
        except Exception as e:
            print(f"Redis 캐시 초기화 오류: {e}")
            return False
    
    @classmethod
    def flush_by_pattern(cls, pattern: str) -> int:
        """패턴에 일치하는 키 삭제"""
        client = cls.get_client()
        if not client:
            return 0
        
        try:
            keys = client.keys(pattern)
            if keys:
                return client.delete(*keys)
            return 0
        except Exception as e:
            print(f"Redis 패턴 삭제 오류: {e}")
            return 0
            
    @classmethod
    def stream_response(cls, key: str, data_stream, ttl: int = CACHE_TTL_DEFAULT) -> bool:
        """스트리밍 응답 캐싱"""
        client = cls.get_client()
        if not client:
            return False
            
        try:
            # 스트림 데이터를 문자열로 변환하여 저장
            client.setex(key, ttl, json.dumps(list(data_stream)))
            return True
        except Exception as e:
            print(f"Redis 스트림 캐싱 오류: {e}")
            return False

# 캐시 키 접두사 (용도별 구분)
class CacheKeys:
    """캐시 키 접두사 상수"""
    SEARCH = "search"  # 검색 결과
    CHAT = "chat"      # 챗봇 응답
    VECTOR = "vector"  # 벡터 검색 결과
    CONVERSATION = "conv"  # 대화 내역
    SETTINGS = "settings"  # 사용자 설정
    SOURCE = "source"  # 출처 정보
    STATS = "stats"    # 통계 정보
    FEEDBACK = "feedback"  # 피드백 정보

# 캐시 통계 수집 함수 추가
async def get_cache_stats():
    """
    캐시 사용량 통계를 반환합니다.
    
    Returns:
        dict: 캐시 통계 정보
    """
    try:
        from fastapi import FastAPI
        from fastapi.applications import get_app
        
        # FastAPI 앱 객체 가져오기
        app = get_app()
        
        # 연결된 Redis 클라이언트 정보
        redis = await get_redis_client()
        if not redis:
            return {
                "status": "error",
                "message": "Redis 클라이언트를 초기화할 수 없습니다."
            }
        
        # Redis 정보 수집
        info = await redis.info()
        db_keys = await redis.dbsize()
        
        # 메모리 사용량
        used_memory = int(info.get('used_memory', 0))
        used_memory_human = info.get('used_memory_human', 'N/A')
        
        # 연결 및 캐시 통계
        total_connections = int(info.get('total_connections_received', 0))
        connected_clients = int(info.get('connected_clients', 0))
        
        # 캐시 적중률 계산
        hit_rate = 0
        if hasattr(app.state, 'cache_hit_count') and hasattr(app.state, 'total_cache_requests'):
            if app.state.total_cache_requests > 0:
                hit_rate = app.state.cache_hit_count / app.state.total_cache_requests
        
        return {
            "status": "success",
            "cache": {
                "total_keys": db_keys,
                "used_memory_bytes": used_memory,
                "used_memory_human": used_memory_human,
                "uptime_seconds": int(info.get('uptime_in_seconds', 0)),
                "hit_rate": hit_rate,
                "hit_count": getattr(app.state, 'cache_hit_count', 0),
                "miss_count": getattr(app.state, 'total_cache_requests', 0) - getattr(app.state, 'cache_hit_count', 0),
                "total_requests": getattr(app.state, 'total_cache_requests', 0),
            },
            "connection": {
                "total_connections": total_connections,
                "connected_clients": connected_clients,
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"캐시 통계 수집 중 오류: {str(e)}"
        }

# 비동기 Redis 클라이언트를 얻기 위한 함수
async def get_redis_client():
    """Redis 비동기 클라이언트를 반환합니다."""
    try:
        redis = await aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=REDIS_PASSWORD,
            encoding="utf-8",
            decode_responses=True,
        )
        await redis.ping()  # 연결 테스트
        return redis
    except Exception as e:
        print(f"Redis 연결 실패: {e}")
        return None 