import redis
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Redis 기반 캐싱 관리자
    - 쿼리 결과 캐싱
    - 벡터 검색 결과 캐싱
    - 자주 묻는 질문 응답 캐싱
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 6379, 
        db: int = 0, 
        password: str = None,
        prefix: str = "rag:",
        default_ttl: int = 86400 * 7  # 기본 7일 캐시
    ):
        """Redis 캐싱 관리자 초기화"""
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,  # 문자열로 디코딩
            socket_timeout=5,
            socket_connect_timeout=5
        )
        self.prefix = prefix
        self.default_ttl = default_ttl
        
        # 초기화 테스트
        try:
            self.redis_client.ping()
            logger.info("Redis 캐시 서버 연결 성공")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Redis 서버 연결 실패: {e}")
            # 연결 실패 시에도 계속 진행 (캐싱 기능 없이도 작동)
            self.redis_client = None
            
    def _get_key(self, key_type: str, value: str) -> str:
        """Redis 키 생성"""
        # 해시 적용하여 키 길이 제한 및 특수문자 문제 해결
        hashed = hashlib.md5(value.encode('utf-8')).hexdigest()
        return f"{self.prefix}{key_type}:{hashed}"
        
    def get(self, key_type: str, value: str) -> Optional[Dict[str, Any]]:
        """캐시에서 값 조회"""
        if not self.redis_client:
            return None
            
        try:
            key = self._get_key(key_type, value)
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"캐시 조회 중 오류: {e}")
            return None
    
    def set(self, key_type: str, value: str, data: Dict[str, Any], ttl: int = None) -> bool:
        """캐시에 값 저장"""
        if not self.redis_client:
            return False
            
        if ttl is None:
            ttl = self.default_ttl
            
        try:
            key = self._get_key(key_type, value)
            serialized = json.dumps(data, ensure_ascii=False)
            return self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"캐시 저장 중 오류: {e}")
            return False
    
    def delete(self, key_type: str, value: str) -> bool:
        """캐시에서 항목 삭제"""
        if not self.redis_client:
            return False
            
        try:
            key = self._get_key(key_type, value)
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"캐시 삭제 중 오류: {e}")
            return False
    
    def flush_category(self, category: str) -> bool:
        """특정 카테고리의 모든 캐시 삭제"""
        if not self.redis_client:
            return False
            
        try:
            pattern = f"{self.prefix}{category}:*"
            cursor = 0
            deleted = 0
            
            while True:
                cursor, keys = self.redis_client.scan(cursor, pattern, 100)
                if keys:
                    deleted += self.redis_client.delete(*keys)
                if cursor == 0:
                    break
                    
            return deleted > 0
        except Exception as e:
            logger.error(f"카테고리 캐시 삭제 중 오류: {e}")
            return False
            
    def get_query_cache(self, query: str, category: str) -> Optional[Dict[str, Any]]:
        """쿼리 캐시 조회 헬퍼 함수"""
        cache_key = f"{query}:{category}"
        return self.get("query", cache_key)
        
    def set_query_cache(self, query: str, category: str, result: Dict[str, Any], ttl: int = None) -> bool:
        """쿼리 캐시 저장 헬퍼 함수"""
        cache_key = f"{query}:{category}"
        return self.set("query", cache_key, result, ttl)
        
    def get_vector_search_cache(self, query: str, category: str) -> Optional[List[Dict[str, Any]]]:
        """벡터 검색 결과 캐시 조회"""
        cache_key = f"{query}:{category}"
        return self.get("vector", cache_key)
        
    def set_vector_search_cache(self, query: str, category: str, results: List[Dict[str, Any]], ttl: int = 3600) -> bool:
        """벡터 검색 결과 캐시 저장 (기본 1시간)"""
        cache_key = f"{query}:{category}"
        return self.set("vector", cache_key, results, ttl)
        
    def cache_conversation_history(self, user_id: str, conversation_id: str, history: List[Dict[str, Any]]) -> bool:
        """대화 내역 캐싱"""
        cache_key = f"{user_id}:{conversation_id}"
        # 대화 내역은 더 오래 유지 (30일)
        return self.set("history", cache_key, history, 86400 * 30)
        
    def get_conversation_history(self, user_id: str, conversation_id: str) -> Optional[List[Dict[str, Any]]]:
        """대화 내역 캐시 조회"""
        cache_key = f"{user_id}:{conversation_id}"
        return self.get("history", cache_key)
        
    def invalidate_query_cache(self, category: str = None) -> bool:
        """쿼리 캐시 무효화
        - 카테고리가 지정된 경우 해당 카테고리 캐시만 삭제
        - 카테고리가 None인 경우 모든 쿼리 캐시 삭제
        """
        if category:
            pattern = f"{self.prefix}query:*{category}*"
        else:
            pattern = f"{self.prefix}query:*"
            
        if not self.redis_client:
            return False
            
        try:
            cursor = 0
            deleted = 0
            
            while True:
                cursor, keys = self.redis_client.scan(cursor, pattern, 100)
                if keys:
                    deleted += self.redis_client.delete(*keys)
                if cursor == 0:
                    break
                    
            return deleted > 0
        except Exception as e:
            logger.error(f"쿼리 캐시 무효화 중 오류: {e}")
            return False
            
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 정보 조회"""
        if not self.redis_client:
            return {"status": "disconnected"}
            
        try:
            info = self.redis_client.info()
            stats = {
                "status": "connected",
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": 0
            }
            
            # 캐시 적중률 계산
            hits = info.get("keyspace_hits", 0)
            misses = info.get("keyspace_misses", 0)
            if hits + misses > 0:
                stats["hit_rate"] = round(hits / (hits + misses) * 100, 2)
                
            # 캐시 키 수 계산
            pattern = f"{self.prefix}*"
            cursor = 0
            key_count = 0
            
            while True:
                cursor, keys = self.redis_client.scan(cursor, pattern, 100)
                key_count += len(keys)
                if cursor == 0:
                    break
                    
            stats["cached_items"] = key_count
            
            return stats
        except Exception as e:
            logger.error(f"캐시 통계 조회 중 오류: {e}")
            return {"status": "error", "message": str(e)} 