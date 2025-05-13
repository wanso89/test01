import os
import json
import logging
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import importlib.util

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vanna_local")

# 상수 정의
DEFAULT_MODEL = os.environ.get("VANNA_MODEL", "mistralai/Mistral-7B-v0.1")
DEFAULT_EMBEDDING_MODEL = os.environ.get("VANNA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CACHE_DIR = os.environ.get("VANNA_CACHE_DIR", "./vanna_cache")
DEFAULT_API_KEY = os.environ.get("VANNA_API_KEY", "my-temp-local-key")

class VannaLocal:
    """로컬에서 Vanna AI를 실행하기 위한 클래스"""
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: str = DEFAULT_CACHE_DIR,
        api_key: str = DEFAULT_API_KEY,
        use_gpu: bool = True,
        gpu_layers: int = 40
    ):
        """
        Vanna Local 인스턴스를 초기화합니다.
        
        Args:
            model: 사용할 LLM 모델
            embedding_model: 사용할 임베딩 모델
            cache_dir: 캐시 디렉토리
            api_key: Vanna API 키 (로컬에서는 실제로 사용되지 않음)
            use_gpu: GPU 사용 여부
            gpu_layers: GPU에 로드할 레이어 수
        """
        self.model_name = model
        self.embedding_model_name = embedding_model
        self.cache_dir = cache_dir
        self.api_key = api_key
        self.use_gpu = use_gpu
        self.gpu_layers = gpu_layers
        
        # 내부 상태
        self.llm = None
        self.embedding_model = None
        self.vanna_instance = None
        self.is_initialized = False
        self.questions = {}  # 질문-SQL 쌍을 저장
        
        logger.info(f"VannaLocal 초기화: model={self.model_name}, embedding_model={self.embedding_model_name}")
        logger.info(f"GPU 사용: {self.use_gpu}, GPU 레이어: {self.gpu_layers}")
    
    def check_dependencies(self) -> bool:
        """
        필요한 의존성을 확인합니다.
        
        Returns:
            bool: 의존성 설치 여부
        """
        required_packages = ["vanna", "ctransformers", "sentence_transformers", "sqlalchemy"]
        missing_packages = []
        
        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"다음 패키지가 설치되어 있지 않습니다: {', '.join(missing_packages)}")
            logger.error(f"pip install {' '.join(missing_packages)} 명령으로 설치하세요.")
            return False
        
        return True
    
    def initialize(self) -> bool:
        """
        Vanna Local을 초기화합니다.
        
        Returns:
            bool: 초기화 성공 여부
        """
        if self.is_initialized:
            logger.info("이미 초기화된 Vanna Local 인스턴스를 재사용합니다.")
            return True
        
        if not self.check_dependencies():
            return False
        
        try:
            logger.info("모델 및 임베딩 모델 로드 중...")
            
            # 캐시 디렉토리 생성
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
            # 모델 로드 방식 결정 (최신 API vs 이전 API)
            vanna_version = self._get_vanna_version()
            
            if self._is_version_at_least(vanna_version, "0.7.0"):
                # 최신 API 사용
                logger.info(f"Vanna 버전 {vanna_version} 감지됨. 최신 API 사용.")
                return self._initialize_new_api()
            else:
                # 이전 API 사용
                logger.info(f"Vanna 버전 {vanna_version} 감지됨. 이전 API 사용.")
                return self._initialize_old_api()
                
        except Exception as e:
            logger.error(f"초기화 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return False
    
    def _get_vanna_version(self) -> str:
        """
        설치된 Vanna 버전을 확인합니다.
        
        Returns:
            str: Vanna 버전
        """
        try:
            import pkg_resources
            return pkg_resources.get_distribution("vanna").version
        except Exception as e:
            logger.warning(f"Vanna 버전 확인 중 오류: {str(e)}")
            return "0.0.0"  # 버전을 알 수 없는 경우 기본값
    
    def _is_version_at_least(self, version: str, min_version: str) -> bool:
        """
        버전이 최소 버전 이상인지 확인합니다.
        
        Args:
            version: 확인할 버전
            min_version: 최소 버전
            
        Returns:
            bool: 버전이 최소 버전 이상인지 여부
        """
        try:
            v_parts = list(map(int, version.split('.')))
            min_parts = list(map(int, min_version.split('.')))
            
            for i in range(max(len(v_parts), len(min_parts))):
                v = v_parts[i] if i < len(v_parts) else 0
                m = min_parts[i] if i < len(min_parts) else 0
                if v > m:
                    return True
                if v < m:
                    return False
            
            return True  # 완전히 동일한 버전
        except Exception as e:
            logger.warning(f"버전 비교 중 오류: {str(e)}")
            return False  # 오류 발생 시 안전하게 False 반환
    
    def _initialize_new_api(self) -> bool:
        """
        최신 Vanna API (0.7.0 이상)로 초기화합니다.
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            from vanna.remote import VannaDefault
            
            logger.info(f"모델 로드 중: {self.model_name}")
            
            # Vanna 인스턴스 생성
            self.vanna_instance = VannaDefault(
                model=self.model_name,
                api_key=self.api_key,
                config={
                    "use_local": True,
                    "embedding_model": self.embedding_model_name,
                    "cache_dir": self.cache_dir,
                    "use_gpu": self.use_gpu,
                    "gpu_layers": self.gpu_layers
                }
            )
            
            logger.info("Vanna 인스턴스 생성 성공")
            
            # 초기화 완료
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"최신 API 초기화 중 오류: {str(e)}")
            traceback.print_exc()
            return False
    
    def _initialize_old_api(self) -> bool:
        """
        이전 Vanna API (0.7.0 미만)로 초기화합니다.
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            import vanna
            
            # Vanna 설정
            vanna.set_api_key(self.api_key)
            
            # 로컬 모델 설정
            logger.info(f"로컬 모델 설정: {self.model_name}")
            vanna.set_model(self.model_name)
            
            # 임베딩 모델 설정
            if hasattr(vanna, "set_embedding_model"):
                logger.info(f"임베딩 모델 설정: {self.embedding_model_name}")
                vanna.set_embedding_model(self.embedding_model_name)
            
            # GPU 사용 설정
            if hasattr(vanna, "set_use_gpu"):
                logger.info(f"GPU 사용 설정: {self.use_gpu}")
                vanna.set_use_gpu(self.use_gpu)
            
            # GPU 레이어 설정
            if hasattr(vanna, "set_gpu_layers"):
                logger.info(f"GPU 레이어 설정: {self.gpu_layers}")
                vanna.set_gpu_layers(self.gpu_layers)
            
            # 캐시 디렉토리 설정
            if hasattr(vanna, "set_cache_dir"):
                logger.info(f"캐시 디렉토리 설정: {self.cache_dir}")
                vanna.set_cache_dir(self.cache_dir)
            
            # 인스턴스 저장
            self.vanna_instance = vanna
            
            # questions 속성 초기화
            if not hasattr(vanna, "questions") or vanna.questions is None:
                vanna.questions = {}
            
            # 초기화 완료
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"이전 API 초기화 중 오류: {str(e)}")
            traceback.print_exc()
            return False
    
    def connect_to_database(self, engine_or_connection_string: Any) -> bool:
        """
        데이터베이스에 연결합니다.
        
        Args:
            engine_or_connection_string: SQLAlchemy 엔진 또는 연결 문자열
            
        Returns:
            bool: 연결 성공 여부
        """
        if not self.is_initialized and not self.initialize():
            return False
        
        try:
            # 엔진 객체 확인
            from sqlalchemy import create_engine
            if isinstance(engine_or_connection_string, str):
                engine = create_engine(engine_or_connection_string)
            else:
                engine = engine_or_connection_string
            
            # 버전에 따라 다른 방법 사용
            if hasattr(self.vanna_instance, "set_engine") and callable(getattr(self.vanna_instance, "set_engine")):
                logger.info("set_engine 메서드로 엔진 설정")
                self.vanna_instance.set_engine(engine)
            elif hasattr(self.vanna_instance, "engine"):
                logger.info("engine 속성으로 엔진 설정")
                self.vanna_instance.engine = engine
            elif hasattr(self.vanna_instance, "connect_to_database") and callable(getattr(self.vanna_instance, "connect_to_database")):
                logger.info("connect_to_database 메서드로 엔진 설정")
                self.vanna_instance.connect_to_database(engine)
            else:
                logger.error("엔진을 설정할 방법을 찾을 수 없습니다")
                return False
            
            logger.info("데이터베이스 연결 성공")
            return True
            
        except Exception as e:
            logger.error(f"데이터베이스 연결 중 오류: {str(e)}")
            traceback.print_exc()
            return False
    
    def train_on_schema(self, schema: str) -> bool:
        """
        스키마 정보로 Vanna를 학습시킵니다.
        
        Args:
            schema: 데이터베이스 스키마 정보
            
        Returns:
            bool: 학습 성공 여부
        """
        if not self.is_initialized and not self.initialize():
            return False
        
        try:
            # 스키마 학습
            if hasattr(self.vanna_instance, "train") and callable(getattr(self.vanna_instance, "train")):
                logger.info("train 메서드로 스키마 학습")
                self.vanna_instance.train(ddl=schema)
            elif hasattr(self.vanna_instance, "add_ddl") and callable(getattr(self.vanna_instance, "add_ddl")):
                logger.info("add_ddl 메서드로 스키마 학습")
                self.vanna_instance.add_ddl(schema)
            else:
                logger.error("스키마 학습을 위한 적절한 메서드를 찾을 수 없습니다")
                return False
            
            logger.info("스키마 학습 성공")
            return True
            
        except Exception as e:
            logger.error(f"스키마 학습 중 오류: {str(e)}")
            traceback.print_exc()
            return False
    
    def generate_sql(self, question: str) -> str:
        """
        자연어 질문을 SQL 쿼리로 변환합니다.
        
        Args:
            question: 자연어 질문
            
        Returns:
            str: 생성된 SQL 쿼리
        """
        if not self.is_initialized and not self.initialize():
            return None
        
        try:
            # SQL 생성 메서드 사용
            if hasattr(self.vanna_instance, "generate_sql") and callable(getattr(self.vanna_instance, "generate_sql")):
                logger.info(f"generate_sql 메서드로 SQL 생성: {question}")
                sql = self.vanna_instance.generate_sql(question)
                logger.info(f"생성된 SQL: {sql}")
                return sql
            
            # ask 메서드 사용
            elif hasattr(self.vanna_instance, "ask") and callable(getattr(self.vanna_instance, "ask")):
                logger.info(f"ask 메서드로 SQL 생성: {question}")
                result = self.vanna_instance.ask(question)
                logger.info(f"생성된 SQL: {result}")
                return result
            
            else:
                logger.error("SQL 생성 메서드를 찾을 수 없습니다")
                return None
                
        except Exception as e:
            logger.error(f"SQL 생성 중 오류: {str(e)}")
            traceback.print_exc()
            return None
    
    def run_sql(self, sql: str) -> List[Dict]:
        """
        SQL 쿼리를 실행합니다.
        
        Args:
            sql: SQL 쿼리
            
        Returns:
            List[Dict]: 쿼리 결과
        """
        if not self.is_initialized and not self.initialize():
            return []
        
        try:
            # Vanna run_sql 메서드 사용
            if hasattr(self.vanna_instance, "run_sql") and callable(getattr(self.vanna_instance, "run_sql")):
                logger.info(f"run_sql 메서드로 SQL 실행: {sql}")
                result = self.vanna_instance.run_sql(sql)
                return result
            
            # 직접 실행
            else:
                logger.error("run_sql 메서드를 찾을 수 없습니다")
                return []
                
        except Exception as e:
            logger.error(f"SQL 실행 중 오류: {str(e)}")
            traceback.print_exc()
            return []
    
    def add_question_sql_pair(self, question: str, sql: str) -> bool:
        """
        질문-SQL 쌍을 추가합니다. Vanna의 학습 데이터로 사용됩니다.
        
        Args:
            question: 자연어 질문
            sql: SQL 쿼리
            
        Returns:
            bool: 성공 여부
        """
        if not self.is_initialized and not self.initialize():
            return False
        
        try:
            # 질문-SQL 쌍 저장
            if not hasattr(self, "questions"):
                self.questions = {}
            
            self.questions[question] = sql
            
            # train 메서드 사용
            if hasattr(self.vanna_instance, "train") and callable(getattr(self.vanna_instance, "train")):
                logger.info(f"train 메서드로 질문-SQL 쌍 추가: {question} -> {sql}")
                self.vanna_instance.train(question=question, sql=sql)
                return True
            
            # add_question_sql 메서드 사용
            elif hasattr(self.vanna_instance, "add_question_sql") and callable(getattr(self.vanna_instance, "add_question_sql")):
                logger.info(f"add_question_sql 메서드로 질문-SQL 쌍 추가: {question} -> {sql}")
                self.vanna_instance.add_question_sql(question, sql)
                return True
            
            else:
                logger.warning("질문-SQL 쌍을 추가할 수 있는 메서드를 찾을 수 없습니다. 로컬에만 저장합니다.")
                return True  # 로컬 저장은 성공
                
        except Exception as e:
            logger.error(f"질문-SQL 쌍 추가 중 오류: {str(e)}")
            traceback.print_exc()
            return False
    
    def save_state(self, file_path: str = "vanna_state.json") -> bool:
        """
        현재 상태를 파일로 저장합니다.
        
        Args:
            file_path: 저장할 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 저장할 상태 구성
            state = {
                "questions": self.questions,
                "model": self.model_name,
                "embedding_model": self.embedding_model_name,
                "cache_dir": self.cache_dir,
                "use_gpu": self.use_gpu,
                "gpu_layers": self.gpu_layers
            }
            
            # 파일로 저장
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            logger.info(f"상태를 {file_path}에 저장했습니다")
            return True
            
        except Exception as e:
            logger.error(f"상태 저장 중 오류: {str(e)}")
            traceback.print_exc()
            return False
    
    def load_state(self, file_path: str = "vanna_state.json") -> bool:
        """
        파일에서 상태를 로드합니다.
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            bool: 로드 성공 여부
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"{file_path} 파일이 존재하지 않습니다")
                return False
            
            # 파일에서 로드
            with open(file_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            
            # 상태 복원
            self.questions = state.get("questions", {})
            self.model_name = state.get("model", self.model_name)
            self.embedding_model_name = state.get("embedding_model", self.embedding_model_name)
            self.cache_dir = state.get("cache_dir", self.cache_dir)
            self.use_gpu = state.get("use_gpu", self.use_gpu)
            self.gpu_layers = state.get("gpu_layers", self.gpu_layers)
            
            # 재초기화
            self.is_initialized = False
            success = self.initialize()
            
            logger.info(f"{file_path}에서 상태를 로드했습니다")
            return success
            
        except Exception as e:
            logger.error(f"상태 로드 중 오류: {str(e)}")
            traceback.print_exc()
            return False

# 싱글톤 인스턴스
_vanna_local = None

def get_vanna_local(force_new: bool = False) -> VannaLocal:
    """
    VannaLocal 싱글톤 인스턴스를 가져옵니다.
    
    Args:
        force_new: 새 인스턴스 강제 생성 여부
        
    Returns:
        VannaLocal: VannaLocal 인스턴스
    """
    global _vanna_local
    
    if _vanna_local is None or force_new:
        _vanna_local = VannaLocal()
        _vanna_local.initialize()
    
    return _vanna_local

if __name__ == "__main__":
    # 테스트 코드
    logger.info("Vanna Local 테스트 시작")
    
    vanna = get_vanna_local()
    
    # 간단한 SQL 생성 테스트
    test_question = "사용자 목록을 보여줘"
    logger.info(f"테스트 질문: {test_question}")
    
    sql = vanna.generate_sql(test_question)
    
    if sql:
        logger.info(f"생성된 SQL: {sql}")
    else:
        logger.error("SQL 생성 실패")
    
    logger.info("Vanna Local 테스트 종료") 