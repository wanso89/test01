import os
import json
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from sqlalchemy import create_engine, text

# MariaDB 연결 설정
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "3Ssoft1!",
    "database": "chatbot"
}

# Vanna 설정
VANNA_API_KEY = os.environ.get("VANNA_API_KEY", "my-temp-local-key")
VANNA_MODEL = os.environ.get("VANNA_MODEL", "mistralai/Mistral-7B-v0.1")  # 로컬에서 사용할 모델
VANNA_EMBEDDING_MODEL = os.environ.get("VANNA_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

class VannaInterface:
    """Vanna AI를 사용하기 위한 인터페이스 클래스"""
    
    def __init__(self, db_config=None, api_key=None, model=None, embedding_model=None, use_local=True):
        """
        Vanna 인터페이스를 초기화합니다.
        
        Args:
            db_config: 데이터베이스 연결 설정
            api_key: Vanna API 키
            model: 사용할 모델
            embedding_model: 사용할 임베딩 모델
            use_local: 로컬 모드 사용 여부
        """
        self.db_config = db_config or DB_CONFIG
        self.api_key = api_key or VANNA_API_KEY
        self.model = model or VANNA_MODEL
        self.embedding_model = embedding_model or VANNA_EMBEDDING_MODEL
        self.use_local = use_local
        self.vanna_instance = None
        self.engine = None
        self.is_initialized = False
        self.schema = None
        
        print(f"VannaInterface 초기화: model={self.model}, embedding_model={self.embedding_model}, use_local={self.use_local}")
    
    def initialize(self):
        """Vanna AI를 초기화합니다."""
        if self.is_initialized and self.vanna_instance:
            print("이미 초기화된 Vanna 인스턴스를 재사용합니다.")
            return True
            
        try:
            print("Vanna AI 초기화 중...")
            
            # SQLAlchemy 엔진 설정
            connection_string = f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/{self.db_config['database']}"
            self.engine = create_engine(connection_string)
            
            # 연결 테스트
            try:
                with self.engine.connect() as conn:
                    result = conn.execute(text("SELECT 1")).fetchone()
                    print(f"SQLAlchemy 연결 성공: {result}")
            except Exception as e:
                print(f"SQLAlchemy 연결 오류: {str(e)}")
                traceback.print_exc()
                return False
            
            # Vanna AI 초기화 (버전에 따라 다른 방법 사용)
            try:
                # 최신 API 사용 (버전 0.7.0 이상)
                from vanna.remote import VannaDefault
                
                # Vanna AI 인스턴스 생성
                self.vanna_instance = VannaDefault(
                    model=self.model,
                    api_key=self.api_key,
                    config={
                        "use_local": self.use_local,
                        "embedding_model": self.embedding_model
                    }
                )
                
                # 엔진 설정
                if hasattr(self.vanna_instance, "set_engine") and callable(getattr(self.vanna_instance, "set_engine")):
                    self.vanna_instance.set_engine(self.engine)
                    print("set_engine 메서드로 SQLAlchemy 엔진 설정 완료")
                elif hasattr(self.vanna_instance, "engine"):
                    self.vanna_instance.engine = self.engine
                    print("engine 속성으로 SQLAlchemy 엔진 설정 완료")
                else:
                    print("Vanna 인스턴스에 엔진을 설정할 방법이 없습니다")
                    return False
                
                print(f"Vanna API 초기화 성공: {self.vanna_instance}")
                
            except ImportError:
                # 이전 API 사용 (버전 0.7.0 미만)
                try:
                    import vanna
                    
                    # Vanna 설정
                    vanna.set_api_key(self.api_key)
                    if self.use_local:
                        vanna.set_model(self.model)
                        if hasattr(vanna, "set_embedding_model"):
                            vanna.set_embedding_model(self.embedding_model)
                    
                    # 엔진 설정
                    vanna.connect_to_database(self.engine)
                    
                    self.vanna_instance = vanna
                    print("이전 Vanna API 초기화 성공")
                    
                except Exception as old_vanna_err:
                    print(f"이전 Vanna API 초기화 실패: {old_vanna_err}")
                    traceback.print_exc()
                    return False
            
            # questions 속성 초기화
            try:
                if not hasattr(self.vanna_instance, "questions") or self.vanna_instance.questions is None:
                    setattr(self.vanna_instance, "questions", {})
                    print("questions 속성을 초기화했습니다.")
            except Exception as attr_err:
                print(f"questions 속성 초기화 중 오류: {attr_err}")
            
            # 스키마 로드 및 학습
            from .get_mariadb_schema import get_mariadb_schema
            self.schema = get_mariadb_schema(self.db_config)
            
            # 스키마 학습
            try:
                if hasattr(self.vanna_instance, "train") and callable(getattr(self.vanna_instance, "train")):
                    print("train 메서드로 스키마 학습")
                    self.vanna_instance.train(ddl=self.schema)
                elif hasattr(self.vanna_instance, "add_ddl") and callable(getattr(self.vanna_instance, "add_ddl")):
                    print("add_ddl 메서드로 스키마 학습")
                    self.vanna_instance.add_ddl(self.schema)
                else:
                    print("스키마 학습을 위한 적절한 메서드를 찾을 수 없습니다.")
            except Exception as train_err:
                print(f"스키마 학습 중 오류 (무시됨): {train_err}")
                pass
            
            # 테스트 쿼리 실행
            test_question = "사용자 목록을 보여줘"
            try:
                self.generate_sql(test_question)
                print("테스트 SQL 생성 성공")
            except Exception as test_err:
                print(f"테스트 SQL 생성 실패: {str(test_err)}")
                # 테스트 실패는 무시하고 계속 진행
            
            self.is_initialized = True
            print("Vanna AI 초기화 완료")
            return True
            
        except Exception as e:
            print(f"Vanna AI 초기화 오류: {str(e)}")
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
        if not self.is_initialized:
            success = self.initialize()
            if not success:
                print("Vanna AI 초기화 실패")
                return None
        
        try:
            # 직접 SQL 생성
            if hasattr(self.vanna_instance, "generate_sql") and callable(getattr(self.vanna_instance, "generate_sql")):
                sql = self.vanna_instance.generate_sql(question)
                print(f"generate_sql 메서드로 생성된 SQL: {sql}")
                return sql
            
            # ask 메서드 사용
            elif hasattr(self.vanna_instance, "ask") and callable(getattr(self.vanna_instance, "ask")):
                result = self.vanna_instance.ask(question)
                print(f"ask 메서드로 생성된 SQL: {result}")
                return result
            
            else:
                print("SQL 생성 메서드를 찾을 수 없습니다.")
                return None
                
        except Exception as e:
            print(f"SQL 생성 중 오류: {str(e)}")
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
        if not self.is_initialized:
            success = self.initialize()
            if not success:
                print("Vanna AI 초기화 실패")
                return []
        
        try:
            # Vanna run_sql 메서드 사용
            if hasattr(self.vanna_instance, "run_sql") and callable(getattr(self.vanna_instance, "run_sql")):
                result = self.vanna_instance.run_sql(sql)
                return result
            
            # 직접 실행
            else:
                with self.engine.connect() as conn:
                    result = conn.execute(text(sql))
                    columns = result.keys()
                    rows = []
                    for row in result:
                        rows.append({col: value for col, value in zip(columns, row)})
                    return rows
                    
        except Exception as e:
            print(f"SQL 실행 중 오류: {str(e)}")
            traceback.print_exc()
            return []

    def get_schema(self) -> str:
        """
        데이터베이스 스키마를 가져옵니다.
        
        Returns:
            str: 데이터베이스 스키마 정보
        """
        if self.schema is None:
            from .get_mariadb_schema import get_mariadb_schema
            self.schema = get_mariadb_schema(self.db_config)
        
        return self.schema
    
    def add_question_sql_pair(self, question: str, sql: str) -> bool:
        """
        질문-SQL 쌍을 추가합니다. Vanna의 학습 데이터로 사용됩니다.
        
        Args:
            question: 자연어 질문
            sql: SQL 쿼리
            
        Returns:
            bool: 성공 여부
        """
        if not self.is_initialized:
            success = self.initialize()
            if not success:
                print("Vanna AI 초기화 실패")
                return False
        
        try:
            # train 메서드 사용
            if hasattr(self.vanna_instance, "train") and callable(getattr(self.vanna_instance, "train")):
                self.vanna_instance.train(question=question, sql=sql)
                print(f"train 메서드로 질문-SQL 쌍 추가 성공: {question} -> {sql}")
                return True
            
            # add_question_sql 메서드 사용
            elif hasattr(self.vanna_instance, "add_question_sql") and callable(getattr(self.vanna_instance, "add_question_sql")):
                self.vanna_instance.add_question_sql(question, sql)
                print(f"add_question_sql 메서드로 질문-SQL 쌍 추가 성공: {question} -> {sql}")
                return True
            
            else:
                print("질문-SQL 쌍을 추가할 수 있는 메서드를 찾을 수 없습니다.")
                return False
                
        except Exception as e:
            print(f"질문-SQL 쌍 추가 중 오류: {str(e)}")
            traceback.print_exc()
            return False

# 싱글톤 인스턴스
_vanna_interface = None

def get_vanna_interface(force_new=False):
    """
    Vanna 인터페이스 인스턴스를 가져옵니다.
    
    Args:
        force_new: 새 인스턴스 강제 생성 여부
        
    Returns:
        VannaInterface: Vanna 인터페이스 인스턴스
    """
    global _vanna_interface
    
    if _vanna_interface is None or force_new:
        _vanna_interface = VannaInterface()
        _vanna_interface.initialize()
    
    return _vanna_interface 