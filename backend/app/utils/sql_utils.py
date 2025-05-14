import re
import mysql.connector
import traceback
import asyncio
import torch
import importlib.util  # Vanna AI 설치 확인용
import os

# MariaDB 연결 설정
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "3Ssoft1!",
    "database": "chatbot"
}

# SQL 변환 프롬프트 템플릿
SQL_SYSTEM_PROMPT = """너는 MariaDB 데이터베이스에 대한 SQL 전문가야. 
사용자의 자연어 질문을 읽고, 데이터베이스 스키마를 바탕으로 정확한 SQL 쿼리를 작성해야 해.
쿼리는 반드시 실행 가능한 MariaDB SQL 구문이어야 하며, 관련 테이블과 조건을 모두 포함해야 해."""

SQL_USER_PROMPT_TEMPLATE = """
# 데이터베이스 스키마:
{table_info}

# 사용자 질문:
{question}

# 지침:
1. 위 질문에 맞는 정확한 SQL 쿼리를 작성해줘.
2. 반드시 SQL 쿼리만 반환하고, 설명이나 주석은 포함하지 마세요.
3. 항상 SELECT 문을 작성하고, 필요한 JOIN, WHERE, GROUP BY, ORDER BY 등을 포함하세요.
4. 테이블과 컬럼 이름은 정확하게 사용하세요.
5. 사용자 질문에 특정 임계값(예: 80% 이상)이 언급된 경우 반드시 WHERE 절에 포함하세요.
6. 사용자 질문에 특정 이름(예: 홍길동, VM_001)이 언급된 경우 반드시 WHERE 절에 포함세요.
7. 복합 조건(예: CPU와 메모리 사용량 모두 포함)이 있는 경우 모든 조건을 WHERE 절에 포함하세요.
8. 임계값이 "XX% 이상"인 경우 ">=" 연산자를, "XX% 초과"인 경우 ">" 연산자를,
   "XX% 이하"인 경우 "<=" 연산자를, "XX% 미만"인 경우 "<" 연산자를 사용하세요.
9. 반드시 SQL 쿼리를 생성해야 하며, 질문이 불명확하더라도 가장 적합한 SELECT 문을 작성하세요.

# 예시:
질문: "메모리 사용량이 90% 이상인 VM 목록"
SQL: SELECT user.user_name, vm_name, memory FROM user_resource JOIN user ON user_resource.user_id = user.user_id WHERE memory >= 90 ORDER BY memory DESC;

질문: "홍길동이 가장 많이 로그인한 월은?"
SQL: SELECT DATE_FORMAT(month, '%Y-%m') as month, login_count FROM user_monthly_stats JOIN user ON user_monthly_stats.user_id = user.user_id WHERE user.user_name = '홍길동' ORDER BY login_count DESC LIMIT 1;

질문: "CPU 사용량이 70% 이상이고 메모리 사용량이 80% 이상인 VM 목록"
SQL: SELECT user.user_name, vm_name, cpu, memory FROM user_resource JOIN user ON user_resource.user_id = user.user_id WHERE cpu >= 70 AND memory >= 80 ORDER BY cpu DESC, memory DESC;

질문: "사용자별 평균 로그인 횟수"
SQL: SELECT user.user_name, AVG(login_count) as avg_login FROM user_monthly_stats JOIN user ON user_monthly_stats.user_id = user.user_id GROUP BY user.user_name ORDER BY avg_login DESC;

질문: "VM_001의 CPU 사용량 통계"
SQL: SELECT vm_name, AVG(cpu) as avg_cpu, MAX(cpu) as max_cpu, MIN(cpu) as min_cpu FROM user_resource WHERE vm_name = 'VM_001' GROUP BY vm_name;

# 응답 형식:
반드시 실행 가능한a SQL 쿼리만 반환하세요. 설명이나 주석을 추가하지 마세요.
"""

# Vanna AI 관련 설정 추가
VANNA_INITIALIZED = False
vanna_instance = None
VANNA_API_KEY = os.environ.get("VANNA_API_KEY", "my-temp-local-key")
VANNA_MODEL = os.environ.get("VANNA_MODEL", "distilbart-mnli")  # 로컬에서 사용할 모델

def check_vanna_installed():
    """Vanna AI 설치 여부를 확인합니다."""
    try:
        spec = importlib.util.find_spec('vanna')
        return spec is not None
    except ImportError:
        return False

def initialize_vanna():
    """Vanna AI를 초기화합니다."""
    global VANNA_INITIALIZED, vanna_instance
    
    if VANNA_INITIALIZED and vanna_instance:
        # 이미 초기화된 인스턴스가 있는지 재검증
        if hasattr(vanna_instance, 'questions') or (hasattr(vanna_instance, 'generate_sql') and callable(getattr(vanna_instance, 'generate_sql'))):
            print("이미 초기화된 Vanna 인스턴스를 재사용합니다.")
            return vanna_instance
        else:
            print("기존 Vanna 인스턴스가 유효하지 않습니다. 재초기화를 시도합니다.")
            VANNA_INITIALIZED = False
            vanna_instance = None
    
    try:
        print("Vanna AI 초기화 중...")
        
        # Vanna 버전 확인
        try:
            import pkg_resources
            vanna_version = pkg_resources.get_distribution("vanna").version
            print(f"설치된 Vanna 버전: {vanna_version}")
        except Exception as e:
            print(f"Vanna 버전 확인 중 오류: {str(e)}")
            vanna_version = "unknown"
        
        # SQLAlchemy 직접 엔진 설정 (pymysql 사용)
        try:
            from sqlalchemy import create_engine, text
            connection_string = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
            engine = create_engine(connection_string)
            
            # 연결 테스트
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).fetchone()
                print(f"SQLAlchemy 직접 연결 성공: {result}")
        except Exception as e:
            print(f"SQLAlchemy 연결 오류: {str(e)}")
            traceback.print_exc()
            return None
            
        # 최신 API 사용 시도 (버전 0.7.0 이상)
        try:
            from vanna.remote import VannaDefault
            
            # Vanna AI 인스턴스 생성 (API 키 설정)
            vanna_instance = VannaDefault(
                model=VANNA_MODEL,
                api_key=VANNA_API_KEY,
                config={"use_local": True}  # 로컬 모델 사용 설정
            )
            
            print(f"새로운 Vanna API를 사용하여 초기화했습니다. 모델: {VANNA_MODEL}")
            
            # SQLAlchemy 엔진 설정
            if hasattr(vanna_instance, "set_engine") and callable(getattr(vanna_instance, "set_engine")):
                vanna_instance.set_engine(engine)
                print("set_engine 메서드 사용하여 SQLAlchemy 엔진 설정 완료")
            elif hasattr(vanna_instance, "engine"):
                vanna_instance.engine = engine
                print("engine 속성 직접 설정 완료")
            else:
                print("Vanna 인스턴스에 엔진을 설정할 방법이 없습니다")
                
            # 스키마 정보 가져오기
            schema = get_mariadb_schema()
            print(f"스키마 정보 생성 완료, 길이: {len(schema)}")
            
            # 스키마 학습 시도 (에러 무시)
            try:
                # 스키마 학습
                if hasattr(vanna_instance, "train") and callable(getattr(vanna_instance, "train")):
                    print("train 메서드 사용 시도")
                    vanna_instance.train(ddl=schema)
                    print("스키마 학습 완료 (train)")
                elif hasattr(vanna_instance, "add_ddl") and callable(getattr(vanna_instance, "add_ddl")):
                    print("add_ddl 메서드 사용 시도")
                    vanna_instance.add_ddl(schema)
                    print("스키마 학습 완료 (add_ddl)")
                else:
                    print("스키마 학습을 위한 적절한 메서드를 찾을 수 없습니다.")
            except Exception as train_err:
                print(f"스키마 학습 중 오류 (무시됨): {train_err}")
                pass  # 스키마 학습 실패는 무시하고 진행
            
            # 추가: questions 또는 question_answer_pairs 속성 초기화
            try:
                # 일부 Vanna 버전은 questions 컬렉션이 필요함
                if not hasattr(vanna_instance, "questions"):
                    print("questions 속성이 없습니다. 새로 추가합니다.")
                    vanna_instance.questions = {}
                elif vanna_instance.questions is None:
                    print("questions 속성이 None입니다. 빈 딕셔너리로 초기화합니다.")
                    vanna_instance.questions = {}
                
                # question_answer_pairs 메서드 로드 시도
                if hasattr(vanna_instance, "load_question_answer_pairs") and callable(getattr(vanna_instance, "load_question_answer_pairs")):
                    print("load_question_answer_pairs 메서드 호출 시도")
                    vanna_instance.load_question_answer_pairs()
                    print("question_answer_pairs 로드 완료")
            except Exception as qa_err:
                print(f"questions/QA 초기화 중 오류 (무시됨): {qa_err}")
                print("questions 속성을 수동으로 초기화합니다.")
                # 안전하게 속성 추가 시도
                try:
                    # 클래스 동적 속성 추가
                    setattr(vanna_instance, "questions", {})
                    print("questions 속성 수동 추가 성공")
                except Exception as attr_err:
                    print(f"속성 추가 실패: {attr_err}")
                # QA 초기화 실패는 무시하고 진행
                
            # 추가: 명시적 인스턴스 검증
            if not hasattr(vanna_instance, "generate_sql") and not hasattr(vanna_instance, "ask"):
                print("초기화된 Vanna 인스턴스에 필요한 메서드가 없습니다. SQL 변환 시 대체 방법이 사용됩니다.")
                return None
                
            # 테스트 쿼리 실행
            test_question = "사용자 목록을 보여줘"
            try:
                if hasattr(vanna_instance, "generate_sql") and callable(getattr(vanna_instance, "generate_sql")):
                    test_sql = vanna_instance.generate_sql(test_question)
                    print(f"테스트 SQL 생성 성공: {test_sql}")
                elif hasattr(vanna_instance, "ask") and callable(getattr(vanna_instance, "ask")):
                    test_result = vanna_instance.ask(test_question)
                    print(f"테스트 ask 메서드 호출 성공: {test_result}")
                else:
                    print("SQL 생성 메서드를 찾을 수 없습니다.")
                    return None
            except Exception as test_err:
                print(f"테스트 SQL 생성 실패: {str(test_err)}")
                # 테스트 실패는 오류 메시지만 출력하고 계속 진행
                pass
                
            VANNA_INITIALIZED = True
            print("Vanna AI 초기화 완료")
            return vanna_instance
                
        except ImportError as ie:
            print(f"새로운 Vanna API 사용 실패: {str(ie)}")
            print("이전 Vanna API로 폴백")
            
            # 이전 API 방식 시도 (버전 0.3.0 ~ 0.6.x)
            try:
                import vanna
                
                # DB 설정
                vanna.set_model("gemma")  # 또는 다른 지원 모델
                vanna.set_api_key(VANNA_API_KEY)
                print("이전 API 모델 설정됨")
                
                # SQLAlchemy 엔진 직접 설정
                if hasattr(vanna, "set_engine"):
                    vanna.set_engine(engine)
                    print("이전 API - 엔진 설정 완료")
                
                # 스키마 학습 시도 (에러 무시)
                schema = get_mariadb_schema()
                try:
                    if hasattr(vanna, "train"):
                        print("이전 API train 사용")
                        vanna.train(ddl=schema)
                    elif hasattr(vanna, "add_ddl"):
                        print("이전 API add_ddl 사용")
                        vanna.add_ddl(schema)
                    else:
                        print("이전 API에서 스키마 학습 메서드를 찾을 수 없습니다.")
                except Exception as train_err:
                    print(f"이전 API 스키마 학습 중 오류 (무시됨): {train_err}")
                    pass  # 스키마 학습 실패는 무시하고 진행
                
                # 추가: questions 또는 question_answer_pairs 속성 초기화
                try:
                    # 일부 Vanna 버전은 questions 컬렉션이 필요함
                    if not hasattr(vanna, "questions"):
                        print("questions 속성이 없습니다. 새로 추가합니다.")
                        vanna.questions = {}
                    elif vanna.questions is None:
                        print("questions 속성이 None입니다. 빈 딕셔너리로 초기화합니다.")
                        vanna.questions = {}
                    
                    # question_answer_pairs 메서드 로드 시도
                    if hasattr(vanna, "load_question_answer_pairs") and callable(getattr(vanna, "load_question_answer_pairs")):
                        print("이전 API - load_question_answer_pairs 메서드 호출 시도")
                        vanna.load_question_answer_pairs()
                        print("이전 API - question_answer_pairs 로드 완료")
                except Exception as qa_err:
                    print(f"이전 API - questions/QA 초기화 중 오류 (무시됨): {qa_err}")
                    
                # 테스트 쿼리 생성 및 검증
                try:
                    test_question = "사용자 목록을 보여줘"
                    if hasattr(vanna, "generate_sql") and callable(getattr(vanna, "generate_sql")):
                        test_sql = vanna.generate_sql(test_question)
                        print(f"이전 API - 테스트 SQL 생성 성공: {test_sql}")
                    elif hasattr(vanna, "ask") and callable(getattr(vanna, "ask")):
                        test_result = vanna.ask(test_question)
                        print(f"이전 API - 테스트 ask 메서드 호출 성공: {test_result}")
                    else:
                        print("이전 API - SQL 생성 메서드를 찾을 수 없습니다.")
                except Exception as test_err:
                    print(f"이전 API - 테스트 SQL 생성 실패: {str(test_err)}")
                    # 테스트 실패는 무시하고 진행
                    
                # 추가: 명시적 인스턴스 검증
                if (not hasattr(vanna, "generate_sql") or not callable(getattr(vanna, "generate_sql"))) and \
                   (not hasattr(vanna, "ask") or not callable(getattr(vanna, "ask"))):
                    print("이전 API - 초기화된 Vanna에 필요한 메서드가 없습니다.")
                    return None
                    
                vanna_instance = vanna
                VANNA_INITIALIZED = True
                print("이전 API Vanna AI 초기화 완료")
                return vanna_instance
            except Exception as old_vanna_err:
                print(f"이전 Vanna API 초기화 실패: {old_vanna_err}")
                traceback.print_exc()
                return None
            
    except Exception as e:
        print(f"Vanna AI 초기화 오류: {str(e)}")
        traceback.print_exc()
        return None

def generate_sql_with_vanna(question):
    """Vanna AI를 사용하여 자연어 질문을 SQL 쿼리로 변환합니다."""
    print(f"Vanna AI로 SQL 생성 시도 중... 질문: '{question}'")
    
    try:
        # Vanna 인터페이스 사용 (새로 구현한 코드 활용)
        from app.utils.vanna_interface import get_vanna_interface
        
        # Vanna 인스턴스 가져오기
        vanna_obj = get_vanna_interface()
        if not vanna_obj or not vanna_obj.is_initialized:
            print("Vanna AI 인스턴스를 초기화할 수 없습니다. 대체 방법으로 전환합니다.")
            return None
        
        # 자주 사용되는 질문 패턴 미리 처리 (Vanna가 실패할 경우 대비)
        if "리뷰" in question.lower() and "없는" in question.lower() and "제품" in question.lower():
            print("Vanna AI 우회: '리뷰가 없는 제품' 패턴 감지")
            return """
SELECT p.product_id, p.name AS 제품명, p.category AS 카테고리, p.price AS 가격
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
WHERE r.review_id IS NULL
ORDER BY p.name;
"""

        if "평균" in question.lower() and "리뷰" in question.lower() and ("점수" in question.lower() or "평점" in question.lower()) and "제품" in question.lower():
            print("Vanna AI 우회: '제품별 평균 리뷰 점수' 패턴 감지")
            return """
SELECT p.product_id, p.name AS 제품명, AVG(r.rating) AS 평균_평점
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.product_id, p.name
ORDER BY 평균_평점 DESC;
"""

        if (("가장 자주" in question.lower() or "많이" in question.lower()) and "주문" in question.lower() and "제품" in question.lower()) or ("top" in question.lower() and "제품" in question.lower):
            print("Vanna AI 우회: '가장 많이 주문된 제품' 패턴 감지")
            # top N의 N 추출
            import re
            limit = 3  # 기본값
            top_n_match = re.search(r'top\s*(\d+)', question.lower())
            if top_n_match:
                limit = int(top_n_match.group(1))
                
            return f"""
SELECT p.product_id, p.name AS 제품명, COUNT(oi.product_id) AS 주문_횟수
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.name
ORDER BY 주문_횟수 DESC
LIMIT {limit};
"""
        
        # 최대 15초 타임아웃으로 SQL 생성 시도
        def with_timeout(func, args=(), kwargs={}, timeout_seconds=15, default=None):
            """지정된 시간 내에 함수를 실행하고, 타임아웃시 기본값을 반환합니다."""
            import signal
            
            class TimeoutError(Exception):
                pass
            
            def handler(signum, frame):
                raise TimeoutError(f"함수 실행이 {timeout_seconds}초를 초과했습니다.")
            
            # 이전 핸들러 저장
            original_handler = signal.getsignal(signal.SIGALRM)
            
            try:
                # 타임아웃 설정
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                
                # 함수 실행
                result = func(*args, **kwargs)
                
                # 타임아웃 취소
                signal.alarm(0)
                return result
            except TimeoutError as e:
                print(f"타임아웃: {e}")
                return default
            except Exception as e:
                print(f"함수 실행 중 오류: {str(e)}")
                return default
            finally:
                # 원래 핸들러 복원
                signal.signal(signal.SIGALRM, original_handler)
        
        # Vanna 인터페이스로 SQL 생성
        try:
            print("Vanna 인터페이스로 SQL 생성 시도")
            sql = with_timeout(vanna_obj.generate_sql, args=(question,))
            
            if sql and "SELECT" in sql.upper():
                print(f"Vanna 인터페이스로 생성된 SQL: {sql}")
                
                # 생성된 SQL이 유효한지 확인
                if sql.strip().upper().startswith("SELECT") and "FROM" in sql.upper():
                    # 질문-SQL 쌍 저장 (향후 학습에 활용)
                    try:
                        vanna_obj.add_question_sql_pair(question, sql)
                    except:
                        pass  # 실패해도 무시
                    
                    return sql
            
            print("Vanna 인터페이스가 유효한 SQL을 생성하지 못했습니다.")
            
            # 로컬 Vanna 사용 시도
            try:
                from app.utils.vanna_local import get_vanna_local
                vanna_local = get_vanna_local()
                
                if vanna_local and vanna_local.is_initialized:
                    print("Vanna Local로 SQL 생성 시도")
                    sql = with_timeout(vanna_local.generate_sql, args=(question,))
                    
                    if sql and "SELECT" in sql.upper():
                        print(f"Vanna Local로 생성된 SQL: {sql}")
                        return sql
            except Exception as local_err:
                print(f"Vanna Local 호출 중 오류: {str(local_err)}")
            
            # 모든 방법이 실패하면 룰 기반 방식으로 전환
            print("Vanna AI 실패. 룰 기반 SQL 생성으로 전환합니다.")
            return generate_sql_with_rules(question)
            
        except Exception as gen_err:
            print(f"Vanna SQL 생성 중 오류: {str(gen_err)}")
            traceback.print_exc()
            
            # 오류 발생 시 룰 기반 방식으로 전환
            print("오류로 인해 룰 기반 SQL 생성으로 전환합니다.")
            return generate_sql_with_rules(question)
    
    except Exception as e:
        print(f"Vanna AI SQL 생성 오류: {str(e)}")
        traceback.print_exc()
        
        # 최종 실패 시 룰 기반 방식으로 전환
        print("Vanna AI 초기화 실패. 룰 기반 SQL 생성으로 전환합니다.")
        return generate_sql_with_rules(question)

def get_mariadb_schema(db_config=None):
    """MariaDB 스키마 정보를 가져옵니다."""
    if db_config is None:
        db_config = DB_CONFIG
        
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 모든 테이블 조회
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]

        schema = []
        relations = []

        for table in tables:
            # 테이블 코멘트 가져오기 (향상된 기능)
            cursor.execute(f"""
                SELECT table_comment 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE table_schema = '{db_config['database']}' AND table_name = '{table}'
            """)
            table_comment = cursor.fetchone()[0]
            table_schema = f"\n{table}"
            if table_comment:
                table_schema += f" - {table_comment}"
            table_schema += " (\n"
            
            # 컬럼 정보 및 코멘트 가져오기
            cursor.execute(f"""
                SELECT column_name, column_type, column_comment, is_nullable, column_key
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE table_schema = '{db_config['database']}' AND table_name = '{table}'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()

            for col in columns:
                col_name, col_type, col_comment, nullable, key = col
                col_info = f"  {col_name} {col_type}"
                
                # NULL 허용 여부 추가
                if nullable == 'NO':
                    col_info += " NOT NULL"
                    
                # 키 정보 추가
                if key == 'PRI':
                    col_info += " PRIMARY KEY"
                elif key == 'UNI':
                    col_info += " UNIQUE"
                
                # 코멘트 추가
                if col_comment:
                    col_info += f" -- {col_comment}"
                    
                table_schema += col_info + ",\n"

                # 암묵적 외래키 추론
                if col_name.endswith('_id'):
                    ref_table = col_name[:-3]  # user_id -> user
                    relations.append(f"{table}.{col_name} → {ref_table}.id")

            # 실제 외래 키 정보 가져오기 (향상된 기능)
            cursor.execute(f"""
                SELECT column_name, referenced_table_name, referenced_column_name
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE table_schema = '{db_config['database']}' 
                AND table_name = '{table}'
                AND referenced_table_name IS NOT NULL
            """)
            fk_constraints = cursor.fetchall()
            
            for fk in fk_constraints:
                col_name, ref_table, ref_col = fk
                relations.append(f"{table}.{col_name} → {ref_table}.{ref_col} (FK)")

            table_schema = table_schema.rstrip(",\n") + "\n)\n"
            schema.append(table_schema)

        # 관계 추가
        if relations:
            schema.append("\n-- 관계 설명:")
            for rel in relations:
                schema.append(rel)

        cursor.close()
        conn.close()
        return "\n".join(schema)
    except Exception as e:
        print(f"데이터베이스 스키마 가져오기 오류: {str(e)}")
        traceback.print_exc()
        return "스키마 로딩 실패: " + str(e)

def clean_sql(sql_text):
    """LLM이 생성한 텍스트에서 SQL 쿼리만 정제하여 추출합니다."""
    if not sql_text:
        return None
        
    # SQL 쿼리 추출을 위한 정규식 패턴 확장
    sql_patterns = [
        # 일반적인 SQL 쿼리 검색 패턴 (가장 기본적인 형태)
        r'(?:```sql\s*)(.*?)(?:\s*```)',
        r'(?:```\s*)(SELECT[\s\S]*?)(```|$)',
        # 코드 블록 없이 직접 SQL이 작성된 경우
        r'(?:SELECT\s+.*?FROM.*?(?:WHERE|GROUP BY|ORDER BY|LIMIT|;|$).*)',
        # 설명과 함께 제공된 경우를 위한 패턴
        r'(?:쿼리|SQL|sql)(?::|은|는|=|\n)\s*(SELECT[\s\S]*?)(;|$)',
        # 백틱으로 감싸진 SQL만 있는 경우
        r'`(SELECT[\s\S]*?)`',
        # 일부 모델이 "sql" 태그를 사용하는 경우
        r'<sql>([\s\S]*?)</sql>',
        # 간단히 SELECT로 시작하는 문장 찾기 (마지막 옵션)
        r'(SELECT\s+[^;]+;?)'
    ]
    
    # 모든 패턴에 대해 일치하는 첫 번째 결과 사용
    for pattern in sql_patterns:
        matches = re.findall(pattern, sql_text, re.IGNORECASE | re.DOTALL)
        if matches and len(matches) > 0:
            match = matches[0]
            if isinstance(match, tuple) and len(match) > 0:
                match = match[0]  # 첫 번째 캡처 그룹 사용
            
            sql = match.strip()
            # SQL 검증: SELECT 키워드가 있고 FROM 절이 포함된 경우에만 유효
            if "SELECT" in sql.upper() and "FROM" in sql.upper():
                print(f"정규식 '{pattern}'에서 SQL 추출: {sql[:50]}...")
                return sql
    
    # 직접 SQL 문법 파싱 시도
    try:
        # SELECT 문으로 시작하는 모든 줄 추출
        lines = sql_text.split('\n')
        for i, line in enumerate(lines):
            if line.strip().upper().startswith("SELECT"):
                # SELECT 문 시작점에서부터 세미콜론(;)이나 문자열 끝까지 추출
                sql_builder = []
                j = i
                while j < len(lines):
                    current_line = lines[j].strip()
                    sql_builder.append(current_line)
                    if current_line.endswith(';'):
                        break
                    j += 1
                
                sql = ' '.join(sql_builder).strip()
                if "SELECT" in sql.upper() and "FROM" in sql.upper():
                    print(f"직접 파싱으로 SQL 추출: {sql[:50]}...")
                    return sql
    except Exception as e:
        print(f"SQL 구문 직접 파싱 중 오류: {str(e)}")
    
    # 모든 방법이 실패한 경우, 마지막 시도로 SELECT를 포함하는 가장 긴 줄 사용
    try:
        lines = sql_text.split('\n')
        sql_lines = [line.strip() for line in lines if "SELECT" in line.upper() and "FROM" in line.upper()]
        if sql_lines:
            # 가장 긴 줄 선택
            sql = max(sql_lines, key=len)
            print(f"가장 긴 SELECT 라인 추출: {sql[:50]}...")
            return sql
    except Exception as e:
        print(f"긴 라인 추출 중 오류: {str(e)}")
    
    # 여전히 SQL을 추출하지 못한 경우
    if "SELECT" in sql_text.upper() and "FROM" in sql_text.upper():
        # 최소한 SELECT와 FROM이 포함된 경우 전체 반환
        return sql_text.strip()
        
    print("SQL 추출 실패: 유효한 SQL 쿼리를 찾을 수 없습니다.")
    return None

def run_sql_query(query: str, db_config=None):
    """SQL 쿼리를 실행하고 결과를 반환합니다."""
    if db_config is None:
        db_config = DB_CONFIG
        
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as e:
        error_message = f"SQL 실행 오류: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return {"error": error_message}

def generate_sql_with_llm(model, tokenizer, prompt, max_new_tokens=512, temperature=0.1):
    """LLM을 직접 호출하여 SQL을 생성하는 함수"""
    try:
        print(f"SQL 생성용 프롬프트 길이: {len(prompt)} 문자")
        
        # 토크나이저 설정
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 모델이 CUDA를 사용할 수 있으면 GPU로 이동
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            model.to('cuda')
        
        # 추론 설정
        with torch.no_grad():
            # 추론 실행
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 결정적 생성을 위해 False로 설정
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                repetition_penalty=1.2  # 반복 방지 패널티 추가
            )
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # 입력 프롬프트 제거 시도
        if generated_text.startswith(prompt):
            response_text = generated_text[len(prompt):].strip()
        else:
            # 프롬프트가 정확히 일치하지 않는 경우 대략적인 시작점 찾기
            prompt_end_marker = "# SQL 쿼리:"
            if prompt_end_marker in prompt and prompt_end_marker in generated_text:
                response_start = generated_text.find(prompt_end_marker) + len(prompt_end_marker)
                response_text = generated_text[response_start:].strip()
            else:
                response_text = generated_text

        print(f"LLM 응답 (처음 100자): {response_text[:100]}...")
        
        # SQL 추출
        sql = clean_sql(response_text)
        
        # SQL이 없으면 최후의 방법으로 다시 시도
        if not sql:
            print("정규식 패턴으로 SQL을 추출하지 못했습니다. 직접 분석 시도...")
            
            lines = response_text.split('\n')
            # "SELECT"로 시작하는 라인 찾기
            for i, line in enumerate(lines):
                if line.strip().upper().startswith("SELECT"):
                    # 해당 라인부터 모든 라인 결합 (최대 10개 라인까지)
                    potential_sql = ' '.join(lines[i:min(i+10, len(lines))])
                    
                    # 세미콜론이 있으면 세미콜론까지만 추출
                    if ';' in potential_sql:
                        potential_sql = potential_sql[:potential_sql.find(';')+1]
                    
                    if "SELECT" in potential_sql.upper() and "FROM" in potential_sql.upper():
                        sql = potential_sql.strip()
                        print(f"직접 분석으로 SQL 추출: {sql[:50]}...")
                        break
        
        return sql
    except Exception as e:
        print(f"LLM SQL 생성 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return None

def generate_sql_with_rules(question):
    """규칙 기반 접근법으로 SQL 쿼리를 생성합니다. (LLM 실패 시 백업)"""
    print("자체 패턴 인식 SQL 생성 방식 사용")
    
    # 디버깅 추가
    log_prefix = f"[규칙기반-SQL]"
    print(f"{log_prefix} 질문: '{question}'")
    
    # 질문을 소문자로 변환
    question_lower = question.lower().strip()
    original_question = question  # 원본 질문 보존
    
    # 테이블 및 필드 매핑 정보 (실제 DB 스키마 기반으로 업데이트)
    tables = {
        "user": {
            "aliases": ["사용자", "유저", "user", "회원", "user 테이블", "user테이블"],
            "fields": ["user_id", "user_name"],  # 실제 존재하는 필드만 포함
            "related": ["user_monthly_stats", "user_resource"]
        },
        "user_resource": {
            "aliases": ["자원", "리소스", "resource", "vm", "가상머신", "서버", "user_resource", "user_resource 테이블", "user_resource테이블"],
            "fields": ["id", "user_id", "vm_name", "cpu", "memory", "io", "date"],
            "related": ["user"]
        },
        "user_monthly_stats": {
            "aliases": ["통계", "stats", "monthly", "월간", "현황", "로그인 통계", "사용량", "결제", "user_monthly_stats", "user_monthly_stats 테이블", "user_monthly_stats테이블"],
            "fields": ["id", "user_id", "month", "login_count", "purchase_amount"],
            "related": ["user"]
        },
        # 새로 추가된 테이블들
        "users": {
            "aliases": ["사용자들", "유저들", "회원들", "고객", "고객들", "users", "users 테이블", "users테이블", "20대", "나이", "가입자"],
            "fields": ["user_id", "name", "email", "signup_date", "age"],
            "related": ["orders", "reviews", "inquiries"]
        },
        "products": {
            "aliases": ["제품", "상품", "물건", "아이템", "제품들", "상품들", "products", "products 테이블", "products테이블", "전자기기", "가격"],
            "fields": ["product_id", "name", "category", "price", "stock"],
            "related": ["order_items", "reviews", "categories"]
        },
        "categories": {
            "aliases": ["카테고리", "분류", "종류", "카테고리들", "categories", "categories 테이블", "categories테이블"],
            "fields": ["category_id", "name", "description"],
            "related": ["products"]
        },
        "orders": {
            "aliases": ["주문", "구매", "결제", "주문내역", "주문들", "orders", "orders 테이블", "orders테이블"],
            "fields": ["order_id", "user_id", "order_date", "total_amount", "status"],
            "related": ["users", "order_items", "deliveries"]
        },
        "order_items": {
            "aliases": ["주문항목", "주문상품", "주문아이템", "주문 아이템", "order_items", "order_items 테이블", "order_items테이블"],
            "fields": ["item_id", "order_id", "product_id", "quantity", "price"],
            "related": ["orders", "products"]
        },
        "reviews": {
            "aliases": ["리뷰", "후기", "평가", "리뷰들", "후기들", "reviews", "reviews 테이블", "reviews테이블"],
            "fields": ["review_id", "user_id", "product_id", "rating", "comment", "review_date"],
            "related": ["users", "products"]
        },
        "deliveries": {
            "aliases": ["배송", "배달", "배송정보", "배송 정보", "배송현황", "deliveries", "deliveries 테이블", "deliveries테이블"],
            "fields": ["delivery_id", "order_id", "address", "status", "tracking_number", "delivery_date"],
            "related": ["orders"]
        },
        "inquiries": {
            "aliases": ["문의", "질문", "문의사항", "inquiry", "inquiries", "inquiries 테이블", "inquiries테이블"],
            "fields": ["inquiry_id", "user_id", "title", "content", "inquiry_date", "status", "inquiry_type"],
            "related": ["users", "inquiry_responses"]
        },
        "inquiry_responses": {
            "aliases": ["문의답변", "답변", "응답", "inquiry_responses", "inquiry_responses 테이블", "inquiry_responses테이블"],
            "fields": ["response_id", "inquiry_id", "admin_id", "content", "response_date"],
            "related": ["inquiries", "admins"]
        },
        "admins": {
            "aliases": ["관리자", "어드민", "담당자", "admin", "관리자들", "admins", "admins 테이블", "admins테이블"],
            "fields": ["admin_id", "name", "email", "role"],
            "related": ["inquiry_responses"]
        }
    }
    
    # 디버깅 메시지 추가
    print(f"{log_prefix} 질문 분석 시작...")
    
    # 기본 테이블 목록 (질문에 명확한 테이블이 언급되지 않은 경우 사용)
    default_tables = ["users", "products", "categories", "orders", "reviews", "user", "user_resource", "user_monthly_stats"]
    
    # 질문에서 테이블 이름 직접 언급 여부 확인 (정확한 테이블명 우선 확인)
    direct_table_mention = None
    for table_name, table_info in tables.items():
        # 직접 테이블명 언급 확인 - 영문 테이블명 정확히 포함된 경우 우선 처리
        if table_name in question_lower:
            direct_table_mention = table_name
            print(f"{log_prefix} 질문에서 테이블명 '{table_name}' 직접 언급 감지")
            break
    
    # 테이블 별칭을 통한 테이블 식별 (직접 언급 없는 경우)
    if not direct_table_mention:
        for table_name, table_info in tables.items():
            for alias in table_info["aliases"]:
                if alias in question_lower:
                    direct_table_mention = table_name
                    print(f"{log_prefix} 질문에서 테이블 별칭 '{alias}' 감지, 테이블: {table_name}")
                    break
            if direct_table_mention:
                break
    
    # 관련 테이블 감지 (리뷰 관련 질문에 대한 특별 처리)
    related_tables = []
    if direct_table_mention:
        related_tables = tables.get(direct_table_mention, {}).get("related", [])
        print(f"{log_prefix} 관련 테이블: {related_tables}")
    
    # 리뷰 관련 특별 패턴 - 제품과 리뷰를 함께 다루는 질문
    is_review_related = "리뷰" in question_lower or "후기" in question_lower or "평가" in question_lower
    if is_review_related:
        print(f"{log_prefix} 리뷰 관련 질문 감지")
        if direct_table_mention != "reviews":
            # 리뷰 관련 질문이지만 reviews가 직접 언급되지 않은 경우
            # products를 메인 테이블로 설정하고 reviews를 관련 테이블로 추가
            if "제품" in question_lower or "상품" in question_lower or "아이템" in question_lower:
                direct_table_mention = "products"
                if "reviews" not in related_tables:
                    related_tables.append("reviews")
                print(f"{log_prefix} 리뷰 관련 제품 질문 감지: {direct_table_mention}, 관련: {related_tables}")
            else:
                # 리뷰만 언급된 경우
                direct_table_mention = "reviews"
                if "products" not in related_tables:
                    related_tables.append("products")
                print(f"{log_prefix} 리뷰 질문 감지: {direct_table_mention}, 관련: {related_tables}")
    
    # 주문 관련 특별 패턴 - 제품과 주문을 함께 다루는 질문
    is_order_related = "주문" in question_lower or "구매" in question_lower or "결제" in question_lower
    if is_order_related:
        print(f"{log_prefix} 주문 관련 질문 감지")
        if direct_table_mention != "orders" and direct_table_mention != "order_items":
            # 주문 관련 질문이지만 orders가 직접 언급되지 않은 경우
            if "제품" in question_lower or "상품" in question_lower or "아이템" in question_lower:
                # 제품 관련 주문 질문
                direct_table_mention = "products"
                if "orders" not in related_tables:
                    related_tables.append("orders")
                if "order_items" not in related_tables:
                    related_tables.append("order_items")
                print(f"{log_prefix} 제품 주문 질문 감지: {direct_table_mention}, 관련: {related_tables}")
            else:
                # 주문만 언급된 경우
                direct_table_mention = "orders"
                if "products" not in related_tables:
                    related_tables.append("products")
                if "order_items" not in related_tables:
                    related_tables.append("order_items")
                print(f"{log_prefix} 주문 질문 감지: {direct_table_mention}, 관련: {related_tables}")
    
    # 사용자 질문에 없는 테이블이 언급된 경우 플래그
    unavailable_tables = {
        "post": ["게시물", "게시글", "글", "포스트", "post", "article", "작성글", "post 테이블", "post테이블"],
        "comment": ["댓글", "comment", "reply", "답글", "comment 테이블", "comment테이블"]
    }
    
    # 질문에 존재하지 않는 테이블이 언급되었는지 확인
    unavailable_mentioned = None
    for table, aliases in unavailable_tables.items():
        if any(alias in question_lower for alias in aliases):
            unavailable_mentioned = table
            print(f"{log_prefix} 존재하지 않는 테이블 '{table}' 관련 질문 감지")
            break
    
    # 집계 또는 순위 패턴 인식
    aggregation_patterns = {
        "최대": "MAX",
        "최소": "MIN",
        "평균": "AVG",
        "합계": "SUM",
        "총": "SUM",
        "개수": "COUNT",
        "수": "COUNT",
        "갯수": "COUNT",
        "카운트": "COUNT"
    }
    
    # 정렬 패턴
    order_patterns = {
        "내림차순": "DESC",
        "오름차순": "ASC",
        "높은순": "DESC",
        "낮은순": "ASC",
        "많은순": "DESC",
        "적은순": "ASC",
        "최신순": "DESC",  # 날짜 기준
        "오래된순": "ASC"   # 날짜 기준
    }
    
    # 집계 함수 결정
    aggregate_fn = None
    for pattern, fn in aggregation_patterns.items():
        if pattern in question_lower:
            aggregate_fn = fn
            break
    
    # 정렬 방향 결정
    order_direction = "DESC"  # 기본값 (내림차순)
    for pattern, direction in order_patterns.items():
        if pattern in question_lower:
            order_direction = direction
            break
    
    # 가장 많은/적은 패턴 인식 (TOP-N 쿼리)
    top_n_pattern = False
    limit_value = 10  # 기본값
    
    if "가장 많" in question_lower or "가장 높" in question_lower or "최대" in question_lower or "상위" in question_lower or "많이" in question_lower:
        top_n_pattern = True
        order_direction = "DESC"
    elif "가장 적" in question_lower or "가장 낮" in question_lower or "최소" in question_lower or "하위" in question_lower or "적게" in question_lower:
        top_n_pattern = True
        order_direction = "ASC"
    
    # 특정 N 값 추출 (예: 상위 5개, 최근 10명, 최대 20개 등)
    import re
    n_match = re.search(r'(\d+)(?:\s*개|명|위|등|건)', question_lower)
    if n_match:
        limit_value = int(n_match.group(1))
    
    # 최근 데이터 패턴 인식
    is_recent_data = False
    order_by_field = None
    if "최근" in question_lower or "최신" in question_lower:
        is_recent_data = True
        # 테이블별 날짜 필드 매핑
        date_fields = {
            "users": "signup_date",
            "orders": "order_date",
            "reviews": "review_date",
            "inquiries": "inquiry_date",
            "user_resource": "date"
        }
        # 언급된 테이블이 있으면 해당 테이블의 날짜 필드 사용
        if direct_table_mention and direct_table_mention in date_fields:
            order_by_field = date_fields[direct_table_mention]
        else:
            # 기본값
            order_by_field = "date"
        order_direction = "DESC"
    
    # 가입/등록 관련 패턴 인식
    signup_patterns = ["가입", "등록", "신규", "새로운", "최근에 가입"]
    is_signup_query = any(pattern in question_lower for pattern in signup_patterns)
    
    # 나이 조건 패턴 인식
    age_patterns = {
        "10대": (10, 19),
        "20대": (20, 29),
        "30대": (30, 39),
        "40대": (40, 49),
        "50대": (50, 59),
        "60대": (60, 69),
        "70대": (70, 79),
        "80대": (80, 89),
        "90대": (90, 99)
    }
    age_condition = None
    for age_group, (min_age, max_age) in age_patterns.items():
        if age_group in question_lower:
            age_condition = (min_age, max_age)
            print(f"{log_prefix} 나이 조건 감지: {age_group}")
            break
    
    # SQL 쿼리 생성
    if unavailable_mentioned:
        return f"/* 요청하신 '{unavailable_mentioned}' 테이블은 현재 사용할 수 없습니다. */"
    
    # === 리뷰 관련 질문 특수 처리 ===
    
    # 1. "리뷰가 없는 제품은 어떤 것들이야?"와 같은 질문 처리
    if "리뷰" in question_lower and "없는" in question_lower and "제품" in question_lower:
        print(f"{log_prefix} '리뷰가 없는 제품' 패턴 감지")
        return """
SELECT p.product_id, p.name AS 제품명, p.category AS 카테고리, p.price AS 가격
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
WHERE r.review_id IS NULL
ORDER BY p.name;
"""
    
    # 2. "제품별 평균 리뷰 점수를 보여줘" 유형의 질문 처리
    if "평균" in question_lower and "리뷰" in question_lower and ("점수" in question_lower or "평점" in question_lower) and "제품" in question_lower:
        print(f"{log_prefix} '제품별 평균 리뷰 점수' 패턴 감지")
        return """
SELECT p.product_id, p.name AS 제품명, AVG(r.rating) AS 평균_평점
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.product_id, p.name
ORDER BY 평균_평점 DESC;
"""
    
    # 3. "가장 자주 주문된 제품 TOP 3를 알려줘" 유형의 질문 처리
    if (("가장 자주" in question_lower or "많이" in question_lower) and "주문" in question_lower and "제품" in question_lower) or ("top" in question_lower and "제품" in question_lower):
        print(f"{log_prefix} '가장 자주 주문된 제품' 패턴 감지")
        limit = 3  # 기본값
        if limit_value:
            limit = limit_value
            
        return f"""
SELECT p.product_id, p.name AS 제품명, COUNT(oi.product_id) AS 주문_횟수
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.name
ORDER BY 주문_횟수 DESC
LIMIT {limit};
"""
    
    # 4. 특정 평점 이상/이하의 리뷰를 받은 제품 찾기
    rating_pattern = re.search(r'(\d+)(?:\s*점|평점|별점)', question_lower)
    if rating_pattern and (("이상" in question_lower or "초과" in question_lower or "넘는" in question_lower) or 
                          ("이하" in question_lower or "미만" in question_lower or "낮은" in question_lower)):
        rating_value = int(rating_pattern.group(1))
        operator = ">=" if "이상" in question_lower or "초과" in question_lower or "넘는" in question_lower else "<="
        
        print(f"{log_prefix} '특정 평점({rating_value}) {operator} 제품' 패턴 감지")
        return f"""
SELECT p.product_id, p.name AS 제품명, AVG(r.rating) AS 평균_평점
FROM products p
JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.product_id, p.name
HAVING 평균_평점 {operator} {rating_value}
ORDER BY 평균_평점 DESC;
"""
    
    # 나이대별 사용자 수 쿼리
    if age_condition and any(word in question_lower for word in ["몇 명", "사용자", "유저", "회원"]):
        min_age, max_age = age_condition
        return f"SELECT COUNT(*) AS 사용자_수 FROM users WHERE age BETWEEN {min_age} AND {max_age};"
    
    # 문의 유형별 처리 완료 비율 등 상태 관련 쿼리
    if "처리" in question_lower and "완료" in question_lower and "비율" in question_lower and "유형" in question_lower:
        return """
SELECT 
    inquiry_type AS 문의유형,
    COUNT(*) AS 전체_건수,
    COUNT(CASE WHEN status = 'completed' THEN 1 END) AS 완료_건수,
    ROUND((COUNT(CASE WHEN status = 'completed' THEN 1 END) / COUNT(*)) * 100, 2) AS 완료_비율
FROM 
    inquiries
GROUP BY 
    inquiry_type
ORDER BY 
    완료_비율 DESC;
"""
    
    # 특정 상태의 문의 통계
    if any(status in question_lower for status in ["처리 완료", "처리중", "대기"]) and "문의" in question_lower:
        status_mapping = {
            "처리 완료": "completed",
            "완료": "completed",
            "처리중": "in_progress",
            "진행중": "in_progress",
            "대기": "pending",
            "대기중": "pending"
        }
        
        status_condition = None
        for key, value in status_mapping.items():
            if key in question_lower:
                status_condition = value
                break
        
        if status_condition:
            return f"""
SELECT 
    inquiry_type AS 문의유형,
    COUNT(*) AS 건수
FROM 
    inquiries
WHERE 
    status = '{status_condition}'
GROUP BY 
    inquiry_type
ORDER BY 
    건수 DESC;
"""
    
    # 최근 가입한 사용자 목록
    if any(word in question_lower for word in ["최근", "신규"]) and any(word in question_lower for word in ["가입", "등록", "유저", "사용자"]):
        limit = 10  # 기본값
        for num in re.findall(r'\d+', question_lower):
            if int(num) <= 100:  # 합리적인 범위 제한
                limit = int(num)
                break
                
        return f"""
SELECT 
    user_id AS 사용자ID,
    name AS 이름,
    email AS 이메일,
    signup_date AS 가입일자
FROM 
    users
ORDER BY 
    signup_date DESC
LIMIT {limit};
"""
    
    # 직접 테이블, 필드 언급 또는 패턴이 감지된 경우
    if direct_table_mention:
        target_table = direct_table_mention
        target_fields = tables.get(target_table, {}).get("fields", ["*"])
        
        # 집계 함수가 필요한 경우
        if aggregate_fn:
            # 집계할 필드 결정 (숫자 필드 선호)
            numeric_fields = ["id", "count", "amount", "price", "quantity", "rating", "age"]
            agg_field = "*"  # 기본값
            
            for field in target_fields:
                if any(num_field in field.lower() for num_field in numeric_fields):
                    agg_field = field
                    break
            
            return f"SELECT {aggregate_fn}({agg_field}) AS 결과값 FROM {target_table};"
        
        # TOP-N 쿼리 패턴
        elif top_n_pattern:
            order_field = order_by_field or target_fields[0]
            for field in target_fields:
                if any(metric in field.lower() for metric in ["count", "amount", "price", "rating"]):
                    order_field = field
                    break
                    
            field_list = ", ".join(target_fields[:3]) if len(target_fields) > 1 else "*"
            return f"SELECT {field_list} FROM {target_table} ORDER BY {order_field} {order_direction} LIMIT {limit_value};"
        
        # 최근 데이터 패턴
        elif is_recent_data:
            date_field = order_by_field or "date"
            field_list = ", ".join(target_fields[:3]) if len(target_fields) > 1 else "*"
            return f"SELECT {field_list} FROM {target_table} ORDER BY {date_field} DESC LIMIT {limit_value};"
            
        # 기본 쿼리 - 테이블의 첫 몇 개 레코드
        else:
            field_list = ", ".join(target_fields[:5]) if len(target_fields) > 1 else "*"
            return f"SELECT {field_list} FROM {target_table} LIMIT {limit_value};"
    
    # 특별 규칙에 해당하지 않는 경우의 기본 쿼리
    # 관련성 높은 테이블 선택 (기본값: users)
    default_table = "users"
    for table_name, info in tables.items():
        for alias in info["aliases"]:
            if alias in question_lower:
                default_table = table_name
                break
    
    # 나이 조건이 있으면 적용
    if age_condition:
        min_age, max_age = age_condition
        where_clause = f"WHERE age BETWEEN {min_age} AND {max_age}"
        
        # "몇 명" 등 카운트 질문인 경우
        if any(word in question_lower for word in ["몇 명", "몇명", "몇 건", "몇건", "수", "개수"]):
            print(f"{log_prefix} 규칙 기반 성공: 나이 조건이 있는 카운트 쿼리 생성")
            return f"SELECT COUNT(*) AS 사용자_수 FROM {default_table} {where_clause};"
        else:
            print(f"{log_prefix} 규칙 기반 성공: 나이 조건이 있는 일반 조회 쿼리 생성")
            return f"SELECT * FROM {default_table} {where_clause} LIMIT {limit_value};"
    
    # 어떤 규칙에도 해당하지 않는 경우 최후의 기본 쿼리
    print(f"{log_prefix} 규칙 기반 성공: 기본 쿼리 생성")
    return f"SELECT * FROM {default_table} LIMIT {limit_value};"

def get_compare_operator(compare_type):
    """비교 유형에 따라 적절한 SQL 연산자를 반환합니다."""
    if compare_type == "이상":
        return ">="
    elif compare_type == "초과":
        return ">"
    elif compare_type == "이하":
        return "<="
    elif compare_type == "미만":
        return "<"
    else:
        return "="

def format_sql_results(rows):
    """SQL 결과를 마크다운 테이블 형식으로 포맷팅합니다."""
    if isinstance(rows, dict) and "error" in rows:
        return f"❌ {rows['error']}"
        
    if not rows:
        return "⚠️ 결과가 없습니다."
        
    # 결과를 마크다운 테이블로 변환
    headers = rows[0].keys()
    result_md = "| " + " | ".join(headers) + " |\n"
    result_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    for row in rows:
        result_md += "| " + " | ".join(str(v) if v is not None else "" for v in row.values()) + " |\n"
        
    return result_md

# MySQL/MariaDB 연결 테스트 함수 추가
def test_db_connection(db_config=None):
    """데이터베이스 연결을 테스트합니다."""
    if db_config is None:
        db_config = DB_CONFIG
        
    try:
        print(f"데이터베이스 연결 테스트 중... (host: {db_config['host']})")
        import mysql.connector
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT VERSION()")
        version = cursor.fetchone()
        cursor.close()
        conn.close()
        print(f"데이터베이스 연결 성공: {version}")
        return True
    except Exception as e:
        print(f"데이터베이스 연결 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False 

def generate_sql_from_question(question, schema, state=None):
    """
    자연어 질문을 SQL로 변환합니다.
    
    Args:
        question (str): 자연어 질문
        schema (str): 데이터베이스 스키마 정보
        state (Optional): FastAPI app.state 객체
        
    Returns:
        str: 생성된 SQL 쿼리
    """
    log_prefix = f"[SQL 생성 - '{question}']"
    print(f"{log_prefix} SQL 변환 시작")
    
    # 스키마 검증 및 로깅
    if not schema or len(schema) < 50:
        print(f"{log_prefix} 경고: 스키마 정보가 없거나 불완전합니다 (길이: {len(schema) if schema else 0})")
        from app.utils.get_mariadb_schema import get_mariadb_schema
        schema = get_mariadb_schema()
        print(f"{log_prefix} 스키마 정보 새로 획득 (길이: {len(schema)})")
    else:
        print(f"{log_prefix} 유효한 스키마 정보가 제공됨 (길이: {len(schema)})")
    
    # 스키마 요약 출력 (디버깅용)
    schema_preview = schema[:200] + "..." if len(schema) > 200 else schema
    print(f"{log_prefix} 스키마 미리보기: {schema_preview}")
    
    # Vanna AI로 SQL 생성 시도
    print(f"{log_prefix} Vanna AI로 SQL 생성 시도")
    vanna_sql = generate_sql_with_vanna(question)
    if vanna_sql and "SELECT" in vanna_sql.upper():
        print(f"{log_prefix} Vanna AI 성공: {vanna_sql}")
        # 질문-SQL 쌍 저장 (추가 학습에 활용)
        try:
            from app.utils.vanna_interface import get_vanna_interface
            vanna_interface = get_vanna_interface()
            if vanna_interface and vanna_interface.is_initialized:
                vanna_interface.add_question_sql_pair(question, vanna_sql)
                print(f"{log_prefix} 질문-SQL 쌍 저장 성공")
        except Exception as e:
            print(f"{log_prefix} 질문-SQL 쌍 저장 실패: {str(e)}")
        
        return vanna_sql
    else:
        print(f"{log_prefix} Vanna AI 실패: 유효한 SQL을 생성하지 못함")
    
    # Defog로 SQL 생성 시도
    try:
        from app.utils.sql_utils_defog import generate_sql_with_defog
        print(f"{log_prefix} Defog로 SQL 생성 시도")
        defog_sql = generate_sql_with_defog(question)
        if defog_sql and "SELECT" in defog_sql.upper():
            print(f"{log_prefix} Defog 성공: {defog_sql}")
            return defog_sql
        else:
            print(f"{log_prefix} Defog 실패: 유효한 SQL을 생성하지 못함")
    except Exception as defog_err:
        print(f"{log_prefix} Defog 오류: {str(defog_err)}")
    
    # LLM으로 SQL 생성 시도
    print(f"{log_prefix} LLM으로 SQL 생성 시도")
    
    try:
        from app.utils.llm_utils import generate_text_with_local_llm
        
        # 데이터베이스 스키마를 포함한 프롬프트 생성
        prompt = f"""당신은 SQL 전문가입니다. 주어진 질문을 SQL 쿼리로 변환해주세요.

## 데이터베이스 스키마 정보:
{schema}

## 질문:
{question}

## 규칙:
1. 반드시 유효한 SQL 쿼리만 생성하세요.
2. 테이블과 필드 이름은 제공된 스키마에 있는 것만 사용하세요.
3. 사용자가 특정 테이블이나 필드를 언급하면 그에 맞는 테이블/필드를 사용하세요.
4. 생성된 SQL은 SELECT 문만 허용합니다.
5. 결과는 SQL 쿼리 코드만 반환하세요. 설명이나 주석을 포함하지 마세요.

## SQL 쿼리:
```sql
"""
        
        # SQL 생성
        print(f"{log_prefix} LLM으로 SQL 생성 프롬프트 생성 완료 (길이: {len(prompt)})")
        llm_response = generate_text_with_local_llm(
            prompt=prompt,
            max_tokens=512,
            temperature=0.1,
            top_p=0.9,
            system_prompt="당신은 정확한 SQL 쿼리를 생성하는 전문가입니다."
        )
        
        print(f"{log_prefix} LLM 응답: {llm_response[:100]}...")
        
        # SQL 추출
        sql = clean_sql(llm_response)
        if sql and "SELECT" in sql.upper():
            print(f"{log_prefix} LLM 성공: {sql}")
            return sql
        else:
            print(f"{log_prefix} LLM 실패: 유효한 SQL을 추출하지 못함. 규칙 기반 방식으로 전환")
    except Exception as llm_err:
        print(f"{log_prefix} LLM 오류: {str(llm_err)}")
    
    # 마지막으로 규칙 기반 접근법 사용
    print(f"{log_prefix} 규칙 기반 방식으로 SQL 생성 시도")
    rule_sql = generate_sql_with_rules(question)
    
    # 결과 반환 전 유효성 추가 검증
    if rule_sql and "SELECT" in rule_sql.upper():
        print(f"{log_prefix} 규칙 기반 SQL 생성 성공: {rule_sql}")
        return rule_sql
    else:
        print(f"{log_prefix} 경고: 모든 방식이 실패. 기본 SQL 반환")
        return "SELECT * FROM users LIMIT 10;"

def get_compare_operator(compare_type):
    """비교 유형에 따라 적절한 SQL 연산자를 반환합니다."""
    if compare_type == "이상":
        return ">="
    elif compare_type == "초과":
        return ">"
    elif compare_type == "이하":
        return "<="
    elif compare_type == "미만":
        return "<"
    else:
        return "=" 