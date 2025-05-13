import os
import traceback
import re
import mysql.connector
import json
from typing import Dict, List, Optional, Any, Union, Tuple

# MariaDB 연결 설정 (기존 설정과 동일)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "3Ssoft1!",
    "database": "chatbot"
}

# Defog API 키 (환경 변수에서 가져옴)
DEFOG_API_KEY = os.environ.get("DEFOG_API_KEY", "")

# Defog 시스템 프롬프트
DEFOG_SYSTEM_PROMPT = """
너는 MariaDB SQL 전문가입니다. 주어진 스키마 정보를 바탕으로 사용자의 자연어 질문을 SQL 쿼리로 변환해주세요.
한국어 질문도 정확하게 이해하고 적절한 SQL을 생성해야 합니다.

다음 규칙을 반드시 지켜주세요:
1. 실행 가능한 MariaDB SQL 구문만 반환하세요
2. 테이블과 컬럼 이름을 정확하게 사용하세요
3. 필요한 경우 JOIN, WHERE, GROUP BY, ORDER BY 등을 적절히 포함하세요
4. 조건절에서는 정확한 연산자를 사용하세요 (예: '90% 이상' -> '>= 90')
5. 결과 컬럼에는 의미 있는 별칭(alias)을 사용하세요
"""

# 테이블별 한글 매핑 (검색 성능 향상용)
TABLE_KR_MAPPING = {
    "users": ["사용자", "유저", "회원", "고객"],
    "products": ["제품", "상품", "물건", "아이템"],
    "orders": ["주문", "구매", "결제"],
    "reviews": ["리뷰", "후기", "평가"],
    "categories": ["카테고리", "분류", "종류"],
    "order_items": ["주문항목", "주문상품", "주문아이템"],
    "deliveries": ["배송", "배달", "배송정보"],
    "inquiries": ["문의", "질문", "문의사항"],
    "user": ["사용자", "유저", "회원"],
    "user_resource": ["자원", "리소스", "VM", "가상머신"],
    "user_monthly_stats": ["통계", "월간", "현황"]
}

# 테이블별 필드 매핑
TABLE_FIELDS = {
    "users": ["user_id", "name", "email", "signup_date", "age"],
    "products": ["product_id", "name", "category", "price", "stock"],
    "orders": ["order_id", "user_id", "order_date", "total_amount", "status"],
    "reviews": ["review_id", "user_id", "product_id", "rating", "comment", "review_date"],
    "user": ["user_id", "user_name"],
    "user_resource": ["id", "user_id", "vm_name", "cpu", "memory", "io", "date"],
    "user_monthly_stats": ["id", "user_id", "month", "login_count", "purchase_amount"]
}

# 주요 조인 관계 매핑
JOIN_RELATIONSHIPS = [
    ("products", "reviews", "product_id"),
    ("users", "reviews", "user_id"),
    ("users", "orders", "user_id"),
    ("orders", "order_items", "order_id"),
    ("products", "order_items", "product_id"),
    ("orders", "deliveries", "order_id"),
    ("users", "inquiries", "user_id"),
    ("user", "user_resource", "user_id"),
    ("user", "user_monthly_stats", "user_id")
]

def preprocess_korean_query(question: str) -> str:
    """
    한국어 질문을 전처리하여 SQL 생성에 유리한 형태로 변환합니다.
    """
    # 질문 정규화
    question = question.strip()
    
    # 연산자 변환 (한글 -> 기호)
    operator_mapping = {
        "이상": ">=",
        "초과": ">",
        "이하": "<=",
        "미만": "<",
        "같은": "=",
        "같지 않은": "!=",
        "다른": "!="
    }
    
    # 연산자 변환 패턴 적용
    for kr_op, en_op in operator_mapping.items():
        # 숫자 + 연산자 패턴 찾기 (예: "90% 이상" -> "90% >=")
        pattern = r'(\d+)(\s*%?\s*' + kr_op + r')'
        question = re.sub(pattern, r'\1 ' + en_op, question)
    
    # 테이블 이름 힌트 추가
    for table, kr_names in TABLE_KR_MAPPING.items():
        for kr_name in kr_names:
            if kr_name in question:
                # 질문에 테이블 힌트 추가 (괄호 안에)
                if f"({table})" not in question and f"({kr_name}: {table})" not in question:
                    question = question.replace(kr_name, f"{kr_name} ({table})")
                break
    
    return question

def get_mariadb_schema(db_config=None) -> str:
    """
    MariaDB 스키마 정보를 가져옵니다.
    """
    if db_config is None:
        db_config = DB_CONFIG
        
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 모든 테이블 조회
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]

        schema = []
        
        for table in tables:
            # 테이블 정의 시작
            schema.append(f"Table: {table}")
            
            # 컬럼 정보 가져오기
            cursor.execute(f"""
                SELECT column_name, column_type, is_nullable, column_key, column_comment
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE table_schema = '{db_config['database']}' AND table_name = '{table}'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            
            column_info = []
            for col in columns:
                col_name, col_type, nullable, key, comment = col
                col_desc = f"  - {col_name} ({col_type})"
                
                if key == 'PRI':
                    col_desc += ", Primary Key"
                elif key == 'UNI':
                    col_desc += ", Unique"
                
                if comment:
                    col_desc += f" // {comment}"
                    
                column_info.append(col_desc)
            
            schema.extend(column_info)
            
            # 외래 키 정보 가져오기
            cursor.execute(f"""
                SELECT column_name, referenced_table_name, referenced_column_name
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE table_schema = '{db_config['database']}' 
                AND table_name = '{table}'
                AND referenced_table_name IS NOT NULL
            """)
            fks = cursor.fetchall()
            
            if fks:
                schema.append("  Foreign Keys:")
                for fk in fks:
                    col, ref_table, ref_col = fk
                    schema.append(f"    - {col} -> {ref_table}.{ref_col}")
            
            schema.append("")  # 빈 줄 추가

        cursor.close()
        conn.close()
        return "\n".join(schema)
    except Exception as e:
        print(f"데이터베이스 스키마 가져오기 오류: {str(e)}")
        traceback.print_exc()
        return f"스키마 로딩 실패: {str(e)}"

def identify_relevant_tables(question: str) -> List[str]:
    """
    질문에서 관련 테이블을 식별합니다.
    """
    relevant_tables = []
    question_lower = question.lower()
    
    # 1. 직접 테이블 이름 매칭
    for table in TABLE_FIELDS.keys():
        if table.lower() in question_lower:
            relevant_tables.append(table)
    
    # 2. 한글 테이블 매핑 확인
    for table, kr_names in TABLE_KR_MAPPING.items():
        for kr_name in kr_names:
            if kr_name in question_lower and table not in relevant_tables:
                relevant_tables.append(table)
                break
    
    # 3. 특수 패턴 감지 (예: 리뷰 관련 질문이면 reviews와 products 테이블 추가)
    if any(keyword in question_lower for keyword in ["리뷰", "후기", "평가", "평점"]) and "reviews" not in relevant_tables:
        relevant_tables.append("reviews")
        if "products" not in relevant_tables:
            relevant_tables.append("products")
    
    if any(keyword in question_lower for keyword in ["주문", "구매", "결제"]) and "orders" not in relevant_tables:
        relevant_tables.append("orders")
        if "products" not in relevant_tables and "제품" in question_lower:
            relevant_tables.append("products")
        if "order_items" not in relevant_tables:
            relevant_tables.append("order_items")
    
    # 관련 테이블이 없으면 기본 테이블 사용
    if not relevant_tables:
        if "사용자" in question_lower or "유저" in question_lower or "회원" in question_lower:
            relevant_tables.append("users")
        elif "제품" in question_lower or "상품" in question_lower:
            relevant_tables.append("products")
        elif "나이" in question_lower or "연령" in question_lower:
            relevant_tables.append("users")
        else:
            # 기본 테이블
            relevant_tables = ["users", "products"]
    
    # 자동 조인 관계 추가
    tables_to_add = []
    for t1, t2, col in JOIN_RELATIONSHIPS:
        if t1 in relevant_tables and t2 not in relevant_tables and t2 not in tables_to_add:
            tables_to_add.append(t2)
        elif t2 in relevant_tables and t1 not in relevant_tables and t1 not in tables_to_add:
            tables_to_add.append(t1)
    
    relevant_tables.extend(tables_to_add)
    return relevant_tables

def filter_schema_for_tables(schema: str, relevant_tables: List[str]) -> str:
    """
    주어진 테이블 목록에 대한 스키마 정보만 필터링합니다.
    """
    filtered_lines = []
    include_table = False
    
    for line in schema.split("\n"):
        if line.startswith("Table: "):
            table_name = line.replace("Table: ", "").strip()
            include_table = table_name in relevant_tables
        
        if include_table or not line.strip():
            filtered_lines.append(line)
    
    return "\n".join(filtered_lines)

def run_sql_query(query: str, db_config=None) -> Union[List[Dict], Dict[str, str]]:
    """
    SQL 쿼리를 실행하고 결과를 반환합니다.
    """
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

def generate_sql_with_defog(question: str) -> str:
    """
    Defog를 사용하여 자연어 질문을 SQL로 변환합니다.
    API 키가 없을 경우 로컬 모델 기반으로 직접 생성합니다.
    """
    try:
        # 전처리된 질문 생성
        processed_question = preprocess_korean_query(question)
        print(f"[Defog] 전처리된 질문: {processed_question}")
        
        # 관련 테이블 식별
        relevant_tables = identify_relevant_tables(question)
        print(f"[Defog] 관련 테이블: {relevant_tables}")
        
        # 관련 테이블에 대한 스키마 필터링
        full_schema = get_mariadb_schema()
        filtered_schema = filter_schema_for_tables(full_schema, relevant_tables)
        print(f"[Defog] 필터링된 스키마 사용: {len(filtered_schema)} 바이트")
        
        # API 키 확인
        if DEFOG_API_KEY and len(DEFOG_API_KEY.strip()) > 0:
            # API 키가 있으면 Defog 클라우드 사용
            try:
                from defog import Defog
                # Defog 클라이언트 초기화
                defog_client = Defog(api_key=DEFOG_API_KEY)
                
                # SQL 쿼리 생성
                response = defog_client.generate_sql(
                    question=processed_question,
                    dialect="mysql",
                    schema=filtered_schema,
                    system_prompt=DEFOG_SYSTEM_PROMPT
                )
                
                # 결과 추출
                sql_query = response.sql_query
                print(f"[Defog] 생성된 SQL: {sql_query}")
                
                return sql_query
            except Exception as e:
                print(f"Defog 클라우드 API 호출 오류: {str(e)}")
                print("로컬 모델 기반 방식으로 전환합니다...")
        
        # API 키가 없거나 API 호출이 실패한 경우 로컬 모델 사용
        print("[Defog] 로컬 모델 기반 SQL 생성 방식 사용")
        
        # 로컬 LLM 모델 (Gemma2-9B-IT)을 사용하여 SQL 생성
        try:
            # FastAPI app.state에서 LLM 모델 가져오기
            # main.py에서 초기화된 모델을 직접 사용할 수 없으므로 우선 규칙 기반 접근법 사용
            from app.utils.sql_utils import generate_sql_with_rules
            
            # 규칙 기반 SQL 생성 함수 호출
            sql_query = generate_sql_with_rules(question)
            
            if "SELECT" in sql_query.upper():
                print(f"[Defog-Local] 규칙 기반 SQL 생성 성공: {sql_query}")
                return sql_query
            else:
                print("[Defog-Local] 규칙 기반 SQL 생성 실패, 기본 쿼리 반환")
                return f"SELECT * FROM {relevant_tables[0] if relevant_tables else 'users'} LIMIT 10;"
        except Exception as local_e:
            print(f"로컬 모델 기반 SQL 생성 오류: {str(local_e)}")
            
            # 최종 fallback: 기본 SQL 쿼리 반환
            default_table = relevant_tables[0] if relevant_tables else "users"
            return f"SELECT * FROM {default_table} LIMIT 10;"
    except ImportError:
        print("Defog 라이브러리가 설치되지 않았습니다. 'pip install defog' 명령으로 설치하세요.")
        return None
    except Exception as e:
        print(f"Defog SQL 생성 오류: {str(e)}")
        traceback.print_exc()
        return None

def generate_sql_with_rules(question: str) -> str:
    """
    규칙 기반 접근법으로 SQL 쿼리를 생성합니다. (Defog 실패 시 백업)
    """
    print("[규칙기반-SQL] 규칙 기반 SQL 생성 방식 사용")
    
    # 질문을 소문자로 변환
    question_lower = question.lower().strip()
    
    # 관련 테이블 감지
    relevant_tables = identify_relevant_tables(question)
    main_table = relevant_tables[0] if relevant_tables else "users"
    
    # 테이블별 필드 정보
    fields = TABLE_FIELDS.get(main_table, ["*"])
    
    # 1. "리뷰가 없는 제품은 어떤 것들이야?"와 같은 질문 처리
    if "리뷰" in question_lower and "없는" in question_lower and "제품" in question_lower:
        return """
SELECT p.product_id, p.name AS 제품명, p.category AS 카테고리, p.price AS 가격
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
WHERE r.review_id IS NULL
ORDER BY p.name;
"""
    
    # 2. "제품별 평균 리뷰 점수를 보여줘" 유형의 질문 처리
    if "평균" in question_lower and "리뷰" in question_lower and ("점수" in question_lower or "평점" in question_lower) and "제품" in question_lower:
        return """
SELECT p.product_id, p.name AS 제품명, AVG(r.rating) AS 평균_평점
FROM products p
LEFT JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.product_id, p.name
ORDER BY 평균_평점 DESC;
"""
    
    # 3. "가장 자주 주문된 제품 TOP 3를 알려줘" 유형의 질문 처리
    if (("가장 자주" in question_lower or "많이" in question_lower) and "주문" in question_lower and "제품" in question_lower) or ("top" in question_lower and "제품" in question_lower):
        # top N의 N 추출
        import re
        limit = 3  # 기본값
        top_n_match = re.search(r'top\s*(\d+)', question_lower)
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
    
    # 기본 쿼리 생성
    field_list = ", ".join(fields[:5]) if len(fields) > 1 else "*"
    return f"SELECT {field_list} FROM {main_table} LIMIT 10;"

def generate_sql_from_question(question: str) -> str:
    """
    자연어 질문을 SQL로 변환합니다. Defog를 우선 사용하고, 실패 시 규칙 기반 접근법으로 대체합니다.
    """
    print(f"[SQL 생성 - '{question}'] SQL 변환 시작")
    
    # Defog로 SQL 생성 시도
    defog_sql = generate_sql_with_defog(question)
    if defog_sql and "SELECT" in defog_sql.upper():
        print(f"[SQL 생성] Defog 성공: {defog_sql}")
        return defog_sql
    
    # Defog 실패 시 규칙 기반 접근법으로 SQL 생성 시도
    print(f"[SQL 생성] Defog 실패, 규칙 기반 접근법으로 전환")
    rule_sql = generate_sql_with_rules(question)
    return rule_sql

def format_sql_results(rows: Union[List[Dict], Dict[str, str]]) -> str:
    """
    SQL 결과를 마크다운 테이블 형식으로 포맷팅합니다.
    """
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

def test_db_connection(db_config=None) -> bool:
    """
    데이터베이스 연결을 테스트합니다.
    """
    if db_config is None:
        db_config = DB_CONFIG
        
    try:
        print(f"데이터베이스 연결 테스트 중... (host: {db_config['host']})")
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
        traceback.print_exc()
        return False 