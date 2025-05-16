import os
import torch
import re
import mysql.connector
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import datetime

# MariaDB 연결 설정
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "3Ssoft1!",
    "database": "his",
    "charset": "utf8mb4",
    "use_unicode": True,
    "connect_timeout": 10
}

# SQLCoder 모델 경로
SQLCODER_MODEL_PATH = "/home/root/sqlcoder-7b-2"

# SQLCoder 시스템 프롬프트
SQLCODER_SYSTEM_PROMPT = """
너는 MariaDB 8.0 SQL 전문가입니다. 주어진 스키마 정보를 바탕으로 사용자의 자연어 질문을 SQL 쿼리로 변환해주세요.
한국어 질문도 정확하게 이해하고 적절한 SQL을 생성해야 합니다.

다음 규칙을 반드시 지켜주세요:
1. 반드시 MariaDB 8.0 문법만 사용하세요
2. PostgreSQL 문법(::, NULLS LAST, filter 등)은 절대 사용하지 마세요
3. 타입 변환은 CAST(x AS CHAR) 또는 CONVERT() 함수를 사용하세요 (VARCHAR 대신 CHAR 사용)
4. 날짜 형식 변환은 DATE_FORMAT() 함수를 사용하세요
5. NULL 처리는 IFNULL(), COALESCE() 함수를 사용하세요
6. 테이블과 컬럼 이름을 정확하게 사용하세요 (대소문자 구분)
7. 필요한 경우 JOIN, WHERE, GROUP BY, ORDER BY 등을 적절히 포함하세요
8. 조건절에서는 정확한 연산자를 사용하세요 (예: '90% 이상' -> '>= 90', '이전 30일' -> 'DATE_SUB(NOW(), INTERVAL 30 DAY)')
9. 결과 컬럼에는 의미 있는 별칭(alias)을 사용하세요
10. 한국어 의미를 SQL 구문으로 정확히 변환하세요 (예: '상위 5개' -> 'LIMIT 5', '많은 순서대로' -> 'ORDER BY ... DESC')
11. 날짜 처리 시 적절한 MariaDB 함수를 사용하세요 (DATEDIFF, DATE_FORMAT, DATE_ADD, CURDATE 등)
12. 집계 함수(COUNT, SUM, AVG, MAX, MIN)와 그룹화를 적절히 사용하세요
13. 윈도우 함수 사용 시 MariaDB 8.0 문법을 따라야 합니다
14. filter 절은 MariaDB에서 지원하지 않으므로 CASE 문을 사용하세요
15. 나이 계산 시 TIMESTAMPDIFF(YEAR, birth_date, CURDATE())를 사용하세요

참고: 
* '~별', '~당', '~마다'는 GROUP BY를 의미할 수 있습니다.
* '최근', '지난'은 날짜 필터링이 필요할 수 있습니다.
* '비율', '퍼센트'는 나눗셈과 100 곱하기가 필요할 수 있습니다.

# MariaDB 8.0 예시 쿼리
-- 예시1: 제품별 평균 평점
SELECT p.name AS product_name, AVG(r.rating) AS avg_rating
FROM products p
JOIN reviews r ON p.product_id = r.product_id
GROUP BY p.product_id, p.name
ORDER BY avg_rating DESC;

-- 예시2: 최근 30일 내 가입한 30세 이상 사용자
SELECT user_id, name, email, 
       TIMESTAMPDIFF(YEAR, birth_date, CURDATE()) AS age
FROM users
WHERE register_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
AND TIMESTAMPDIFF(YEAR, birth_date, CURDATE()) >= 30;

-- 예시3: 카테고리별 완료된 작업 비율
SELECT 
    category,
    COUNT(*) AS total_count,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_count,
    (SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*)) * 100 AS completion_rate
FROM tasks
GROUP BY category
ORDER BY completion_rate DESC;
"""

# SQLCoder 사용자 프롬프트 템플릿
SQLCODER_USER_PROMPT_TEMPLATE = """
### 데이터베이스 스키마:
{table_info}

### 사용자 질문:
{question}

### MariaDB SQL:
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

# 로컬 변수로 모델과 토크나이저 저장
_sqlcoder_model = None
_sqlcoder_tokenizer = None


def load_sqlcoder_model():
    """
    SQLCoder 모델과 토크나이저를 로드합니다.
    처음 호출시에만 로드하고, 이후에는 캐시된 모델을 반환합니다.
    """
    global _sqlcoder_model, _sqlcoder_tokenizer
    
    # 이미 로드된 모델이 있으면 재사용
    if _sqlcoder_model is not None and _sqlcoder_tokenizer is not None:
        return _sqlcoder_model, _sqlcoder_tokenizer
    
    try:
        print(f"SQLCoder 모델 로드 시작: {SQLCODER_MODEL_PATH}")
        
        # 메모리 최적화를 위한 양자화 설정
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            SQLCODER_MODEL_PATH,
            use_fast=True,
            trust_remote_code=True,
        )
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            SQLCODER_MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # 토크나이저 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 평가 모드로 설정 (배치 정규화 레이어 비활성화)
        model.eval()
        
        # 모델 메모리 사용량 출력
        if torch.cuda.is_available():
            print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        _sqlcoder_model = model
        _sqlcoder_tokenizer = tokenizer
        
        print("SQLCoder 모델 로드 완료")
        return model, tokenizer
    
    except Exception as e:
        print(f"SQLCoder 모델 로드 중 오류 발생: {e}")
        traceback.print_exc()
        return None, None


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
        "다른": "!=",
        "크거나 같은": ">=",
        "작거나 같은": "<="
    }
    
    # 날짜 표현 변환
    date_mapping = {
        "오늘": "CURDATE()",
        "어제": "DATE_SUB(CURDATE(), INTERVAL 1 DAY)",
        "그제": "DATE_SUB(CURDATE(), INTERVAL 2 DAY)",
        "이번 주": "DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY)",
        "지난 주": "DATE_SUB(DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY), INTERVAL 7 DAY)",
        "이번 달": "DATE_FORMAT(CURDATE(), '%Y-%m-01')",
        "지난 달": "DATE_FORMAT(DATE_SUB(CURDATE(), INTERVAL 1 MONTH), '%Y-%m-01')"
    }
    
    # 정렬 및 제한 표현 변환
    sort_limit_mapping = {
        "상위": "ORDER BY ... DESC LIMIT",
        "최근": "ORDER BY date_col DESC LIMIT",
        "많은 순": "ORDER BY ... DESC",
        "적은 순": "ORDER BY ... ASC",
        "높은 순": "ORDER BY ... DESC",
        "낮은 순": "ORDER BY ... ASC"
    }
    
    # 연산자 변환 패턴 적용 (개선된 정규식)
    for kr_op, en_op in operator_mapping.items():
        # 숫자 + 연산자 패턴 찾기 (예: "90% 이상" -> "90% >=")
        pattern = r'(\d+(?:\.\d+)?)(\s*%?\s*' + kr_op + r')'
        question = re.sub(pattern, r'\1 ' + en_op, question)
    
    # 테이블 이름 힌트 추가
    for table, kr_names in TABLE_KR_MAPPING.items():
        for kr_name in kr_names:
            if kr_name in question:
                # 질문에 테이블 힌트 추가 (괄호 안에)
                if f"({table})" not in question and f"({kr_name}: {table})" not in question:
                    question = question.replace(kr_name, f"{kr_name} ({table})")
                break
    
    # 날짜 표현에 대한 SQL 힌트 추가
    for kr_date, sql_date in date_mapping.items():
        if kr_date in question:
            # 이미 힌트가 있는지 확인
            if f"({sql_date})" not in question:
                question = question.replace(kr_date, f"{kr_date} ({sql_date})")
    
    # 정렬 및 제한 표현에 대한 SQL 힌트 추가
    for kr_sort, sql_sort in sort_limit_mapping.items():
        if kr_sort in question:
            if f"({sql_sort})" not in question:
                question = question.replace(kr_sort, f"{kr_sort} ({sql_sort})")
    
    # "최근 N일" 패턴 처리
    days_pattern = r'최근\s+(\d+)\s*일'
    days_match = re.search(days_pattern, question)
    if days_match:
        days = days_match.group(1)
        sql_hint = f"DATE_SUB(CURDATE(), INTERVAL {days} DAY)"
        original = days_match.group(0)
        question = question.replace(original, f"{original} ({sql_hint})")
    
    # "N개" -> "LIMIT N" 패턴 처리
    limit_pattern = r'(\d+)\s*개'
    limit_match = re.search(limit_pattern, question)
    if limit_match:
        limit = limit_match.group(1)
        original = limit_match.group(0)
        # 앞에 "상위", "최근" 등의 단어가 있는지 확인
        if re.search(r'(상위|최근|처음)\s+' + limit_pattern, question):
            question = question.replace(original, f"{original} (LIMIT {limit})")
    
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
    for table in TABLE_KR_MAPPING.keys():
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
    
    return relevant_tables


def filter_schema_for_tables(schema: str, relevant_tables: List[str]) -> str:
    """
    전체 스키마에서 관련 테이블 정보만 추출합니다.
    """
    if not relevant_tables:
        return schema  # 관련 테이블이 식별되지 않으면 전체 스키마 반환
    
    schema_lines = schema.split('\n')
    filtered_lines = []
    include_table = False
    current_table = None
    
    for line in schema_lines:
        # 새로운 테이블 시작 감지
        if line.startswith("Table: "):
            table_name = line.replace("Table: ", "").strip()
            current_table = table_name
            include_table = table_name in relevant_tables
        
        # 현재 테이블이 관련 테이블이면 라인 포함
        if include_table or current_table is None:
            filtered_lines.append(line)
            
        # 빈 줄이면 현재 테이블 정의 종료
        elif line.strip() == "":
            # 빈 줄 추가 후 다음 테이블 확인 준비
            filtered_lines.append(line)
            current_table = None
            include_table = False
    
    return "\n".join(filtered_lines)


def generate_sql_with_sqlcoder(question: str, schema: str = None) -> str:
    """
    SQLCoder 모델을 사용하여 자연어 질문을 SQL로 변환합니다.
    
    Args:
        question: 사용자 질문
        schema: 데이터베이스 스키마 정보
        
    Returns:
        str: 생성된 SQL 쿼리
    """
    try:
        if schema is None:
            # 스키마 정보가 없으면 SQLCoder용 스키마 가져오기
            from app.utils.get_mariadb_schema import get_schema_for_sqlcoder
            schema = get_schema_for_sqlcoder()
            
            # 스키마 정보 오류 체크
            if schema.startswith("ERROR:"):
                print(f"스키마 정보 로드 오류: {schema}")
                schema = "# 데이터베이스 스키마 로드 실패\n기본 테이블 정보를 바탕으로 SQL을 생성합니다."
        
        # 모델과 토크나이저 로드
        model, tokenizer = load_sqlcoder_model()
        
        if model is None or tokenizer is None:
            print("SQLCoder 모델 로드 실패")
            return "-- SQLCoder 모델 로드 실패"
        
        # SQLCoder 프롬프트 생성
        prompt = f"""### 마리아디비(MariaDB) 스키마:
{schema}

### 사용자 질문:
{question}

### 마리아디비(MariaDB) SQL 쿼리:
"""
        
        print(f"SQLCoder 프롬프트 생성 완료 (길이: {len(prompt)} 문자)")
        
        # 모델 입력 토큰화 및 생성
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 생성 설정
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": 512,
                "temperature": 0.1,  # 낮은 온도로 결정적인 응답 생성
                "top_p": 0.95,
                "repetition_penalty": 1.1,
                "num_return_sequences": 1,
                "pad_token_id": tokenizer.eos_token_id,
                "do_sample": True
            }
            
            try:
                # SQL 생성
                outputs = model.generate(**inputs, **generation_config)
                
                # 생성된 텍스트 디코딩
                sql = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
                
                # 결과 정리 (주석, 여러 쿼리 등 제거)
                sql = sql.strip()
                
                # 추가 쿼리나 주석 제거 (첫 번째 '--;' 또는 ';' 이후 텍스트 제거)
                for delimiter in ["--;", ";"]:
                    pos = sql.find(delimiter)
                    if pos > 0:
                        # 첫 번째 구분자 위치에서 자르기
                        sql = sql[:pos + 1]
                        break
                
                # 마지막에 세미콜론 추가되어 있지 않으면 추가
                if not sql.endswith(";"):
                    sql += ";"
                
                return sql
            except Exception as gen_error:
                print(f"SQL 생성 중 오류: {str(gen_error)}")
                return f"-- SQL 생성 중 오류 발생: {str(gen_error)}"
    except Exception as e:
        print(f"SQLCoder 쿼리 생성 오류: {str(e)}")
        return f"-- SQL 생성 중 오류 발생: {str(e)}"


def run_sql_query(query: str, db_config=None) -> Union[List[Dict], Dict[str, str]]:
    """
    SQL 쿼리를 실행하고 결과를 반환합니다.
    
    Args:
        query: 실행할 SQL 쿼리
        db_config: 데이터베이스 연결 설정
        
    Returns:
        Union[List[Dict], Dict[str, str]]: 쿼리 결과 또는 오류 정보
    """
    if db_config is None:
        db_config = DB_CONFIG
    
    # SQL 주석 제거
    clean_query = re.sub(r'--.*?\n', '\n', query)
    clean_query = re.sub(r'/\*.*?\*/', '', clean_query, flags=re.DOTALL)
    
    # SQL 쿼리에서 위험한 명령어 체크
    if any(cmd in clean_query.upper() for cmd in [
        "DROP", "TRUNCATE", "DELETE", "UPDATE", "ALTER", 
        "CREATE DATABASE", "GRANT", "REVOKE"
    ]):
        return {"error": "위험한 SQL 명령어는 실행할 수 없습니다"}
    
    try:
        # MariaDB 연결
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # 쿼리 실행
        try:
            print(f"SQL 쿼리 실행: {clean_query}")
            cursor.execute(clean_query)
            
            # SELECT 문인 경우 결과 반환
            if clean_query.strip().upper().startswith("SELECT"):
                # 결과 가져오기
                results = cursor.fetchall()
                
                # 시간 형식 데이터 처리
                results = [{k: (v.isoformat() if isinstance(v, (datetime.date, datetime.datetime)) else v) 
                          for k, v in row.items()} for row in results]
                
                # 결과를 마크다운 테이블로 변환
                if not results:
                    return "⚠️ 조건에 맞는 레코드를 찾을 수 없습니다."
                
                # 마크다운 테이블 생성
                headers = list(results[0].keys())
                markdown_table = "| " + " | ".join(headers) + " |\n"
                markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                
                for row in results:
                    row_values = []
                    for header in headers:
                        value = row[header]
                        if value is None:
                            value = "NULL"
                        elif isinstance(value, str) and len(value) > 50:
                            value = value[:47] + "..."
                        row_values.append(str(value))
                    markdown_table += "| " + " | ".join(row_values) + " |\n"
                
                return markdown_table
            else:
                # SELECT가 아닌 경우 (INSERT, CREATE 등)
                rows_affected = cursor.rowcount
                return f"쿼리가 성공적으로 실행되었습니다. {rows_affected}개의 행이 영향을 받았습니다."
        except mysql.connector.Error as e:
            print(f"SQL 쿼리 실행 오류: {str(e)}")
            return {"error": str(e)}
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        print(f"데이터베이스 연결 오류: {str(e)}")
        return {"error": f"데이터베이스 연결 오류: {str(e)}"}


def fix_sql_for_mariadb(sql_query: str) -> str:
    """
    PostgreSQL 문법을 MariaDB 문법으로 변환합니다.
    
    Args:
        sql_query: SQLCoder가 생성한 SQL 쿼리
        
    Returns:
        str: MariaDB 호환 SQL 쿼리
    """
    # 원본 쿼리 보존
    original_query = sql_query
    
    try:
        # 1. ::text -> CAST(... AS CHAR) 변환
        sql_query = re.sub(r'(\w+)::text', r'CAST(\1 AS CHAR)', sql_query)
        
        # 2. ::date -> CAST(... AS DATE) 변환
        sql_query = re.sub(r'(\w+)::date', r'CAST(\1 AS DATE)', sql_query)
        
        # 3. ::integer -> CAST(... AS SIGNED) 변환
        sql_query = re.sub(r'(\w+)::integer', r'CAST(\1 AS SIGNED)', sql_query)
        
        # 4. CAST(... AS varchar) -> CAST(... AS CHAR) 변환
        sql_query = re.sub(r'CAST\s*\((.*?)\s+AS\s+varchar(\(\d+\))?\)', r'CAST(\1 AS CHAR)', sql_query)
        
        # 5. NULLS LAST, NULLS FIRST 제거
        sql_query = re.sub(r'\s+NULLS\s+(FIRST|LAST)', '', sql_query)
        
        # 6. filter(...) WHERE 구문 변환 (복잡한 패턴은 단순화)
        sql_query = re.sub(r'filter\s*\(\s*WHERE\s+(.*?)\s*\)', r'CASE WHEN \1 THEN 1 ELSE 0 END', sql_query)
        
        # 7. 복잡한 문자열 변환 (TO_CHAR 등)
        sql_query = re.sub(r'TO_CHAR\s*\(\s*(.*?)\s*,\s*\'([^\']*)\'\s*\)', r'DATE_FORMAT(\1, \'\2\')', sql_query)
        
        # 8. EXTRACT(YEAR FROM date) -> YEAR(date) 변환
        sql_query = re.sub(r'EXTRACT\s*\(\s*YEAR\s+FROM\s+(.*?)\s*\)', r'YEAR(\1)', sql_query)
        sql_query = re.sub(r'EXTRACT\s*\(\s*MONTH\s+FROM\s+(.*?)\s*\)', r'MONTH(\1)', sql_query)
        sql_query = re.sub(r'EXTRACT\s*\(\s*DAY\s+FROM\s+(.*?)\s*\)', r'DAY(\1)', sql_query)
        
        # 9. 'YYYY-MM-DD' 형식의 문자열을 MySQL 형식으로 변환
        sql_query = re.sub(r'\'YYYY-MM-DD\'', r'\'%Y-%m-%d\'', sql_query)
        
        # 10. age() 함수를 TIMESTAMPDIFF로 변환
        sql_query = re.sub(r'age\s*\(\s*(.*?)\s*\)', r'TIMESTAMPDIFF(YEAR, \1, CURDATE())', sql_query)
        
        # 11. 현재 날짜 함수 (current_date -> CURDATE())
        sql_query = re.sub(r'current_date', r'CURDATE()', sql_query, flags=re.IGNORECASE)
        
        # 쿼리가 변경되었으면 로그 출력
        if original_query != sql_query:
            print("[SQL 변환] PostgreSQL → MariaDB 문법 변환 적용됨")
            
        return sql_query
    except Exception as e:
        print(f"SQL 변환 중 오류 발생: {str(e)}")
        # 변환 중 오류 발생 시 원본 쿼리 반환
        return original_query


def generate_sql_from_question(question: str) -> str:
    """
    자연어 질문을 SQL 쿼리로 변환합니다.
    
    Args:
        question: 사용자 질문
        
    Returns:
        str: 생성된 SQL 쿼리
    """
    try:
        # 원본 질문 출력
        print(f"원본 질문: {question}")
        
        # 1. 전처리 및 관련 테이블 식별
        processed_question = preprocess_korean_query(question)
        print(f"전처리된 질문: {processed_question}")
        
        relevant_tables = identify_relevant_tables(processed_question)
        print(f"식별된 관련 테이블: {relevant_tables}")
        
        # 2. 스키마 정보 가져오기
        full_schema = get_mariadb_schema()
        
        # 관련 테이블로 스키마 필터링 (선택사항)
        filtered_schema = filter_schema_for_tables(full_schema, relevant_tables) if relevant_tables else full_schema
        
        # 3. SQL 쿼리 생성
        sql_query = generate_sql_with_sqlcoder(processed_question, filtered_schema)
        
        # 4. MariaDB 호환성을 위한 후처리
        sql_query = fix_sql_for_mariadb(sql_query)
        
        return sql_query
    except Exception as e:
        print(f"SQL 생성 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return f"-- SQL 생성 중 오류 발생: {str(e)}"


def test_db_connection(db_config=None) -> bool:
    """
    데이터베이스 연결을 테스트합니다.
    """
    if db_config is None:
        db_config = DB_CONFIG
        
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchall()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"데이터베이스 연결 테스트 오류: {str(e)}")
        return False 