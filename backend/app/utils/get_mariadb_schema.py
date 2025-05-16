import os
import traceback
import mysql.connector
from typing import Dict, Optional, Any, List

# MariaDB 연결 설정 (환경 변수에서 가져오기)
DEFAULT_DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "3Ssoft1!"),
    "database": os.environ.get("DB_NAME", "his"),
    "port": int(os.environ.get("DB_PORT", "3306")),
    "connect_timeout": 10,  # 연결 타임아웃 설정 추가
    "charset": "utf8mb4",   # 문자셋 추가
    "use_unicode": True     # 유니코드 사용 설정
}

# 한국어 데이터 타입 설명 맵핑
KR_DATA_TYPE_MAPPING = {
    "int": "정수형",
    "tinyint": "작은 정수형",
    "smallint": "작은 정수형",
    "mediumint": "중간 정수형",
    "bigint": "큰 정수형",
    "float": "실수형",
    "double": "실수형",
    "decimal": "고정 소수점형",
    "varchar": "가변 문자열",
    "char": "고정 문자열",
    "text": "긴 문자열",
    "longtext": "매우 긴 문자열",
    "mediumtext": "중간 문자열",
    "tinytext": "짧은 문자열",
    "date": "날짜형",
    "time": "시간형",
    "datetime": "날짜시간형",
    "timestamp": "타임스탬프",
    "year": "연도형",
    "enum": "열거형",
    "set": "집합형",
    "blob": "이진 데이터",
    "json": "JSON 데이터"
}

# 테이블 설명 사전 (사용자 정의)
TABLE_DESCRIPTIONS = {
    "users": "사용자 정보를 저장하는 테이블",
    "products": "제품 정보를 저장하는 테이블",
    "orders": "주문 정보를 저장하는 테이블",
    "order_items": "주문 항목 정보를 저장하는 테이블",
    "categories": "카테고리 정보를 저장하는 테이블",
    "reviews": "리뷰 정보를 저장하는 테이블",
    "user_resource": "사용자 자원 정보를 저장하는 테이블",
    "user_monthly_stats": "사용자 월간 통계 정보를 저장하는 테이블"
}

def get_mariadb_schema(db_config: Optional[Dict[str, Any]] = None) -> str:
    """
    MariaDB 스키마 정보를 가져옵니다.
    
    Args:
        db_config: 데이터베이스 연결 설정
        
    Returns:
        str: 데이터베이스 스키마 정보 (DDL 형식)
    """
    if db_config is None:
        db_config = DEFAULT_DB_CONFIG
        
    try:
        # 연결 테스트 추가
        print(f"MariaDB 연결 시도 - 호스트: {db_config['host']}, 사용자: {db_config['user']}, DB: {db_config['database']}")
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # 연결 테스트 쿼리
        cursor.execute("SELECT 1")
        cursor.fetchone()

        # 모든 테이블 조회
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        
        if not tables:
            return "데이터베이스에 테이블이 없습니다."

        schema_parts = []
        
        # 각 테이블의 CREATE TABLE 문 가져오기
        for table in tables:
            # 바이트 객체 처리
            table_name = table.decode('utf-8') if isinstance(table, bytes) else table
            
            cursor.execute(f"SHOW CREATE TABLE `{table_name}`")
            create_result = cursor.fetchone()
            create_table_stmt = create_result[1]
            
            # 바이트 객체 처리
            if isinstance(create_table_stmt, bytes):
                create_table_stmt = create_table_stmt.decode('utf-8')
                
            schema_parts.append(create_table_stmt + ";")
            
            # 테이블 설명 추가 (한국어 지원 강화)
            if table_name in TABLE_DESCRIPTIONS:
                schema_parts.append(f"\n-- 테이블 설명: {TABLE_DESCRIPTIONS[table_name]}")
            
            # 주석 형식으로 상세 정보 추가 (선택 사항)
            cursor.execute(f"""
                SELECT 
                    column_name, 
                    column_type, 
                    is_nullable, 
                    column_key, 
                    column_comment,
                    column_default
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE table_schema = '{db_config['database']}' AND table_name = '{table_name}'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            
            schema_parts.append(f"\n-- Table `{table_name}` Column Details:")
            for col in columns:
                col_name, col_type, nullable, key, comment, default = col
                
                # 바이트 객체 처리
                col_name = col_name.decode('utf-8') if isinstance(col_name, bytes) else col_name
                col_type = col_type.decode('utf-8') if isinstance(col_type, bytes) else col_type
                nullable = nullable.decode('utf-8') if isinstance(nullable, bytes) else nullable
                key = key.decode('utf-8') if isinstance(key, bytes) else key
                comment = comment.decode('utf-8') if isinstance(comment, bytes) else comment
                if default is not None:
                    default = default.decode('utf-8') if isinstance(default, bytes) else default
                
                col_info = f"--   {col_name} ({col_type})"
                
                # 한국어 데이터 타입 설명 추가
                base_type = col_type.split('(')[0].lower()
                if base_type in KR_DATA_TYPE_MAPPING:
                    col_info += f" [{KR_DATA_TYPE_MAPPING[base_type]}]"
                
                if key == 'PRI':
                    col_info += ", Primary Key"
                elif key == 'UNI':
                    col_info += ", Unique"
                
                if nullable == 'NO':
                    col_info += ", NOT NULL"
                
                if default is not None:
                    col_info += f", DEFAULT {default}"
                    
                if comment:
                    col_info += f" // {comment}"
                    
                schema_parts.append(col_info)
            
            # 외래 키 정보
            cursor.execute(f"""
                SELECT
                    constraint_name,
                    column_name,
                    referenced_table_name,
                    referenced_column_name
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE table_schema = '{db_config['database']}' 
                  AND table_name = '{table_name}'
                  AND referenced_table_name IS NOT NULL
            """)
            fks = cursor.fetchall()
            
            if fks:
                schema_parts.append(f"\n-- Table `{table_name}` Foreign Keys:")
                for fk in fks:
                    constraint, column, ref_table, ref_column = fk
                    
                    # 바이트 객체 처리
                    constraint = constraint.decode('utf-8') if isinstance(constraint, bytes) else constraint
                    column = column.decode('utf-8') if isinstance(column, bytes) else column
                    ref_table = ref_table.decode('utf-8') if isinstance(ref_table, bytes) else ref_table
                    ref_column = ref_column.decode('utf-8') if isinstance(ref_column, bytes) else ref_column
                    
                    schema_parts.append(f"--   {constraint}: {column} -> {ref_table}.{ref_column}")
            
            # 인덱스 정보
            cursor.execute(f"""
                SELECT 
                    index_name,
                    GROUP_CONCAT(column_name ORDER BY seq_in_index) as columns,
                    non_unique
                FROM INFORMATION_SCHEMA.STATISTICS
                WHERE table_schema = '{db_config['database']}' 
                  AND table_name = '{table_name}'
                GROUP BY index_name, non_unique
            """)
            indexes = cursor.fetchall()
            
            if indexes:
                schema_parts.append(f"\n-- Table `{table_name}` Indexes:")
                for idx in indexes:
                    idx_name, columns, non_unique = idx
                    
                    # 바이트 객체 처리
                    idx_name = idx_name.decode('utf-8') if isinstance(idx_name, bytes) else idx_name
                    columns = columns.decode('utf-8') if isinstance(columns, bytes) else columns
                    non_unique = non_unique if isinstance(non_unique, int) else (int(non_unique) if isinstance(non_unique, bytes) else non_unique)
                    
                    idx_type = "Non-Unique" if non_unique == 1 else "Unique"
                    schema_parts.append(f"--   {idx_name} ({idx_type}): {columns}")
            
            schema_parts.append("\n")  # 빈 줄 추가

        # 뷰 정보 가져오기
        cursor.execute(f"""
            SELECT 
                table_name, 
                view_definition
            FROM INFORMATION_SCHEMA.VIEWS
            WHERE table_schema = '{db_config['database']}'
        """)
        views = cursor.fetchall()
        
        if views:
            schema_parts.append("\n-- Views:")
            for view in views:
                view_name, view_def = view
                
                # 바이트 객체 처리
                view_name = view_name.decode('utf-8') if isinstance(view_name, bytes) else view_name
                view_def = view_def.decode('utf-8') if isinstance(view_def, bytes) else view_def
                
                schema_parts.append(f"CREATE OR REPLACE VIEW `{view_name}` AS {view_def};")
                schema_parts.append("\n")

        cursor.close()
        conn.close()
        return "\n".join(schema_parts)
    
    except Exception as e:
        print(f"데이터베이스 스키마 가져오기 오류: {str(e)}")
        traceback.print_exc()
        return f"ERROR: 스키마 로딩 실패: {str(e)}"

def get_schema_for_sqlcoder(db_config: Optional[Dict[str, Any]] = None) -> str:
    """
    SQLCoder에 적합한 형식으로 스키마 정보를 가져옵니다.
    
    Args:
        db_config: 데이터베이스 연결 설정
        
    Returns:
        str: 데이터베이스 스키마 정보 (SQLCoder용 형식)
    """
    if db_config is None:
        db_config = DEFAULT_DB_CONFIG
        
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 모든 테이블 조회
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]

        schema_lines = []
        
        # 전체 설명 추가
        schema_lines.append("# 데이터베이스 스키마")
        schema_lines.append("아래는 현재 데이터베이스의 테이블 구조입니다. 한국어 질문을 SQL로 변환할 때 참고하세요.\n")
        
        for table in tables:
            # 테이블 정의 시작
            table_name = table.decode('utf-8') if isinstance(table, bytes) else table
            table_description = f" ({TABLE_DESCRIPTIONS.get(table_name, '테이블')})" if table_name in TABLE_DESCRIPTIONS else ""
            schema_lines.append(f"## 테이블: {table_name}{table_description}")
            
            # 컬럼 정보 가져오기
            cursor.execute(f"""
                SELECT 
                    column_name, 
                    column_type, 
                    is_nullable, 
                    column_key, 
                    column_comment,
                    column_default
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE table_schema = '{db_config['database']}' AND table_name = '{table_name}'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            
            schema_lines.append("컬럼 목록:")
            for col in columns:
                col_name, col_type, nullable, key, comment, default = col
                
                # 바이트 객체 처리
                col_name = col_name.decode('utf-8') if isinstance(col_name, bytes) else col_name
                col_type = col_type.decode('utf-8') if isinstance(col_type, bytes) else col_type
                nullable = nullable.decode('utf-8') if isinstance(nullable, bytes) else nullable
                key = key.decode('utf-8') if isinstance(key, bytes) else key
                comment = comment.decode('utf-8') if isinstance(comment, bytes) else comment
                if default is not None:
                    default = default.decode('utf-8') if isinstance(default, bytes) else default
                
                # 한국어 데이터 타입 설명 추가
                base_type = col_type.split('(')[0].lower()
                kr_type = f" [{KR_DATA_TYPE_MAPPING.get(base_type, '일반')}]" if base_type in KR_DATA_TYPE_MAPPING else ""
                
                col_desc = f"* {col_name} ({col_type}){kr_type}"
                
                # 키 정보 추가
                constraints = []
                if key == 'PRI':
                    constraints.append("기본키")
                elif key == 'UNI':
                    constraints.append("고유키")
                
                if nullable == 'NO':
                    constraints.append("NOT NULL")
                
                if default is not None:
                    constraints.append(f"기본값: {default}")
                
                if constraints:
                    col_desc += f" - {', '.join(constraints)}"
                    
                # 주석 정보 추가
                if comment:
                    col_desc += f" - 설명: {comment}"
                    
                schema_lines.append(col_desc)
            
            # 외래 키 정보 가져오기
            cursor.execute(f"""
                SELECT
                    column_name,
                    referenced_table_name,
                    referenced_column_name
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE table_schema = '{db_config['database']}' 
                  AND table_name = '{table_name}'
                  AND referenced_table_name IS NOT NULL
            """)
            fks = cursor.fetchall()
            
            if fks:
                schema_lines.append("\n외래 키:")
                for fk in fks:
                    col, ref_table, ref_col = fk
                    
                    # 바이트 객체 처리
                    col = col.decode('utf-8') if isinstance(col, bytes) else col
                    ref_table = ref_table.decode('utf-8') if isinstance(ref_table, bytes) else ref_table
                    ref_col = ref_col.decode('utf-8') if isinstance(ref_col, bytes) else ref_col
                    
                    schema_lines.append(f"* {col} -> {ref_table}.{ref_col}")
            
            schema_lines.append("\n")  # 빈 줄 추가

        cursor.close()
        conn.close()
        return "\n".join(schema_lines)
    
    except Exception as e:
        print(f"SQLCoder용 스키마 가져오기 오류: {str(e)}")
        traceback.print_exc()
        return f"ERROR: 스키마 로딩 실패: {str(e)}"

def get_schema_for_vanna(db_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Vanna AI에 적합한 형식으로 스키마 정보를 가져옵니다.
    
    Args:
        db_config: 데이터베이스 연결 설정
        
    Returns:
        str: 데이터베이스 스키마 정보 (Vanna용 형식)
    """
    if db_config is None:
        db_config = DEFAULT_DB_CONFIG
        
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # 모든 테이블 조회
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]

        schema_lines = []
        
        for table in tables:
            # 바이트 객체 처리
            table_name = table.decode('utf-8') if isinstance(table, bytes) else table
            
            # 테이블 정의 시작
            schema_lines.append(f"Table: {table_name}")
            
            # 컬럼 정보 가져오기
            cursor.execute(f"""
                SELECT 
                    column_name, 
                    column_type, 
                    is_nullable, 
                    column_key, 
                    column_comment,
                    column_default
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE table_schema = '{db_config['database']}' AND table_name = '{table_name}'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            
            column_info = []
            for col in columns:
                col_name, col_type, nullable, key, comment, default = col
                
                # 바이트 객체 처리
                col_name = col_name.decode('utf-8') if isinstance(col_name, bytes) else col_name
                col_type = col_type.decode('utf-8') if isinstance(col_type, bytes) else col_type
                nullable = nullable.decode('utf-8') if isinstance(nullable, bytes) else nullable
                key = key.decode('utf-8') if isinstance(key, bytes) else key
                comment = comment.decode('utf-8') if isinstance(comment, bytes) else comment
                if default is not None:
                    default = default.decode('utf-8') if isinstance(default, bytes) else default
                
                col_desc = f"  - {col_name} ({col_type})"
                
                if key == 'PRI':
                    col_desc += ", Primary Key"
                elif key == 'UNI':
                    col_desc += ", Unique"
                
                if nullable == 'NO':
                    col_desc += ", NOT NULL"
                
                if default is not None:
                    col_desc += f", DEFAULT {default}"
                    
                if comment:
                    col_desc += f" // {comment}"
                    
                column_info.append(col_desc)
            
            schema_lines.extend(column_info)
            
            # 외래 키 정보 가져오기
            cursor.execute(f"""
                SELECT
                    column_name,
                    referenced_table_name,
                    referenced_column_name
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                WHERE table_schema = '{db_config['database']}' 
                  AND table_name = '{table_name}'
                  AND referenced_table_name IS NOT NULL
            """)
            fks = cursor.fetchall()
            
            if fks:
                schema_lines.append("  Foreign Keys:")
                for fk in fks:
                    col, ref_table, ref_col = fk
                    
                    # 바이트 객체 처리
                    col = col.decode('utf-8') if isinstance(col, bytes) else col
                    ref_table = ref_table.decode('utf-8') if isinstance(ref_table, bytes) else ref_table
                    ref_col = ref_col.decode('utf-8') if isinstance(ref_col, bytes) else ref_col
                    
                    schema_lines.append(f"    - {col} -> {ref_table}.{ref_col}")
            
            # 인덱스 정보
            cursor.execute(f"""
                SELECT 
                    index_name,
                    GROUP_CONCAT(column_name ORDER BY seq_in_index) as columns,
                    non_unique
                FROM INFORMATION_SCHEMA.STATISTICS
                WHERE table_schema = '{db_config['database']}' 
                  AND table_name = '{table_name}'
                  AND index_name != 'PRIMARY'
                GROUP BY index_name, non_unique
            """)
            indexes = cursor.fetchall()
            
            if indexes:
                schema_lines.append("  Indexes:")
                for idx in indexes:
                    idx_name, columns, non_unique = idx
                    
                    # 바이트 객체 처리
                    idx_name = idx_name.decode('utf-8') if isinstance(idx_name, bytes) else idx_name
                    columns = columns.decode('utf-8') if isinstance(columns, bytes) else columns
                    non_unique = non_unique if isinstance(non_unique, int) else (int(non_unique) if isinstance(non_unique, bytes) else non_unique)
                    
                    idx_type = "Non-Unique" if non_unique == 1 else "Unique"
                    schema_lines.append(f"    - {idx_name} ({idx_type}): {columns}")
            
            schema_lines.append("")  # 빈 줄 추가

        # 테이블 간 관계 정보 추가
        schema_lines.append("Relationships:")
        cursor.execute(f"""
            SELECT 
                kcu.table_name,
                kcu.column_name,
                kcu.referenced_table_name,
                kcu.referenced_column_name
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
            WHERE kcu.table_schema = '{db_config['database']}'
              AND kcu.referenced_table_name IS NOT NULL
            ORDER BY kcu.table_name, kcu.referenced_table_name
        """)
        relationships = cursor.fetchall()
        
        for rel in relationships:
            table, column, ref_table, ref_column = rel
            
            # 바이트 객체 처리
            table = table.decode('utf-8') if isinstance(table, bytes) else table
            column = column.decode('utf-8') if isinstance(column, bytes) else column
            ref_table = ref_table.decode('utf-8') if isinstance(ref_table, bytes) else ref_table
            ref_column = ref_column.decode('utf-8') if isinstance(ref_column, bytes) else ref_column
            
            schema_lines.append(f"  - {table}.{column} -> {ref_table}.{ref_column}")

        cursor.close()
        conn.close()
        return "\n".join(schema_lines)
    
    except Exception as e:
        print(f"Vanna용 스키마 가져오기 오류: {str(e)}")
        traceback.print_exc()
        return f"ERROR: Vanna용 스키마 로딩 실패: {str(e)}"

def test_db_connection(db_config: Optional[Dict[str, Any]] = None) -> bool:
    """
    데이터베이스 연결을 테스트합니다.
    
    Args:
        db_config: 데이터베이스 연결 설정
        
    Returns:
        bool: 연결 성공 여부
    """
    if db_config is None:
        db_config = DEFAULT_DB_CONFIG
        
    try:
        print(f"MariaDB 연결 테스트 - 호스트: {db_config['host']}, 사용자: {db_config['user']}, DB: {db_config['database']}")
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"데이터베이스 연결 테스트 오류: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 스크립트를 직접 실행할 경우 테스트 코드
    print("MariaDB 연결 테스트:")
    
    if test_db_connection():
        print("데이터베이스 연결 성공!")
        print("\n스키마 정보:")
        schema = get_mariadb_schema()
        print(schema[:1000] + "...\n(잘려진 출력)")
        
        print("\nSQLCoder용 스키마 정보:")
        sqlcoder_schema = get_schema_for_sqlcoder()
        print(sqlcoder_schema[:1000] + "...\n(잘려진 출력)")
        
        print("\nVanna용 스키마 정보:")
        vanna_schema = get_schema_for_vanna()
        print(vanna_schema[:1000] + "...\n(잘려진 출력)")
    else:
        print("데이터베이스 연결 실패!") 