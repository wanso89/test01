import os
import traceback
import mysql.connector
from typing import Dict, Optional, Any

# MariaDB 연결 설정 (환경 변수에서 가져오기)
DEFAULT_DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASSWORD", "3Ssoft1!"),
    "database": os.environ.get("DB_NAME", "chatbot"),
    "port": int(os.environ.get("DB_PORT", "3306")),
    "connect_timeout": 10  # 연결 타임아웃 설정 추가
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
            cursor.execute(f"SHOW CREATE TABLE `{table}`")
            create_table_stmt = cursor.fetchone()[1]
            schema_parts.append(create_table_stmt + ";")
            
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
                WHERE table_schema = '{db_config['database']}' AND table_name = '{table}'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            
            schema_parts.append(f"\n-- Table `{table}` Column Details:")
            for col in columns:
                col_name, col_type, nullable, key, comment, default = col
                col_info = f"--   {col_name} ({col_type})"
                
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
                  AND table_name = '{table}'
                  AND referenced_table_name IS NOT NULL
            """)
            fks = cursor.fetchall()
            
            if fks:
                schema_parts.append(f"\n-- Table `{table}` Foreign Keys:")
                for fk in fks:
                    constraint, column, ref_table, ref_column = fk
                    schema_parts.append(f"--   {constraint}: {column} -> {ref_table}.{ref_column}")
            
            # 인덱스 정보
            cursor.execute(f"""
                SELECT 
                    index_name,
                    GROUP_CONCAT(column_name ORDER BY seq_in_index) as columns,
                    non_unique
                FROM INFORMATION_SCHEMA.STATISTICS
                WHERE table_schema = '{db_config['database']}' 
                  AND table_name = '{table}'
                GROUP BY index_name, non_unique
            """)
            indexes = cursor.fetchall()
            
            if indexes:
                schema_parts.append(f"\n-- Table `{table}` Indexes:")
                for idx in indexes:
                    idx_name, columns, non_unique = idx
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
                schema_parts.append(f"CREATE OR REPLACE VIEW `{view_name}` AS {view_def};")
                schema_parts.append("\n")

        cursor.close()
        conn.close()
        return "\n".join(schema_parts)
    
    except Exception as e:
        print(f"데이터베이스 스키마 가져오기 오류: {str(e)}")
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
            # 테이블 정의 시작
            schema_lines.append(f"Table: {table}")
            
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
                WHERE table_schema = '{db_config['database']}' AND table_name = '{table}'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            
            column_info = []
            for col in columns:
                col_name, col_type, nullable, key, comment, default = col
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
                  AND table_name = '{table}'
                  AND referenced_table_name IS NOT NULL
            """)
            fks = cursor.fetchall()
            
            if fks:
                schema_lines.append("  Foreign Keys:")
                for fk in fks:
                    col, ref_table, ref_col = fk
                    schema_lines.append(f"    - {col} -> {ref_table}.{ref_col}")
            
            # 인덱스 정보
            cursor.execute(f"""
                SELECT 
                    index_name,
                    GROUP_CONCAT(column_name ORDER BY seq_in_index) as columns,
                    non_unique
                FROM INFORMATION_SCHEMA.STATISTICS
                WHERE table_schema = '{db_config['database']}' 
                  AND table_name = '{table}'
                  AND index_name != 'PRIMARY'
                GROUP BY index_name, non_unique
            """)
            indexes = cursor.fetchall()
            
            if indexes:
                schema_lines.append("  Indexes:")
                for idx in indexes:
                    idx_name, columns, non_unique = idx
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
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"데이터베이스 연결 테스트 오류: {str(e)}")
        return False

if __name__ == "__main__":
    # 스크립트를 직접 실행할 경우 테스트 코드
    print("MariaDB 연결 테스트:")
    
    if test_db_connection():
        print("데이터베이스 연결 성공!")
        print("\n스키마 정보:")
        schema = get_mariadb_schema()
        print(schema[:1000] + "...\n(잘려진 출력)")
        
        print("\nVanna용 스키마 정보:")
        vanna_schema = get_schema_for_vanna()
        print(vanna_schema[:1000] + "...\n(잘려진 출력)")
    else:
        print("데이터베이스 연결 실패!") 