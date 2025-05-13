"""
Defog SQL 모듈 초기화 유틸리티
"""

import os
import traceback
from typing import Tuple

def initialize_defog() -> Tuple[bool, str]:
    """
    Defog SQL 모듈을 초기화합니다.
    API 키가 없어도 로컬 모드로 정상 작동하도록 설정합니다.
    
    Returns:
        Tuple[bool, str]: 초기화 성공 여부와 메시지
    """
    try:
        # Defog 모듈 가져오기
        try:
            from defog import Defog
        except ImportError:
            return False, "Defog 모듈이 설치되지 않았습니다. 'pip install defog' 명령으로 설치하세요."
        
        # 환경 변수에서 API 키 확인
        defog_api_key = os.environ.get("DEFOG_API_KEY", "")
        
        # 데이터베이스 연결 테스트
        try:
            from app.utils.sql_utils_defog import test_db_connection
            db_connected = test_db_connection()
            
            if not db_connected:
                return False, "Defog 데이터베이스 연결 실패"
            
            # API 키 설정 확인 (있으면 사용, 없어도 로컬 모드 사용)
            if defog_api_key and len(defog_api_key.strip()) > 0:
                # Defog 인스턴스 테스트 초기화
                try:
                    test_instance = Defog(api_key=defog_api_key)
                    return True, "Defog 초기화 성공 (API 키 설정됨)"
                except Exception as api_err:
                    print(f"Defog API 키 검증 오류: {str(api_err)}")
                    print("API 키 오류가 있으나 로컬 모드로 계속 진행합니다...")
            
            # API 키가 없거나 유효하지 않은 경우 - 로컬 모드로 설정
            print("Defog를 로컬 모드로 초기화합니다...")
            
            # 로컬 모드 설정 - utils_sql_defog.py에서 자체 로직 사용
            return True, "Defog 초기화 성공 (로컬 모드)"
                
        except Exception as db_err:
            return False, f"Defog 데이터베이스 연결 테스트 실패: {str(db_err)}"
            
    except Exception as e:
        traceback.print_exc()
        return False, f"Defog 초기화 중 오류: {str(e)}" 