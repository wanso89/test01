"""
SQLCoder 모듈 초기화 유틸리티
"""

import os
import traceback
from typing import Tuple

def initialize_sqlcoder() -> Tuple[bool, str]:
    """
    SQLCoder 모델을 초기화합니다.
    
    Returns:
        Tuple[bool, str]: 초기화 성공 여부와 메시지
    """
    try:
        # SQLCoder 관련 모듈 가져오기
        try:
            from app.utils.sqlcoder_utils import load_sqlcoder_model, test_db_connection
        except ImportError:
            return False, "SQLCoder 모듈을 임포트할 수 없습니다. 관련 파일이 올바르게 설치되었는지 확인하세요."
        
        # 데이터베이스 연결 테스트
        try:
            db_connected = test_db_connection()
            
            if not db_connected:
                return False, "SQLCoder 데이터베이스 연결 실패"
            
            # SQLCoder 모델 설정 확인
            try:
                # 모델과 토크나이저 로드
                model, tokenizer = load_sqlcoder_model()
                
                if model is None or tokenizer is None:
                    print("SQLCoder 모델을 로드할 수 없습니다.")
                    print("로컬 모드로 계속 진행합니다...")
                    return True, "SQLCoder 초기화 성공 (로컬 모드)"
                
                print("SQLCoder 모델을 성공적으로 로드했습니다.")
                return True, "SQLCoder 초기화 성공"
                
            except Exception as model_err:
                print(f"SQLCoder 모델 로드 오류: {str(model_err)}")
                print("모델 로드 오류가 있으나 로컬 모드로 계속 진행합니다...")
                return True, "SQLCoder 초기화 성공 (로컬 모드)"
            
        except Exception as db_err:
            return False, f"SQLCoder 데이터베이스 연결 테스트 실패: {str(db_err)}"
            
    except Exception as e:
        traceback.print_exc()
        return False, f"SQLCoder 초기화 중 오류: {str(e)}" 