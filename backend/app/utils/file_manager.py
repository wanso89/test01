import os
import json
import traceback
from elasticsearch import Elasticsearch
from typing import Dict, List, Any, Optional, Tuple
import uuid
import shutil

def find_file_by_name(filename: str, uploads_dir: str = "app/static/uploads") -> Optional[str]:
    """
    uploads 디렉토리에서 특정 파일명을 포함하는 파일을 찾습니다.
    파일명에 UUID 접두사가 있는 경우 그대로 찾고,
    없는 경우 UUID_filename 패턴으로 찾습니다.
    
    Args:
        filename: 찾을 파일명
        uploads_dir: 업로드 디렉토리 경로
    
    Returns:
        파일의 전체 경로 또는 None (파일이 없는 경우)
    """
    # 직접 파일명으로 존재하는지 확인
    direct_path = os.path.join(uploads_dir, filename)
    if os.path.exists(direct_path):
        return direct_path
    
    # UUID 패턴으로 존재하는지 확인
    for f in os.listdir(uploads_dir):
        if f.endswith(f"_{filename}") or f == filename:
            return os.path.join(uploads_dir, f)
    
    return None

def delete_file_from_es(
    es_client: Elasticsearch, filename: str, index_name: str
) -> Tuple[bool, int, str]:
    """
    Elasticsearch에서 특정 파일명에 해당하는 모든 문서를 삭제합니다.
    
    Args:
        es_client: Elasticsearch 클라이언트
        filename: 삭제할 파일명
        index_name: 인덱스 이름
        
    Returns:
        (성공 여부, 삭제된 문서 수, 메시지)
    """
    try:
        # 해당 파일의 문서 개수 확인
        count_query = {
            "query": {
                "term": {
                    "source": filename
                }
            }
        }
        count_response = es_client.count(index=index_name, body=count_query)
        doc_count = count_response.get("count", 0)
        
        if doc_count == 0:
            return True, 0, f"파일 '{filename}'에 해당하는 문서가 없습니다."
        
        # 삭제 실행
        delete_query = {
            "query": {
                "term": {
                    "source": filename
                }
            }
        }
        
        delete_response = es_client.delete_by_query(
            index=index_name, 
            body=delete_query,
            refresh=True  # 즉시 인덱스 갱신
        )
        
        deleted_count = delete_response.get("deleted", 0)
        return True, deleted_count, f"ES에서 {deleted_count}개 문서 삭제됨"
    
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        return False, 0, f"ES 문서 삭제 중 오류: {error_msg}"

def delete_indexed_file(
    es_client: Elasticsearch, 
    filename: str, 
    index_name: str,
    uploads_dir: str = "app/static/uploads"
) -> Dict[str, Any]:
    """
    Elasticsearch에서 특정 파일에 해당하는 모든 인덱스를 삭제하고,
    저장소에서 해당 파일을 삭제합니다.
    
    Args:
        es_client: Elasticsearch 클라이언트
        filename: 삭제할 파일명
        index_name: 인덱스 이름
        uploads_dir: 업로드 디렉토리 경로
        
    Returns:
        결과 정보를 담은 딕셔너리
    """
    result = {
        "status": "success",
        "file": filename,
        "es_deleted": False,
        "disk_deleted": False,
        "es_docs_deleted": 0,
        "message": ""
    }
    
    # 1. Elasticsearch에서 삭제
    es_success, deleted_count, es_message = delete_file_from_es(
        es_client, filename, index_name
    )
    
    result["es_deleted"] = es_success
    result["es_docs_deleted"] = deleted_count
    
    # 2. 파일 시스템에서 삭제
    try:
        file_path = find_file_by_name(filename, uploads_dir)
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            result["disk_deleted"] = True
            print(f"디스크에서 파일 삭제됨: {file_path}")
        else:
            result["message"] += f"디스크에서 파일을 찾을 수 없습니다. "
    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        result["message"] += f"디스크 파일 삭제 중 오류: {error_msg} "
    
    # 3. 종합 결과 메시지 생성
    if result["es_deleted"] and result["disk_deleted"]:
        result["status"] = "success"
        result["message"] = f"파일 '{filename}'이(가) 완전히 삭제되었습니다."
    elif result["es_deleted"]:
        result["status"] = "partial_success"
        result["message"] = f"파일 '{filename}'의 인덱스는 삭제되었으나 디스크에서 파일을 삭제하지 못했습니다."
    else:
        result["status"] = "error"
        result["message"] = f"파일 '{filename}' 삭제에 실패했습니다. " + result["message"]
    
    return result 