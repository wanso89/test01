#!/usr/bin/env python
"""
OCR 기능 테스트 스크립트
PaddleOCR을 사용하여 이미지에서 텍스트를 추출합니다.
"""

import os
import sys
import time
from PIL import Image
import numpy as np
from paddleocr import PaddleOCR

def main():
    # 인수 확인
    if len(sys.argv) != 2:
        print(f"사용법: python {sys.argv[0]} <이미지_파일_경로>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # 파일 존재 확인
    if not os.path.exists(image_path):
        print(f"오류: 파일 '{image_path}'를 찾을 수 없습니다.")
        sys.exit(1)
    
    # PaddleOCR 초기화
    print(f"PaddleOCR 초기화 중...")
    start_time = time.time()
    
    ocr = PaddleOCR(
        use_angle_cls=True,  # 텍스트 방향 감지
        lang="korean",       # 한국어 지원
        use_gpu=True,        # GPU 사용 (가능한 경우)
        show_log=False,      # 로그 출력 비활성화
        det_db_thresh=0.3,   # 텍스트 검출 임계값
        det_db_box_thresh=0.5 # 텍스트 박스 검출 임계값
    )
    
    init_time = time.time() - start_time
    print(f"PaddleOCR 초기화 완료 (소요 시간: {init_time:.2f}초)")
    
    # 이미지 로드 및 OCR 처리
    print(f"이미지 '{image_path}' OCR 처리 중...")
    ocr_start_time = time.time()
    
    try:
        # 이미지 로드
        image = Image.open(image_path)
        image_np = np.array(image)
        
        # OCR 처리
        result = ocr.ocr(image_np, cls=True)
        
        ocr_time = time.time() - ocr_start_time
        print(f"OCR 처리 완료 (소요 시간: {ocr_time:.2f}초)")
        
        # 결과 출력
        if result and result[0]:
            print("\n--- 추출된 텍스트 ---")
            for line in result[0]:
                if line and len(line) >= 2 and line[1] and len(line[1]) >= 2:
                    text, confidence = line[1][0], line[1][1]
                    print(f"[신뢰도: {confidence:.2f}] {text}")
        else:
            print("텍스트가 추출되지 않았습니다.")
        
    except Exception as e:
        print(f"OCR 처리 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 