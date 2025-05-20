#!/usr/bin/env python
"""
Tesseract OCR 테스트 스크립트
이미지 파일에서 텍스트를 추출합니다.
"""

import pytesseract
from PIL import Image
import sys
import time
import os.path
import re

def test_ocr(image_path):
    """테서렉트 OCR을 사용하여 이미지에서 텍스트를 추출합니다."""
    
    if not os.path.exists(image_path):
        print(f"오류: 파일 '{image_path}'이 존재하지 않습니다.")
        return False
    
    try:
        # 테서렉트 버전 확인
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract OCR 버전: {version}")
        except Exception as e:
            print(f"경고: Tesseract 버전을 확인할 수 없습니다: {e}")
        
        # 사용 가능한 언어 확인
        try:
            langs = pytesseract.get_languages()
            print(f"사용 가능한 언어: {', '.join(langs) if langs else '없음'}")
        except Exception as e:
            print(f"경고: Tesseract 언어 목록을 확인할 수 없습니다: {e}")
        
        # 이미지 로드
        print(f"이미지 로드 중: {image_path}")
        start_time = time.time()
        image = Image.open(image_path)
        print(f"이미지 크기: {image.width} x {image.height}, 포맷: {image.format}")
        
        # OCR 수행
        print("OCR 처리 중...")
        text = pytesseract.image_to_string(image, lang='kor+eng')
        
        # 텍스트 정리
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        # 결과 출력
        elapsed = time.time() - start_time
        print(f"OCR 처리 완료 (소요 시간: {elapsed:.2f}초)")
        print("\n----- 추출된 텍스트 -----")
        print(cleaned_text)
        print("--------------------------\n")
        
        # 단어별 신뢰도 정보
        print("단어별 위치 및 정보 분석...")
        data = pytesseract.image_to_data(image, lang='kor+eng', output_type=pytesseract.Output.DICT)
        
        n_boxes = len(data['text'])
        word_count = sum(1 for text in data['text'] if text.strip())
        print(f"감지된 단어 수: {word_count}")
        
        # 신뢰도 정보 출력
        if word_count > 0:
            print("\n----- 높은 신뢰도 단어 샘플 -----")
            for i in range(n_boxes):
                if data['text'][i].strip() and float(data['conf'][i]) > 70:
                    print(f"단어: '{data['text'][i]}', 신뢰도: {data['conf'][i]}")
                    # 최대 5개까지만 출력
                    if i >= 5:
                        print("...")
                        break
        
        return True
    
    except Exception as e:
        print(f"OCR 처리 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"사용법: python {sys.argv[0]} <이미지_파일_경로>")
        sys.exit(1)
    
    success = test_ocr(sys.argv[1])
    sys.exit(0 if success else 1) 