#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PaddleOCR 테스트 스크립트
"""

import os
import sys
import asyncio
import argparse
import time
from pathlib import Path
import logging # 로깅 모듈 추가

# 필요한 모듈 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "backend")))

# PaddleOCR 유틸리티 임포트
from app.utils.ocr_utils import (
    extract_text_from_image, 
    extract_text_from_pdf_with_ocr,
    get_paddle_ocr_version
)

# 로거 설정
logger = logging.getLogger("ocr_processing")
logger.setLevel(logging.INFO)

# 파일 핸들러 추가 (로그를 파일에 기록)
log_file_path = "ocr_process.log"
file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8') # mode='w'로 매번 새로 작성
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# 콘솔 핸들러 추가 (기존 print 대신 logger 사용 가능)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)

def print_separator(title):
    """구분선 출력 함수"""
    print("\n" + "="*50)
    print(f" {title} ".center(50, "="))
    print("="*50 + "\n")

async def test_image_ocr(image_path):
    """이미지 OCR 테스트"""
    print_separator("이미지 OCR 테스트")
    print(f"테스트 이미지: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"오류: 이미지 파일이 존재하지 않습니다: {image_path}")
        return
    
    start_time = time.time()
    extracted_text = await extract_text_from_image(image_path)
    elapsed_time = time.time() - start_time
    
    print(f"\n추출된 텍스트 ({elapsed_time:.2f}초 소요):")
    print("-" * 50)
    print(extracted_text)
    print("-" * 50)
    print(f"총 {len(extracted_text)} 글자 추출됨")

async def test_pdf_ocr(pdf_path):
    """PDF OCR 테스트"""
    print_separator("PDF OCR 테스트")
    print(f"테스트 PDF: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        print(f"오류: PDF 파일이 존재하지 않습니다: {pdf_path}")
        return
    
    start_time = time.time()
    extracted_text = await extract_text_from_pdf_with_ocr(pdf_path)
    elapsed_time = time.time() - start_time
    
    # 결과 파일로 저장
    output_file_path = "test_pdf_ocr_result.txt"
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        print(f"PDF OCR 결과가 다음 파일에 저장되었습니다: {output_file_path}")
    except Exception as e:
        print(f"PDF OCR 결과를 파일에 저장하는 중 오류 발생: {e}")

    print(f"\n추출된 텍스트 ({elapsed_time:.2f}초 소요):")
    print("-" * 50)
    # 텍스트가 너무 길면 일부만 출력
    if len(extracted_text) > 500:
        print(extracted_text[:500] + "...\n(텍스트가 너무 길어 일부만 표시합니다)")
    else:
        print(extracted_text)
    print("-" * 50)
    print(f"총 {len(extracted_text)} 글자 추출됨")

async def main():
    parser = argparse.ArgumentParser(description="PaddleOCR 테스트 스크립트")
    parser.add_argument("--image", help="OCR로 처리할 이미지 파일 경로")
    parser.add_argument("--pdf", help="OCR로 처리할 PDF 파일 경로")
    parser.add_argument("--all", action="store_true", help="기본 테스트 이미지와 PDF 모두 처리")
    
    args = parser.parse_args()
    
    logger.info(f"로그 파일 위치: {os.path.abspath(log_file_path)}")

    # PaddleOCR 버전 출력
    paddle_version = get_paddle_ocr_version()
    print(f"PaddleOCR 버전: {paddle_version}")
    
    if args.all:
        # 기본 테스트 파일 사용
        image_path = "test_ocr_image.png"
        pdf_path = "test.pdf"  # 적절한 테스트 PDF 파일 필요
        
        if os.path.exists(image_path):
            await test_image_ocr(image_path)
        else:
            print(f"기본 테스트 이미지를 찾을 수 없습니다: {image_path}")
        
        if os.path.exists(pdf_path):
            await test_pdf_ocr(pdf_path)
        else:
            print(f"기본 테스트 PDF를 찾을 수 없습니다: {pdf_path}")
    else:
        if args.image:
            await test_image_ocr(args.image)
        
        if args.pdf:
            await test_pdf_ocr(args.pdf)
        
        if not (args.image or args.pdf):
            print("사용법: python test_paddle_ocr.py --image 이미지파일 --pdf PDF파일")
            print("       python test_paddle_ocr.py --all")

if __name__ == "__main__":
    asyncio.run(main()) 