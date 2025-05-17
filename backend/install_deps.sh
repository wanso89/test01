#!/bin/bash
# RAG 챗봇 백엔드 의존성 설치 스크립트

# 가상환경 활성화 (필요시 주석 해제)
# cd /home/test_code/test01/rag-chatbot/backend
# source /home/test_code/.venv/bin/activate

echo "필수 패키지 설치 중..."
pip install -r requirements.txt

# Hugging Face 모델 캐시 디렉토리 생성
mkdir -p /home/test_code/.cache/huggingface/hub

echo "Gemma 토크나이저 관련 패키지 수동 설치"
pip install sentencepiece==0.2.0 protobuf==4.25.1

echo "BitsAndBytes 설치 확인"
pip install -U bitsandbytes>=0.43.0

echo "모든 의존성 설치 완료" 