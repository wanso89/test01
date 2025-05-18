#!/bin/bash

echo "최적화된 RAG 챗봇 재시작 스크립트"
echo "====================================="

# 현재 작업 디렉토리 확인
CURRENT_DIR=$(pwd)
echo "작업 디렉토리: $CURRENT_DIR"

# 가상 환경 활성화
if [ -d ".venv" ]; then
  echo "가상 환경 활성화..."
  source .venv/bin/activate
else
  echo "가상 환경(.venv)이 존재하지 않습니다. 생성 후 다시 시도하세요."
  exit 1
fi

# Redis 서버 상태 확인
echo "Redis 서버 상태 확인..."
if systemctl is-active --quiet redis; then
  echo "✅ Redis 서버가 실행 중입니다."
else
  echo "⚠️ Redis 서버가 실행되지 않았습니다. 시작합니다..."
  sudo systemctl start redis
  if [ $? -ne 0 ]; then
    echo "❌ Redis 서버 시작 실패. Redis가 설치되어 있는지 확인하세요."
    echo "설치 명령어: sudo yum install redis -y"
    exit 1
  fi
fi

# 백엔드 서버가 이미 실행 중인지 확인
if [ -f "server.pid" ]; then
  PID=$(cat server.pid)
  if ps -p $PID > /dev/null; then
    echo "기존 백엔드 프로세스(PID: $PID) 종료 중..."
    kill -9 $PID
    sleep 2
  else
    echo "이전 PID 파일이 존재하지만 프로세스는 실행 중이 아닙니다."
  fi
  rm -f server.pid
fi

# 추가: pid.txt 확인
if [ -f "pid.txt" ]; then
  PID=$(cat pid.txt)
  if ps -p $PID > /dev/null; then
    echo "기존 백엔드 프로세스(PID: $PID) 종료 중..."
    kill -9 $PID
    sleep 2
  fi
  rm -f pid.txt
fi

# 백엔드 시작
echo "최적화된 백엔드 서버 시작 중..."
cd backend
nohup uvicorn app.main:app --host 0.0.0.0 --port 8001 > ../nohup.out 2>&1 &
NEW_PID=$!
cd ..
echo $NEW_PID > server.pid

echo "백엔드 서버 PID: $NEW_PID (server.pid에 저장됨)"
echo "로그 확인: tail -f nohup.out"
echo "====================================="
echo "최적화된 서버가 성공적으로 시작되었습니다!"
echo "성능 확인: http://localhost:8001/api/chat (POST)" 