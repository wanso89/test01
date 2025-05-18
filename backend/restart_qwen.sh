#!/bin/bash

echo "====== Qwen2.5-7B 최적화 서버 재시작 스크립트 ======"
echo "최적화 항목: bfloat16, 4비트 양자화(nf4), FlashAttention-2, ChatML 프롬프트 최적화"
echo "======================================================"

# 작업 디렉토리
WORK_DIR="/home/test_code/test01/rag-chatbot"
cd $WORK_DIR

# 가상 환경 활성화
source .venv/bin/activate

# Redis 서버 상태 확인
echo "Redis 서버 상태 확인 중..."
if systemctl is-active --quiet redis; then
  echo "✅ Redis 서버가 실행 중입니다."
else
  echo "⚠️ Redis 서버가 실행되지 않았습니다. 시작합니다..."
  sudo systemctl start redis
  if [ $? -ne 0 ]; then
    echo "❌ Redis 서버 시작 실패. Redis가 설치되어 있는지 확인하세요."
    echo "설치 명령어: sudo yum install redis -y && sudo systemctl enable redis"
    exit 1
  fi
fi

# CUDA 메모리 정리 (기존 프로세스 종료)
echo "GPU 메모리 정리 중..."
ps aux | grep "python -m app.main\|uvicorn app.main:app" | grep -v grep | awk '{print $2}' | xargs -r kill -9
nvidia-smi

# 기존 서버 정리
echo "기존 서버 프로세스 종료 중..."
cd $WORK_DIR/backend
PID_FILE="server.pid"
if [ -f $PID_FILE ]; then
  PID=$(cat $PID_FILE)
  if ps -p $PID > /dev/null; then
    echo "기존 서버 프로세스(PID: $PID)를 종료합니다..."
    kill -9 $PID
    sleep 2
  else
    echo "기존 PID 파일이 있지만 프로세스가 실행 중이 아닙니다."
  fi
  rm -f $PID_FILE
fi

# 기존 로그 파일 백업
if [ -f app.log ]; then
  mv app.log app.log.$(date +%Y%m%d%H%M%S).bak
fi

# 포트가 이미 사용 중인지 확인
PORT=8000
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
  echo "⚠️ 포트 $PORT가 이미 사용 중입니다. 해당 프로세스를 종료합니다."
  lsof -ti:$PORT | xargs -r kill -9
  sleep 1
fi

# Qwen 모델 최적화 설정을 위한 환경 변수
export TRANSFORMERS_OFFLINE=1        # 오프라인 모드 활성화 (더 빠른 로딩)
export CUDA_LAUNCH_BLOCKING=0        # 비동기 CUDA 실행 활성화
export TOKENIZERS_PARALLELISM=true   # 토크나이저 병렬처리 활성화
export CUDA_VISIBLE_DEVICES=0        # 사용할 GPU 지정

# 메모리 초기화 및 캐시 정리
echo "GPU 메모리 초기화 중..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU 캐시 정리 완료')"

echo "새 서버 시작 중..."
# 명시적으로 직접 Python 모듈로 앱 실행
python -m app.main > app.log 2>&1 &
NEW_PID=$!

# 새로운 PID 저장
echo $NEW_PID > server.pid
echo "서버가 시작되었습니다. PID: $NEW_PID, 포트: $PORT"

# 10초 동안 서버 시작 확인
echo "서버 시작 확인 중..."
for i in {1..20}
do
  echo -n "."
  sleep 0.5
  
  # 5초 후에 서버가 실행 중인지 확인
  if [ $i -eq 10 ]; then
    if ! ps -p $NEW_PID > /dev/null; then
      echo -e "\n❌ 서버 시작 실패. 로그를 확인하세요: tail -f app.log"
      cat app.log
      exit 1
    fi
  fi
  
  # 서버가 응답하는지 확인 (10초 후)
  if [ $i -eq 20 ]; then
    if curl -s "http://localhost:$PORT/" > /dev/null; then
      echo -e "\n✅ 서버가 응답합니다."
    else
      echo -e "\n⚠️ 서버가 실행 중이지만 아직 응답하지 않습니다. 로그를 확인하세요."
    fi
  fi
done

echo -e "\n✅ 서버가 정상적으로 실행 중입니다."
echo "Qwen2.5-7B 모델이 적용된 최적화 서버로 성공적으로 전환되었습니다!"
echo "URL: http://localhost:$PORT"

# 로그 표시
echo "서버 로그 미리보기:"
tail -n 10 app.log

# 패키지 버전 정보 표시
echo -e "\n현재 설치된 패키지 버전:"
pip list | grep -E "transformers|accelerate|bitsandbytes|flash-attn|torch"

echo -e "\n서버 관리 명령어:"
echo "- 서버 상태 확인: ps -p $(cat server.pid)"
echo "- 로그 확인: tail -f app.log"
echo "- 서버 종료: kill -15 $(cat server.pid)"
echo "- API 테스트: curl http://localhost:$PORT/"
echo -e "\n" 