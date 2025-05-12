#!/bin/bash

# 변수 정의
LOG_FILE="./chatbot.log"
PID_FILE="./server.pid"
CURRENT_DIR=$(pwd)

echo "===== $(date) - 챗봇 서버 재시작 시작 =====" >> $LOG_FILE

# 기존 서버 프로세스 종료
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "기존 서버 프로세스(PID: $PID) 종료 중..." >> $LOG_FILE
        kill $PID
        sleep 5
        
        # 프로세스가 여전히 살아있는지 확인
        if ps -p $PID > /dev/null; then
            echo "정상 종료 실패, 강제 종료 시도..." >> $LOG_FILE
            kill -9 $PID
            sleep 2
        fi
    else
        echo "PID 파일이 존재하지만, 프로세스가 이미 종료됨" >> $LOG_FILE
    fi
    rm -f "$PID_FILE"
else
    echo "PID 파일이 없음, 서버가 실행 중이지 않은 것으로 판단" >> $LOG_FILE
fi

# 가상 환경 활성화
echo "가상 환경 활성화..." >> $LOG_FILE
source /home/test_code/test01/rag-chatbot/.venv/bin/activate

# 메모리 캐시 정리 (시스템 레벨)
echo "시스템 메모리 캐시 정리..." >> $LOG_FILE
sync
echo 3 > /proc/sys/vm/drop_caches

# 서버 시작
echo "새 서버 프로세스 시작..." >> $LOG_FILE
nohup uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --log-level debug >> $LOG_FILE 2>&1 &

# PID 저장
NEW_PID=$!
echo $NEW_PID > "$PID_FILE"
echo "새 서버 시작됨 (PID: $NEW_PID)" >> $LOG_FILE

echo "===== $(date) - 챗봇 서버 재시작 완료 =====" >> $LOG_FILE

# 로그 출력
echo "서버가 재시작되었습니다. 최신 로그 확인:"
tail -n 20 $LOG_FILE

echo "서버 재시작 완료 (PID: $NEW_PID)" 