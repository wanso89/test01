#!/bin/bash

echo "Defog SQL 변환 모듈 설치 시작..."

# 가상환경 활성화 여부 확인
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "가상환경이 활성화되어 있지 않습니다. 가상환경을 활성화해주세요."
    echo "예: source .venv/bin/activate"
    exit 1
fi

# Defog 라이브러리 설치
echo "Defog 라이브러리 설치 중..."
pip install defog

# 환경 변수 설정
echo "API 키를 설정합니다..."
read -p "Defog API 키를 입력하세요 (없으면 Enter): " API_KEY

if [ -z "$API_KEY" ]; then
    echo "API 키가 입력되지 않았습니다. 무료 로컬 모델을 사용합니다."
    # 환경 변수 설정 (.env 파일이 있는지 확인 후 추가)
    if [ -f "backend/.env" ]; then
        # .env 파일에 DEFOG_API_KEY가 이미 있는지 확인
        if grep -q "DEFOG_API_KEY" backend/.env; then
            # 값 업데이트
            sed -i 's/^DEFOG_API_KEY=.*/DEFOG_API_KEY=/' backend/.env
        else
            # 새로 추가
            echo "DEFOG_API_KEY=" >> backend/.env
        fi
    else
        # .env 파일 생성
        echo "DEFOG_API_KEY=" > backend/.env
        echo ".env 파일을 생성했습니다."
    fi
else
    # API 키 저장
    if [ -f "backend/.env" ]; then
        # .env 파일에 DEFOG_API_KEY가 이미 있는지 확인
        if grep -q "DEFOG_API_KEY" backend/.env; then
            # 값 업데이트
            sed -i "s/^DEFOG_API_KEY=.*/DEFOG_API_KEY=$API_KEY/" backend/.env
        else
            # 새로 추가
            echo "DEFOG_API_KEY=$API_KEY" >> backend/.env
        fi
    else
        # .env 파일 생성
        echo "DEFOG_API_KEY=$API_KEY" > backend/.env
        echo ".env 파일을 생성했습니다."
    fi
    echo "API 키가 .env 파일에 저장되었습니다."
fi

# OpenAI 기본 키가 있는지 확인 (로컬 실행에 필요)
if grep -q "OPENAI_API_KEY" backend/.env; then
    echo "OpenAI API 키가 이미 설정되어 있습니다."
else
    read -p "OpenAI API 키를 입력하세요 (로컬 실행에 필요, 없으면 Enter): " OPENAI_KEY
    if [ -n "$OPENAI_KEY" ]; then
        echo "OPENAI_API_KEY=$OPENAI_KEY" >> backend/.env
        echo "OpenAI API 키가 .env 파일에 저장되었습니다."
    fi
fi

# 필요한 추가 패키지 설치
echo "필요한 추가 패키지를 설치합니다..."
pip install python-dotenv sqlalchemy

echo "Defog 설치 완료!"
echo "이제 SQL 변환 모듈을 사용할 수 있습니다."
echo "테스트하려면 다음 명령어를 실행하세요:"
echo "python -c \"from app.utils.sql_utils_defog import test_db_connection; print(test_db_connection())\""

# 실행 권한 부여
chmod +x backend/install_defog.sh

echo "스크립트에 실행 권한을 부여했습니다." 