#!/bin/bash

echo "💡 Defog (오픈소스 로컬 모드) 설치를 시작합니다..."

# 가상환경 활성화 여부 확인
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "❗ 가상환경이 활성화되어 있지 않습니다. 먼저 가상환경을 활성화해주세요."
    echo "예: source .venv/bin/activate"
    exit 1
fi

# Defog 저장소 clone
echo "📥 GitHub에서 Defog 저장소를 클론합니다..."
git clone https://github.com/defog-ai/defog.git defog_local

cd defog_local || exit

# 로컬 설치
echo "📦 Defog 패키지를 로컬 설치합니다..."
pip install -e .

# 환경 변수 안내
echo ""
echo "🔐 Defog는 로컬 모드로 실행되므로 API 키는 필요하지 않습니다."
echo "   (DEFOG_API_KEY 없이 사용 가능)"
echo ""

# 종속성 설치
echo "⚙️ 추가 종속성 설치 중 (sqlalchemy, python-dotenv 등)..."
pip install python-dotenv sqlalchemy

# 테스트 코드 안내
echo ""
echo "✅ 설치 완료! 예시 테스트:"
echo "python -c "from defog import Defog; print(Defog().generate_sql(question='20대 사용자 수', schema='CREATE TABLE users (user_id INT, age INT)', dialect='mysql'))""

