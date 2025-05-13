# Defog SQL 통합 가이드

이 프로젝트에는 자연어 질문을 SQL 쿼리로 변환하는 기능이 통합되어 있습니다. 최근에 기존 Vanna AI에서 Defog로 전환하여 더 정확하고 안정적인 SQL 쿼리 생성이 가능해졌습니다.

## 주요 기능

- **자연어 질문을 SQL로 변환**: 질문을 입력하면 데이터베이스 스키마를 기반으로 SQL 쿼리를 생성합니다.
- **여러 방식 지원**: Defog(기본), Vanna AI, 규칙 기반 접근법을 통합적으로 사용합니다.
- **한국어 질문 지원**: 한국어로 된 질문도 효과적으로 처리합니다.

## 설치 방법

### 1. Defog 설치

아래 스크립트를 실행하여 Defog를 설치하고 설정합니다:

```bash
cd /home/test_code/test01/rag-chatbot
source .venv/bin/activate
./backend/install_defog.sh
```

### 2. 환경 변수 설정

자체 API 키를 사용하려면 `.env` 파일에 아래 내용을 추가하세요:

```
DEFOG_API_KEY=your_api_key_here
```

## 사용 방법

API 호출 예시:

```javascript
// SQL 쿼리 생성 API 호출
const response = await fetch('/api/sql-query', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: '리뷰가 없는 제품은 어떤 것들이야?'
  }),
});

const result = await response.json();
console.log(result.sql);  // 생성된 SQL 쿼리
console.log(result.results);  // 실행 결과
```

## Defog vs Vanna AI 비교

| 항목 | Defog | Vanna AI |
|------|-------|----------|
| 정확도 | 높음 | 중간 |
| 한국어 지원 | 좋음 (전처리 기능 포함) | 제한적 |
| 커스터마이징 | 가능 | 제한적 |
| 설치 용이성 | 쉬움 | 쉬움 |
| 오류 처리 | 우수 | 중간 |

## 팁과 문제 해결

- 질문에 테이블 이름을 명시적으로 포함하면 더 정확한 결과를 얻을 수 있습니다.
- 여러 테이블의 관계가 필요한 복잡한 쿼리도 잘 처리합니다.
- API 키가 없는 경우에도 로컬 모델로 작동합니다.

## 지원하는 질문 유형

- "리뷰가 없는 제품은 어떤 것들이야?"
- "제품별 평균 리뷰 점수를 보여줘"
- "가장 자주 주문된 제품 TOP 3를 알려줘"
- "20대 사용자는 몇 명인가요?" 