"""
피드백 데이터 분석 및 검색 품질 개선을 위한 유틸리티 모듈
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
from datetime import datetime
import math

# 피드백 저장 디렉토리 설정
FEEDBACK_DIR = "app/feedback"
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "feedback.json")


class FeedbackAnalyzer:
    """
    사용자 피드백 데이터를 분석하여 검색 개선에 활용하는 클래스
    """
    
    def __init__(self):
        """초기화 함수"""
        self.feedbacks = []
        self.load_feedbacks()
        
        # 문서 품질 점수 캐시
        self.doc_quality_cache = {}
        self.cache_timestamp = None
        
    def load_feedbacks(self) -> List[Dict[str, Any]]:
        """피드백 데이터 로드"""
        if not os.path.exists(FEEDBACK_FILE):
            self.feedbacks = []
            return []
            
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                self.feedbacks = json.load(f)
            return self.feedbacks
        except Exception as e:
            print(f"피드백 데이터 로드 중 오류: {e}")
            self.feedbacks = []
            return []
            
    def get_feedback_stats(self) -> Dict[str, Any]:
        """피드백 통계 정보 반환"""
        if not self.feedbacks:
            return {
                "total": 0,
                "positive": 0,
                "negative": 0,
                "avg_rating": 0,
                "top_positive_reasons": [],
                "top_negative_reasons": []
            }
            
        # 기본 통계
        total = len(self.feedbacks)
        positive = sum(1 for f in self.feedbacks if f.get("feedbackType") == "up")
        negative = total - positive
        
        # 평균 평점
        ratings = [f.get("rating", 0) for f in self.feedbacks if "rating" in f]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # 이유 분석
        positive_reasons = []
        negative_reasons = []
        
        for f in self.feedbacks:
            if "reasons" in f and isinstance(f["reasons"], list):
                if f.get("feedbackType") == "up":
                    positive_reasons.extend(f["reasons"])
                else:
                    negative_reasons.extend(f["reasons"])
                    
        # 가장 많은 이유 상위 5개
        top_positive = Counter(positive_reasons).most_common(5)
        top_negative = Counter(negative_reasons).most_common(5)
        
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "positive_percentage": round((positive / total) * 100, 2) if total > 0 else 0,
            "avg_rating": round(avg_rating, 2),
            "top_positive_reasons": [{"reason": r, "count": c} for r, c in top_positive],
            "top_negative_reasons": [{"reason": r, "count": c} for r, c in top_negative]
        }
        
    def extract_source_paths(self) -> Dict[str, Dict[str, int]]:
        """
        피드백에서 언급된 소스 파일 경로와 피드백 유형을 추출
        """
        # 문서 경로 및 관련 피드백 카운트
        source_feedback = defaultdict(lambda: {"positive": 0, "negative": 0})
        
        for f in self.feedbacks:
            # content에서 소스 경로 추출 (형식: [문서 N])
            content = f.get("content", "")
            if not content:
                continue
                
            # 소스 표시 패턴 [문서 N] 찾기
            doc_refs = re.findall(r'\[문서\s+(\d+)\]', content)
            
            # 직접적인 경로 언급 찾기 (".pdf", ".docx" 등 확장자 포함)
            path_refs = re.findall(r'\b[\w\-\_\.\/]+\.(pdf|docx|xlsx|txt|md|py|java|cpp|json)\b', content, re.IGNORECASE)
            
            if doc_refs or path_refs:
                feedback_type = "positive" if f.get("feedbackType") == "up" else "negative"
                
                for path in path_refs:
                    source_feedback[path][feedback_type] += 1
                    
        return source_feedback
        
    def calculate_document_quality_scores(self) -> Dict[str, float]:
        """
        피드백을 기반으로 문서별 품질 점수 계산 (검색 순위 가중치에 활용)
        - 양수: 문서 관련성 증가
        - 음수: 문서 관련성 감소
        - 0: 중립 또는 데이터 없음
        """
        # 캐시된 점수가 있고 30분 이내면 재사용
        current_time = datetime.now()
        if (self.cache_timestamp and 
            (current_time - self.cache_timestamp).total_seconds() < 1800 and 
            self.doc_quality_cache):
            return self.doc_quality_cache
            
        # 소스 파일별 피드백 카운트 가져오기
        source_feedback = self.extract_source_paths()
        
        # 문서 품질 점수 계산
        doc_quality_scores = {}
        
        for source_path, counts in source_feedback.items():
            positive = counts["positive"]
            negative = counts["negative"]
            
            # 최소 피드백 수 조건 (노이즈 감소)
            total = positive + negative
            if total < 2:
                continue
                
            # 긍정/부정 비율 기반 품질 점수 계산
            if total > 0:
                # 신뢰도 가중 점수: 0.5(중립) 기준, ±0.5 범위
                # 피드백 수가 많을수록 신뢰도 증가
                confidence = min(1.0, math.log(total + 1) / 4)  # 로그 스케일 신뢰도
                raw_score = positive / total
                
                # 0.5를 중심으로 하는 품질 점수 (-0.5 ~ +0.5)
                quality_score = (raw_score - 0.5) * confidence
                
                # 최종 점수는 -0.5 ~ +0.5 범위
                doc_quality_scores[source_path] = round(quality_score, 3)
        
        # 캐시 업데이트
        self.doc_quality_cache = doc_quality_scores
        self.cache_timestamp = current_time
        
        return doc_quality_scores
        
    def extract_frequent_questions(self, min_count: int = 3) -> List[Dict[str, Any]]:
        """
        자주 묻는 질문 패턴 추출
        """
        # 질문 패턴 저장
        question_patterns = []
        
        # 피드백 내용에서 질문 추출
        for f in self.feedbacks:
            content = f.get("content", "")
            if not content:
                continue
                
            # 질문 형식 찾기 (물음표로 끝나는 문장)
            questions = re.findall(r'([^.!?]+\?)', content)
            question_patterns.extend(questions)
            
        # 자주 등장하는 질문 패턴 필터링
        common_questions = Counter(question_patterns).most_common()
        frequent_questions = [
            {"question": q, "count": c} 
            for q, c in common_questions 
            if c >= min_count
        ]
        
        return frequent_questions
    
    def get_query_boosting_factors(self) -> Dict[str, float]:
        """
        자주 묻는 질문에 기반한 쿼리 가중치 계수 반환
        """
        # 문서 품질 점수 계산
        doc_scores = self.calculate_document_quality_scores()
        
        # 키워드 기반 가중치 계수 (검색 개선에 활용)
        keyword_boost_factors = {}
        
        # 자주 묻는 질문에서 키워드 추출
        freq_questions = self.extract_frequent_questions(min_count=2)
        
        # 질문에서 핵심 키워드 추출
        keywords = []
        for q_data in freq_questions:
            question = q_data["question"]
            # 간단한 키워드 추출 (더 복잡한 NLP 기술 적용 가능)
            words = re.findall(r'\b[\w가-힣]{2,}\b', question)
            keywords.extend(words)
            
        # 키워드 빈도수 계산
        keyword_counts = Counter(keywords)
        
        # 자주 등장하는 키워드에 가중치 부여
        for keyword, count in keyword_counts.items():
            if count >= 2:  # 최소 2회 이상 등장
                # 로그 스케일 가중치 (키워드 빈도에 따라 0.1~0.5 범위)
                boost = min(0.5, 0.1 * math.log(count + 1, 2))
                keyword_boost_factors[keyword] = round(boost, 2)
                
        return {
            "document_quality_scores": doc_scores,
            "keyword_boost_factors": keyword_boost_factors
        }
        
    def analyze_feedback_trends(self, days: int = 30) -> Dict[str, Any]:
        """
        최근 N일간의 피드백 추세 분석
        """
        if not self.feedbacks:
            return {
                "daily_counts": [],
                "sentiment_trend": [],
                "rating_trend": []
            }
            
        # 날짜별 통계
        daily_stats = defaultdict(lambda: {"positive": 0, "negative": 0, "ratings": []})
        
        # 현재 시간
        now = datetime.now()
        
        # 피드백 데이터 분석
        for f in self.feedbacks:
            # 타임스탬프 파싱
            timestamp_str = f.get("timestamp", "")
            if not timestamp_str:
                continue
                
            try:
                # ISO 형식 타임스탬프 파싱
                feedback_time = datetime.fromisoformat(timestamp_str)
                
                # 지정된 기간(days) 내 피드백만 포함
                days_diff = (now - feedback_time).days
                if days_diff > days:
                    continue
                    
                # 날짜만 추출 (시간 제외)
                date_key = feedback_time.strftime("%Y-%m-%d")
                
                # 피드백 유형 및 평점 추가
                feedback_type = "positive" if f.get("feedbackType") == "up" else "negative"
                daily_stats[date_key][feedback_type] += 1
                
                # 평점이 있으면 추가
                if "rating" in f:
                    daily_stats[date_key]["ratings"].append(f["rating"])
            except Exception as e:
                print(f"피드백 타임스탬프 파싱 오류: {e}")
                continue
                
        # 결과 데이터 구성
        sorted_dates = sorted(daily_stats.keys())
        
        daily_counts = []
        sentiment_trend = []
        rating_trend = []
        
        for date in sorted_dates:
            stats = daily_stats[date]
            pos = stats["positive"]
            neg = stats["negative"]
            total = pos + neg
            
            daily_counts.append({
                "date": date,
                "positive": pos,
                "negative": neg,
                "total": total
            })
            
            # 감성 비율 (긍정 비율)
            if total > 0:
                sentiment = round((pos / total) * 100, 1)
            else:
                sentiment = 50  # 중립
                
            sentiment_trend.append({
                "date": date,
                "sentiment": sentiment
            })
            
            # 평균 평점
            ratings = stats["ratings"]
            avg_rating = round(sum(ratings) / len(ratings), 1) if ratings else 0
            
            rating_trend.append({
                "date": date,
                "avg_rating": avg_rating
            })
            
        return {
            "daily_counts": daily_counts,
            "sentiment_trend": sentiment_trend,
            "rating_trend": rating_trend
        }


class SearchQualityOptimizer:
    """
    피드백 분석을 기반으로 검색 품질을 최적화하는 클래스
    """
    
    def __init__(self):
        self.feedback_analyzer = FeedbackAnalyzer()
        
    def get_query_optimizations(self, query: str) -> Dict[str, Any]:
        """
        사용자 쿼리에 대한 최적화 정보 반환
        """
        # 피드백 기반 가중치 요소 가져오기
        boost_factors = self.feedback_analyzer.get_query_boosting_factors()
        
        # 문서 품질 점수
        doc_scores = boost_factors.get("document_quality_scores", {})
        
        # 키워드 가중치
        keyword_boosts = boost_factors.get("keyword_boost_factors", {})
        
        # 쿼리 내 키워드가 가중치 목록에 있는지 확인
        query_words = re.findall(r'\b[\w가-힣]{2,}\b', query.lower())
        matched_keywords = {}
        
        for word in query_words:
            if word in keyword_boosts:
                matched_keywords[word] = keyword_boosts[word]
                
        # 최적화 지침 생성
        optimizations = {
            "document_boosts": doc_scores,
            "keyword_matches": matched_keywords,
            "optimization_applied": len(matched_keywords) > 0 or len(doc_scores) > 0
        }
        
        return optimizations
        
    def apply_optimizations_to_query(self, query: str, es_query: Dict) -> Dict:
        """
        Elasticsearch 쿼리에 최적화 적용
        """
        # 최적화 정보 가져오기
        optimizations = self.get_query_optimizations(query)
        
        # 최적화 적용이 필요 없으면 원본 쿼리 반환
        if not optimizations.get("optimization_applied", False):
            return es_query
            
        # 문서 품질 점수 적용 (function_score 쿼리 사용)
        doc_boosts = optimizations.get("document_boosts", {})
        keyword_matches = optimizations.get("keyword_matches", {})
        
        # 원본 쿼리를 function_score 쿼리로 변환
        # (이미 function_score가 있으면 그대로 사용)
        if "function_score" not in es_query["query"]:
            original_query = es_query["query"]
            es_query["query"] = {
                "function_score": {
                    "query": original_query,
                    "functions": [],
                    "score_mode": "sum",
                    "boost_mode": "multiply"
                }
            }
            
        # 문서 품질 점수에 따른 가중치 함수 추가
        if doc_boosts:
            for doc_path, score in doc_boosts.items():
                # 문서 경로에 대한 가중치 함수 추가
                score_func = {
                    "filter": {
                        "term": {
                            "source": doc_path
                        }
                    },
                    "weight": 1 + score  # -0.5 ~ +0.5 범위의 점수를 0.5 ~ 1.5 범위로 변환
                }
                es_query["query"]["function_score"]["functions"].append(score_func)
                
        # 키워드 가중치 적용
        if keyword_matches:
            for keyword, boost in keyword_matches.items():
                # 매칭된 키워드에 대한 가중치 함수 추가
                keyword_func = {
                    "filter": {
                        "match": {
                            "text": keyword
                        }
                    },
                    "weight": 1 + boost  # 0.1 ~ 0.5 범위의 가중치를 1.1 ~ 1.5 범위로 변환
                }
                es_query["query"]["function_score"]["functions"].append(keyword_func)
                
        return es_query 