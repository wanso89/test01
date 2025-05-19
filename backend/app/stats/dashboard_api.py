from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
import logging

router = APIRouter()

class TimeRange(str, Enum):
    day = "day"
    week = "week"
    month = "month"
    quarter = "quarter"
    year = "year"

class DashboardStatsRequest(BaseModel):
    startDate: str
    endDate: str
    timeRange: TimeRange

@router.post("/api/dashboard/stats")
async def get_dashboard_stats(request: DashboardStatsRequest):
    """대시보드 통계 데이터 API - 모의 데이터 반환"""
    try:
        # 실제 데이터베이스 연동 대신 모의 데이터 반환
        mock_data = {
            "usageStats": {
                "totalQueries": random.randint(800, 1500),
                "totalChats": random.randint(500, 900),
                "activeUsers": random.randint(30, 60),
                "averageQueriesPerDay": random.randint(40, 80),
                "queryCountByDate": [
                    {"date": "1월", "value": random.randint(100, 200)},
                    {"date": "2월", "value": random.randint(150, 250)},
                    {"date": "3월", "value": random.randint(200, 350)},
                    {"date": "4월", "value": random.randint(200, 300)},
                    {"date": "5월", "value": random.randint(300, 400)},
                    {"date": "6월", "value": random.randint(300, 350)},
                    {"date": "7월", "value": random.randint(350, 450)},
                    {"date": "8월", "value": random.randint(350, 450)},
                    {"date": "9월", "value": random.randint(400, 500)},
                    {"date": "10월", "value": random.randint(450, 600)},
                    {"date": "11월", "value": random.randint(450, 550)},
                    {"date": "12월", "value": random.randint(600, 700)}
                ]
            },
            "queryAnalytics": {
                "successRate": round(random.uniform(88, 98), 1),
                "averageQueryTime": round(random.uniform(1.5, 2.5), 1),
                "queryDistribution": [
                    {"category": "SQL 질의", "value": random.randint(40, 60)},
                    {"category": "문서 검색", "value": random.randint(40, 60)}
                ],
                "queryTimeDistribution": [
                    {"range": "0-1초", "value": random.randint(200, 300)},
                    {"range": "1-2초", "value": random.randint(350, 450)},
                    {"range": "2-3초", "value": random.randint(300, 400)},
                    {"range": "3-5초", "value": random.randint(150, 200)},
                    {"range": "5초+", "value": random.randint(50, 80)}
                ]
            },
            "topQueries": [
                {"text": "매출 데이터 분석", "count": random.randint(70, 100), "category": "SQL"},
                {"text": "제품 재고 현황", "count": random.randint(65, 85), "category": "SQL"},
                {"text": "프로젝트 문서 검색", "count": random.randint(60, 75), "category": "문서"},
                {"text": "인사정보 조회", "count": random.randint(55, 70), "category": "SQL"},
                {"text": "구매내역 조회", "count": random.randint(50, 65), "category": "SQL"},
                {"text": "신규 직원 매뉴얼", "count": random.randint(45, 60), "category": "문서"},
                {"text": "고객 피드백 분석", "count": random.randint(40, 55), "category": "문서"},
                {"text": "시스템 오류 해결방법", "count": random.randint(40, 50), "category": "문서"}
            ],
            "responseTimes": {
                "average": round(random.uniform(1.5, 2.0), 1),
                "trend": [
                    {"date": "월", "value": round(random.uniform(2.0, 2.5), 1)},
                    {"date": "화", "value": round(random.uniform(1.9, 2.3), 1)},
                    {"date": "수", "value": round(random.uniform(1.7, 2.1), 1)},
                    {"date": "목", "value": round(random.uniform(1.6, 1.9), 1)},
                    {"date": "금", "value": round(random.uniform(1.7, 2.0), 1)},
                    {"date": "토", "value": round(random.uniform(1.5, 1.8), 1)},
                    {"date": "일", "value": round(random.uniform(1.4, 1.7), 1)}
                ],
                "sqlProcessing": round(random.uniform(0.4, 0.6), 1),
                "vectorSearch": round(random.uniform(0.6, 0.8), 1),
                "llmGeneration": round(random.uniform(0.5, 0.7), 1)
            },
            "systemStatus": {
                "cpu": random.randint(25, 40),
                "memory": random.randint(50, 70),
                "storage": random.randint(40, 60),
                "uptime": "99.98%",
                "lastRestart": "2023-11-01 03:15 AM",
                "activeConnections": random.randint(20, 35)
            },
            "dataSources": {
                "totalDocuments": random.randint(1000, 1500),
                "totalSizeMB": random.randint(700, 900),
                "types": [
                    {"type": "PDF", "count": random.randint(450, 550)},
                    {"type": "DOCX", "count": random.randint(300, 350)},
                    {"type": "TXT", "count": random.randint(150, 200)},
                    {"type": "PPT", "count": random.randint(130, 170)},
                    {"type": "XLS", "count": random.randint(70, 90)}
                ],
                "recentlyAdded": [
                    {"name": "2023년 3분기 실적보고서.pdf", "size": "4.2MB", "date": "2023-11-15"},
                    {"name": "신규 프로젝트 제안서.docx", "size": "2.8MB", "date": "2023-11-14"},
                    {"name": "인사평가 지침.pdf", "size": "1.5MB", "date": "2023-11-10"},
                    {"name": "제품 매뉴얼 v2.1.pdf", "size": "8.7MB", "date": "2023-11-08"}
                ]
            }
        }
        return {"status": "success", "data": mock_data}
    except Exception as e:
        logging.error(f"대시보드 통계 데이터 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"대시보드 통계 데이터를 불러오는데 실패했습니다: {str(e)}")

@router.get("/api/system/status")
async def get_system_status():
    """시스템 상태 정보 API - 서버 리소스 현황 등 반환"""
    try:
        # 실제 시스템 모니터링 대신 모의 데이터 반환
        system_status = {
            "cpu": random.randint(25, 50),
            "memory": random.randint(50, 75),
            "storage": random.randint(40, 60),
            "uptime": "99.98%",
            "lastRestart": (datetime.now() - timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d %H:%M"),
            "activeConnections": random.randint(15, 40)
        }
        return {"status": "success", "data": system_status}
    except Exception as e:
        logging.error(f"시스템 상태 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"시스템 상태 정보를 불러오는데 실패했습니다: {str(e)}")

@router.get("/api/datasources/stats")
async def get_datasources_stats():
    """데이터 소스 통계 API - 인덱싱된 문서 정보 등 반환"""
    try:
        # 실제 데이터 소스 통계 대신 모의 데이터 반환
        data_sources = {
            "totalDocuments": random.randint(800, 1500),
            "totalSizeMB": random.randint(600, 1000),
            "types": [
                {"type": "PDF", "count": random.randint(400, 600)},
                {"type": "DOCX", "count": random.randint(250, 400)},
                {"type": "TXT", "count": random.randint(150, 250)},
                {"type": "PPT", "count": random.randint(100, 200)},
                {"type": "XLS", "count": random.randint(50, 100)}
            ],
            "recentlyAdded": [
                {"name": f"{datetime.now().year}년 {random.choice(['1', '2', '3', '4'])}분기 보고서.pdf", "size": f"{random.randint(2, 8)}.{random.randint(1, 9)}MB", "date": (datetime.now() - timedelta(days=random.randint(0, 5))).strftime("%Y-%m-%d")},
                {"name": "신규 프로젝트 제안서.docx", "size": f"{random.randint(1, 5)}.{random.randint(1, 9)}MB", "date": (datetime.now() - timedelta(days=random.randint(5, 10))).strftime("%Y-%m-%d")},
                {"name": "운영 매뉴얼 v3.0.pdf", "size": f"{random.randint(3, 10)}.{random.randint(1, 9)}MB", "date": (datetime.now() - timedelta(days=random.randint(10, 15))).strftime("%Y-%m-%d")},
                {"name": "시스템 구성도.pptx", "size": f"{random.randint(5, 15)}.{random.randint(1, 9)}MB", "date": (datetime.now() - timedelta(days=random.randint(15, 20))).strftime("%Y-%m-%d")}
            ]
        }
        return {"status": "success", "data": data_sources}
    except Exception as e:
        logging.error(f"데이터 소스 통계 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"데이터 소스 통계를 불러오는데 실패했습니다: {str(e)}") 