import React, { useState, useEffect, useCallback } from "react";
import { FiChevronLeft, FiExternalLink, FiMaximize, FiMinimize, FiRefreshCw, FiInfo, FiFilter, 
  FiDownload, FiAlertCircle, FiMessageCircle, FiDatabase, FiBarChart2, FiClock, FiActivity,
  FiCalendar, FiPackage, FiUsers, FiSearch, FiPieChart, FiTrendingUp, FiGrid, FiList } from "react-icons/fi";
import UsageStatsCard from "./UsageStatsCard";
import QueryAnalyticsCard from "./QueryAnalyticsCard";
import TopQueriesCard from "./TopQueriesCard";
import ResponseTimeCard from "./ResponseTimeCard";
import SystemStatusCard from "./SystemStatusCard";
import DataSourcesCard from "./DataSourcesCard";

// API 기본 URL 상수 정의
const API_BASE_URL = 'http://172.10.2.70:8000';

// 백업용 모의 데이터 (API 실패 시 사용)
const MOCK_DATA = {
  usageStats: {
    totalQueries: 1245,
    totalChats: 873,
    activeUsers: 48,
    averageQueriesPerDay: 67,
    queryCountByDate: [
      { date: "1월", value: 150 },
      { date: "2월", value: 220 },
      { date: "3월", value: 310 },
      { date: "4월", value: 275 },
      { date: "5월", value: 380 },
      { date: "6월", value: 320 },
      { date: "7월", value: 420 },
      { date: "8월", value: 390 },
      { date: "9월", value: 450 },
      { date: "10월", value: 560 },
      { date: "11월", value: 510 },
      { date: "12월", value: 670 }
    ]
  },
  queryAnalytics: {
    successRate: 94.2,
    averageQueryTime: 1.8,
    queryDistribution: [
      { category: "SQL 질의", value: 58 },
      { category: "문서 검색", value: 42 }
    ],
    queryTimeDistribution: [
      { range: "0-1초", value: 240 },
      { range: "1-2초", value: 420 },
      { range: "2-3초", value: 340 },
      { range: "3-5초", value: 180 },
      { range: "5초+", value: 65 }
    ]
  },
  topQueries: [
    { text: "매출 데이터 분석", count: 87, category: "SQL" },
    { text: "제품 재고 현황", count: 72, category: "SQL" },
    { text: "프로젝트 문서 검색", count: 68, category: "문서" },
    { text: "인사정보 조회", count: 65, category: "SQL" },
    { text: "구매내역 조회", count: 58, category: "SQL" },
    { text: "신규 직원 매뉴얼", count: 52, category: "문서" },
    { text: "고객 피드백 분석", count: 51, category: "문서" },
    { text: "시스템 오류 해결방법", count: 47, category: "문서" }
  ],
  responseTimes: {
    average: 1.8,
    trend: [
      { date: "월", value: 2.3 },
      { date: "화", value: 2.1 },
      { date: "수", value: 1.9 },
      { date: "목", value: 1.7 },
      { date: "금", value: 1.8 },
      { date: "토", value: 1.6 },
      { date: "일", value: 1.5 }
    ],
    sqlProcessing: 0.5,
    vectorSearch: 0.7,
    llmGeneration: 0.6
  },
  systemStatus: {
    cpu: 32,
    memory: 64,
    storage: 48,
    uptime: "99.98%",
    lastRestart: "2023-11-01 03:15 AM",
    activeConnections: 26
  },
  dataSources: {
    totalDocuments: 1250,
    totalSizeMB: 856,
    types: [
      { type: "PDF", count: 520 },
      { type: "DOCX", count: 320 },
      { type: "TXT", count: 180 },
      { type: "PPT", count: 150 },
      { type: "XLS", count: 80 }
    ],
    recentlyAdded: [
      { name: "2023년 3분기 실적보고서.pdf", size: "4.2MB", date: "2023-11-15" },
      { name: "신규 프로젝트 제안서.docx", size: "2.8MB", date: "2023-11-14" },
      { name: "인사평가 지침.pdf", size: "1.5MB", date: "2023-11-10" },
      { name: "제품 매뉴얼 v2.1.pdf", size: "8.7MB", date: "2023-11-08" }
    ]
  }
};

function Dashboard({ setMode }) {
  const [dashboardData, setDashboardData] = useState(MOCK_DATA);
  const [isLoading, setIsLoading] = useState(false);
  const [timeRange, setTimeRange] = useState("month");
  const [expandedCard, setExpandedCard] = useState(null);
  const [error, setError] = useState(null);
  const [apiAvailable, setApiAvailable] = useState(true);
  const [layout, setLayout] = useState("grid"); // 'grid' 또는 'list'
  const [filterCategory, setFilterCategory] = useState("all");

  // 날짜 범위 계산 함수
  const getDateRange = (range) => {
    const now = new Date();
    const endDate = now.toISOString().split('T')[0]; // YYYY-MM-DD
    let startDate;

    switch (range) {
      case 'day':
        startDate = endDate; // 오늘
        break;
      case 'week':
        // 7일 전
        const weekAgo = new Date(now);
        weekAgo.setDate(now.getDate() - 7);
        startDate = weekAgo.toISOString().split('T')[0];
        break;
      case 'month':
        // 한 달 전
        const monthAgo = new Date(now);
        monthAgo.setMonth(now.getMonth() - 1);
        startDate = monthAgo.toISOString().split('T')[0];
        break;
      case 'quarter':
        // 3개월 전
        const quarterAgo = new Date(now);
        quarterAgo.setMonth(now.getMonth() - 3);
        startDate = quarterAgo.toISOString().split('T')[0];
        break;
      case 'year':
        // 1년 전
        const yearAgo = new Date(now);
        yearAgo.setFullYear(now.getFullYear() - 1);
        startDate = yearAgo.toISOString().split('T')[0];
        break;
      default:
        // 기본값 (한 달)
        const defaultPeriod = new Date(now);
        defaultPeriod.setMonth(now.getMonth() - 1);
        startDate = defaultPeriod.toISOString().split('T')[0];
    }

    return { startDate, endDate };
  };

  // 대시보드 데이터 로드 함수
  const loadDashboardData = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const { startDate, endDate } = getDateRange(timeRange);
      
      const response = await fetch(`${API_BASE_URL}/api/dashboard/stats`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          startDate,
          endDate,
          timeRange
        })
      });
      
      if (!response.ok) {
        throw new Error(`API 요청 실패: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success') {
        setDashboardData(data.data);
        setApiAvailable(true);
      } else {
        console.warn('API 응답 오류:', data.message || '알 수 없는 오류');
        setError(data.message || '데이터를 불러오는 중 오류가 발생했습니다.');
        // 폴백: 모의 데이터 사용
        setDashboardData(MOCK_DATA);
      }
    } catch (err) {
      console.error('대시보드 데이터 로드 중 오류:', err);
      setError('데이터를 불러오는 중 오류가 발생했습니다. 다시 시도해 주세요.');
      setApiAvailable(false);
      // 폴백: 모의 데이터 사용
      setDashboardData(MOCK_DATA);
    } finally {
      setIsLoading(false);
    }
  }, [timeRange]);

  // 시스템 상태 데이터 로드 함수
  const loadSystemStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/system/status`);
      
      if (!response.ok) {
        throw new Error(`API 요청 실패: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success' && data.data) {
        // 시스템 상태 데이터만 업데이트 (안전하게 처리)
        setDashboardData(prevData => ({
          ...prevData,
          systemStatus: {
            ...MOCK_DATA.systemStatus, // 기본값 설정
            ...data.data // 받아온 데이터로 덮어쓰기
          }
        }));
      } else {
        console.warn('시스템 상태 응답에 유효한 데이터가 없습니다:', data);
      }
    } catch (err) {
      console.warn('시스템 상태 데이터 로드 중 오류:', err);
      // 시스템 상태 로드 실패해도 다른 데이터는 유지
    }
  }, []);

  // 데이터 소스 정보 로드 함수
  const loadDataSources = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/datasources/stats`);
      
      if (!response.ok) {
        throw new Error(`API 요청 실패: ${response.status}`);
      }
      
      const data = await response.json();
      
      if (data.status === 'success' && data.data) {
        // 데이터 소스 정보만 업데이트 (안전하게 처리)
        setDashboardData(prevData => {
          // 기존 데이터가 없을 경우를 대비한 안전한 업데이트
          const safeUpdate = {
            ...prevData,
            dataSources: {
              // 먼저 목업 데이터로 기본값 설정
              ...MOCK_DATA.dataSources,
              // 그 다음 API 응답으로 덮어쓰기
              ...data.data
            }
          };
          
          return safeUpdate;
        });
      } else {
        console.warn('데이터 소스 응답에 유효한 데이터가 없습니다:', data);
      }
    } catch (err) {
      console.warn('데이터 소스 정보 로드 중 오류:', err);
      // 데이터 소스 로드 실패해도 다른 데이터는 유지
    }
  }, []);

  // 데이터 초기 로드 및 주기적 갱신
  useEffect(() => {
    // 초기 로드
    loadDashboardData();
    loadSystemStatus();
    loadDataSources();
    
    // 주기적 갱신 (15초마다)
    const interval = setInterval(() => {
      loadSystemStatus(); // 시스템 상태는 자주 갱신
    }, 15000);
    
    // 정리 함수
    return () => clearInterval(interval);
  }, [loadDashboardData, loadSystemStatus, loadDataSources, timeRange]);

  // 날짜 범위 변경 시 데이터 다시 로드
  useEffect(() => {
    loadDashboardData();
  }, [timeRange, loadDashboardData]);

  // 새로고침 함수 (수동 새로고침)
  const refreshData = () => {
    loadDashboardData();
    loadSystemStatus();
    loadDataSources();
  };

  // 카드 확장/축소 토글 함수
  const toggleCardExpansion = (cardId) => {
    if (expandedCard === cardId) {
      setExpandedCard(null);
    } else {
      setExpandedCard(cardId);
    }
  };
  
  // 모드 전환 토글 컴포넌트
  const ModeToggleSwitch = () => {
    // 모드 전환 핸들러
    const handleToggleMode = (newMode) => (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      // 같은 모드 클릭 시 무시
      if (newMode === 'dashboard') return;
      
      try {
        // 모드 전환 함수 호출
        if (typeof setMode === 'function') {
          setMode(newMode);
          console.log(`${newMode} 모드로 전환 함수 호출`);
        
          // 버튼 효과
          const button = e.currentTarget;
          button.classList.add('scale-95');
          setTimeout(() => {
            button.classList.remove('scale-95');
          }, 200);
        }
      } catch (err) {
        console.error('모드 전환 중 오류 발생:', err);
      }
    };

    return (
      <div className="fixed right-6 top-1/2 transform -translate-y-1/2 z-20">
        <div className="bg-gray-800/90 backdrop-blur-md rounded-full p-2 shadow-lg border border-gray-700/50 flex flex-col gap-3">
          {/* 배경 효과 - 활성화된 모드에 따라 움직임 */}
          <div className="absolute inset-x-1.5 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500/20 to-indigo-600/20 filter blur-sm transition-all duration-300 ease-in-out pointer-events-none" 
               style={{ 
                 top: '5.4rem',
                 opacity: 0.7
               }}>
          </div>
          
          {/* 챗봇 모드 버튼 */}
          <button 
            onClick={handleToggleMode('chat')}
            className="relative transition-all duration-300 w-8 h-8 rounded-full flex items-center justify-center bg-gray-800/80 text-gray-400 hover:text-gray-200 hover:bg-gray-700/60"
            title="챗봇 모드로 전환"
          >
            <FiMessageCircle size={14} className="transition-all duration-300" />
          </button>
          
          {/* SQL 모드 버튼 */}
          <button 
            onClick={handleToggleMode('sql')}
            className="relative transition-all duration-300 w-8 h-8 rounded-full flex items-center justify-center bg-gray-800/80 text-gray-400 hover:text-gray-200 hover:bg-gray-700/60"
            title="SQL 질의 모드로 전환"
          >
            <FiDatabase size={14} className="transition-all duration-300" />
          </button>
          
          {/* 대시보드 모드 버튼 */}
          <button 
            className="relative transition-all duration-300 w-8 h-8 rounded-full flex items-center justify-center bg-gradient-to-br from-emerald-500/80 to-teal-600/80 text-white shadow-md shadow-emerald-500/20"
            title="대시보드 모드로 전환"
          >
            {/* 활성화 효과 - 고리 애니메이션 */}
            <div className="absolute inset-0 rounded-full border border-emerald-400/30 animate-ping opacity-30"></div>
            <div className="absolute inset-0 rounded-full bg-gradient-to-br from-emerald-500/5 to-teal-600/5 animate-pulse"></div>
            
            <FiBarChart2 size={14} className="transition-all duration-300" />
          </button>
        </div>
      </div>
    );
  };

  // 카드 렌더링 함수
  const renderCard = (id, title, children, width = "full", height = "auto", icon = null) => {
    const isExpanded = expandedCard === id;
    
    return (
      <div 
        className={`${isExpanded ? 'col-span-full row-span-2' : width === 'full' ? 'col-span-full' : width === 'half' ? 'col-span-1' : 'col-span-1'} 
                   transition-all duration-300 bg-gray-800/50 backdrop-blur-sm border border-gray-700/50 rounded-xl shadow-lg overflow-hidden`}
        style={{ 
          height: isExpanded ? '70vh' : height === 'auto' ? 'auto' : height === 'tall' ? '380px' : '280px'
        }}
      >
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700/50 bg-gray-800/50">
          <div className="flex items-center gap-2">
            {icon && <span className="text-emerald-400">{icon}</span>}
            <h3 className="font-medium text-white">{title}</h3>
          </div>
          <div className="flex items-center gap-1">
            <button 
              onClick={() => toggleCardExpansion(id)}
              className="p-1.5 hover:bg-gray-700/50 rounded-md text-gray-400 hover:text-white transition-colors" 
              title={isExpanded ? "축소" : "확장"}
            >
              {isExpanded ? <FiMinimize size={14} /> : <FiMaximize size={14} />}
            </button>
          </div>
        </div>
        
        <div className={`${isExpanded ? 'p-6' : 'p-4'} h-[calc(100%-48px)] overflow-auto`}>
          {children}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* 헤더 바 */}
      <div className="h-16 flex items-center justify-between px-6 bg-gray-900 border-b border-gray-800 shadow-sm z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-md">
            <FiBarChart2 size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-teal-500 bg-clip-text text-transparent">
              대시보드
            </h1>
            <p className="text-xs text-gray-400 mt-0.5">시스템 현황 및 사용 통계 요약</p>
          </div>
        </div>
        
        {/* 헤더 컨트롤 영역 */}
        <div className="flex items-center gap-3">
          {/* 기간 필터 */}
          <div className="relative flex-shrink-0">
            <select 
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="appearance-none bg-gray-800 text-gray-200 text-sm rounded-lg px-3 py-2 pr-8 border border-gray-700 focus:outline-none focus:border-emerald-500"
            >
              <option value="day">오늘</option>
              <option value="week">최근 7일</option>
              <option value="month">최근 30일</option>
              <option value="quarter">최근 90일</option>
              <option value="year">최근 1년</option>
            </select>
            <FiCalendar className="absolute top-1/2 right-3 transform -translate-y-1/2 text-gray-400 pointer-events-none" size={14} />
          </div>
          
          {/* 레이아웃 전환 */}
          <div className="flex bg-gray-800 rounded-lg p-0.5 border border-gray-700">
            <button 
              className={`p-1.5 rounded ${layout === 'grid' ? 'bg-emerald-600 text-white' : 'text-gray-400 hover:text-white'}`}
              onClick={() => setLayout('grid')}
              title="그리드 보기"
            >
              <FiGrid size={14} />
            </button>
            <button 
              className={`p-1.5 rounded ${layout === 'list' ? 'bg-emerald-600 text-white' : 'text-gray-400 hover:text-white'}`}
              onClick={() => setLayout('list')}
              title="리스트 보기"
            >
              <FiList size={14} />
            </button>
          </div>
          
          {/* 새로고침 버튼 */}
          <button 
            onClick={refreshData} 
            disabled={isLoading}
            className={`p-2 rounded-lg transition-colors ${isLoading 
              ? 'bg-gray-700 text-gray-500 cursor-not-allowed' 
              : 'bg-gray-800 text-gray-300 hover:bg-emerald-600/40 hover:text-white hover:border-emerald-500/50'
            } border border-gray-700`}
            title="데이터 새로고침"
          >
            <FiRefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* 대시보드 콘텐츠 */}
      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
        {/* 오류 메시지 */}
        {error && (
          <div className="mb-4 px-4 py-3 bg-red-900/20 border border-red-800/30 rounded-lg text-red-300 flex items-center gap-2">
            <FiAlertCircle size={16} className="flex-shrink-0" />
            <p className="text-sm">{error}</p>
            <button 
              onClick={refreshData}
              className="ml-auto px-2 py-1 bg-red-800/30 hover:bg-red-700/40 rounded text-xs text-red-200 transition-colors"
            >
              다시 시도
            </button>
          </div>
        )}
        
        {/* 주요 지표 요약 (Quick Stats) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {/* 총 쿼리 수 */}
          <div className="bg-gradient-to-br from-blue-900/40 to-indigo-900/40 rounded-xl p-4 border border-blue-800/30 flex items-center">
            <div className="w-12 h-12 rounded-lg bg-blue-500/20 flex items-center justify-center mr-4">
              <FiSearch size={24} className="text-blue-400" />
            </div>
            <div>
              <p className="text-gray-400 text-sm">총 쿼리 수</p>
              <h3 className="text-2xl font-bold text-white mt-1">
                {dashboardData.usageStats?.totalQueries?.toLocaleString() || 0}
              </h3>
            </div>
          </div>
          
          {/* 총 대화 수 */}
          <div className="bg-gradient-to-br from-purple-900/40 to-indigo-900/40 rounded-xl p-4 border border-purple-800/30 flex items-center">
            <div className="w-12 h-12 rounded-lg bg-purple-500/20 flex items-center justify-center mr-4">
              <FiMessageCircle size={24} className="text-purple-400" />
            </div>
            <div>
              <p className="text-gray-400 text-sm">총 대화 수</p>
              <h3 className="text-2xl font-bold text-white mt-1">
                {dashboardData.usageStats?.totalChats?.toLocaleString() || 0}
              </h3>
            </div>
          </div>
          
          {/* 활성 사용자 */}
          <div className="bg-gradient-to-br from-emerald-900/40 to-teal-900/40 rounded-xl p-4 border border-emerald-800/30 flex items-center">
            <div className="w-12 h-12 rounded-lg bg-emerald-500/20 flex items-center justify-center mr-4">
              <FiUsers size={24} className="text-emerald-400" />
            </div>
            <div>
              <p className="text-gray-400 text-sm">활성 사용자</p>
              <h3 className="text-2xl font-bold text-white mt-1">
                {dashboardData.usageStats?.activeUsers?.toLocaleString() || 0}
              </h3>
            </div>
          </div>
          
          {/* 평균 응답 시간 */}
          <div className="bg-gradient-to-br from-amber-900/40 to-orange-900/40 rounded-xl p-4 border border-amber-800/30 flex items-center">
            <div className="w-12 h-12 rounded-lg bg-amber-500/20 flex items-center justify-center mr-4">
              <FiClock size={24} className="text-amber-400" />
            </div>
            <div>
              <p className="text-gray-400 text-sm">평균 응답 시간</p>
              <h3 className="text-2xl font-bold text-white mt-1">
                {dashboardData.responseTimes?.average?.toFixed(1) || 0}
                <span className="text-sm text-gray-400 ml-1">초</span>
              </h3>
            </div>
          </div>
        </div>
        
        {/* 메인 대시보드 그리드 */}
        <div className={`${layout === 'grid' ? 'grid grid-cols-1 md:grid-cols-2' : 'flex flex-col'} gap-4`}>
          {/* 사용량 통계 */}
          {renderCard(
            'usage-stats', 
            '사용량 통계', 
            <UsageStatsCard data={dashboardData.usageStats} isExpanded={expandedCard === 'usage-stats'} />,
            'half',
            'tall',
            <FiTrendingUp size={16} />
          )}
          
          {/* 쿼리 분석 */}
          {renderCard(
            'query-analytics', 
            '쿼리 분석', 
            <QueryAnalyticsCard data={dashboardData.queryAnalytics} isExpanded={expandedCard === 'query-analytics'} />,
            'half',
            'tall',
            <FiPieChart size={16} />
          )}
          
          {/* 인기 쿼리 */}
          {renderCard(
            'top-queries', 
            '인기 쿼리', 
            <TopQueriesCard data={dashboardData.topQueries} isExpanded={expandedCard === 'top-queries'} />,
            'half',
            'auto',
            <FiSearch size={16} />
          )}
          
          {/* 응답 시간 */}
          {renderCard(
            'response-time', 
            '응답 시간', 
            <ResponseTimeCard data={dashboardData.responseTimes} isExpanded={expandedCard === 'response-time'} />,
            'half',
            'auto',
            <FiActivity size={16} />
          )}
          
          {/* 시스템 상태 */}
          {renderCard(
            'system-status', 
            '시스템 상태', 
            <SystemStatusCard data={dashboardData.systemStatus} isExpanded={expandedCard === 'system-status'} />,
            'half',
            'auto',
            <FiActivity size={16} />
          )}
          
          {/* 데이터 소스 */}
          {renderCard(
            'data-sources', 
            '데이터 소스', 
            <DataSourcesCard data={dashboardData.dataSources} isExpanded={expandedCard === 'data-sources'} />,
            'half',
            'auto',
            <FiPackage size={16} />
          )}
        </div>
      </div>
      
      {/* 모드 전환 스위치 */}
      <ModeToggleSwitch />
    </div>
  );
}

export default Dashboard; 