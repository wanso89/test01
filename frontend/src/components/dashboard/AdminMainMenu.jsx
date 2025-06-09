import React from "react";
import { 
  FiDatabase, FiBarChart2, FiUsers, FiSettings, FiFileText, 
  FiMessageCircle, FiActivity, FiPieChart, FiServer, FiSearch,
  FiTrendingUp, FiGrid, FiList, FiPackage, FiUploadCloud, FiAlertCircle,
  FiCpu, FiHardDrive, FiRefreshCw, FiClock, FiCalendar
} from "react-icons/fi";
import { Link } from "react-router-dom";

const AdminMainMenu = ({ onNavigate }) => {
  // 메뉴 아이템 정의
  const menuItems = [
    {
      id: 'dashboard',
      title: '대시보드',
      description: '시스템 현황 및 사용 통계 요약',
      icon: <FiPieChart size={24} />,
      color: 'from-emerald-500 to-teal-600',
      textColor: 'from-emerald-400 to-teal-500'
    },
    {
      id: 'conversations',
      title: '대화 분석',
      description: '사용자 대화 내역 및 패턴 분석',
      icon: <FiMessageCircle size={24} />,
      color: 'from-blue-500 to-indigo-600',
      textColor: 'from-blue-400 to-indigo-500'
    },
    {
      id: 'datasources',
      title: '데이터 소스',
      description: '임베딩된 문서 및 데이터 관리',
      icon: <FiDatabase size={24} />,
      color: 'from-indigo-500 to-purple-600',
      textColor: 'from-indigo-400 to-purple-500'
    },
    {
      id: 'users',
      title: '사용자 관리',
      description: '사용자 계정 및 권한 설정',
      icon: <FiUsers size={24} />,
      color: 'from-purple-500 to-pink-600',
      textColor: 'from-purple-400 to-pink-500'
    },
    {
      id: 'upload',
      title: '문서 업로드',
      description: '새 문서 업로드 및 임베딩',
      icon: <FiUploadCloud size={24} />,
      color: 'from-orange-500 to-red-600',
      textColor: 'from-orange-400 to-red-500'
    },
    {
      id: 'system',
      title: '시스템 상태',
      description: '서버 및 리소스 모니터링',
      icon: <FiServer size={24} />,
      color: 'from-red-500 to-pink-600',
      textColor: 'from-red-400 to-pink-500'
    },
    {
      id: 'analytics',
      title: '사용 통계',
      description: '사용량 및 성능 분석',
      icon: <FiBarChart2 size={24} />,
      color: 'from-yellow-500 to-amber-600',
      textColor: 'from-yellow-400 to-amber-500'
    },
    {
      id: 'settings',
      title: '시스템 설정',
      description: '시스템 환경 설정 및 구성',
      icon: <FiSettings size={24} />,
      color: 'from-teal-500 to-cyan-600',
      textColor: 'from-teal-400 to-cyan-500'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* 헤더 */}
      <div className="h-16 flex items-center justify-between px-6 bg-gray-900 border-b border-gray-800 shadow-sm z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-md">
            <FiGrid size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent">
              관리자 메인메뉴
            </h1>
            <p className="text-xs text-gray-400 mt-0.5">시스템 관리 기능 모음</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <button
            onClick={() => window.location.reload()}
            className="p-2 rounded-lg bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
            title="새로고침"
          >
            <FiRefreshCw size={16} />
          </button>
        </div>
      </div>
      
      {/* 메인 콘텐츠 */}
      <div className="p-6">
        <div className="max-w-7xl mx-auto">
          {/* 환영 메시지 */}
          <div className="mb-8 bg-gradient-to-br from-gray-800/50 to-gray-900/50 p-6 rounded-xl border border-gray-700/50 shadow-lg">
            <h2 className="text-2xl font-bold text-white mb-2">관리자 대시보드에 오신 것을 환영합니다</h2>
            <p className="text-gray-300">RAG 챗봇 시스템의 모든 관리 기능에 접근할 수 있습니다. 아래 메뉴에서 원하는 기능을 선택하세요.</p>
            <div className="mt-4 flex items-center text-sm text-gray-400">
              <FiClock className="mr-2" size={14} />
              <span>현재 시간: {new Date().toLocaleString('ko-KR')}</span>
            </div>
          </div>
          
          {/* 메뉴 그리드 */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {menuItems.map((item) => (
              <div 
                key={item.id}
                onClick={() => onNavigate(item.id)}
                className="bg-gray-800/60 hover:bg-gray-800/90 border border-gray-700/50 rounded-xl p-5 cursor-pointer transition-all duration-300 transform hover:-translate-y-1 hover:shadow-lg"
              >
                <div className="flex items-start">
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${item.color} flex items-center justify-center shadow-md`}>
                    {item.icon}
                  </div>
                  <div className="ml-4 flex-1">
                    <h3 className={`text-lg font-semibold bg-gradient-to-r ${item.textColor} bg-clip-text text-transparent`}>
                      {item.title}
                    </h3>
                    <p className="text-sm text-gray-400 mt-1">{item.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
          
          {/* 시스템 상태 요약 */}
          <div className="mt-10 bg-gray-800/40 rounded-xl p-6 border border-gray-700/50">
            <h3 className="text-lg font-medium text-white mb-4 flex items-center">
              <FiActivity className="mr-2 text-indigo-400" size={18} />
              시스템 상태 요약
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gray-800/70 rounded-lg p-4 border border-gray-700/50">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-400">CPU 사용률</div>
                  <div className="p-1.5 bg-blue-900/30 rounded-md">
                    <FiCpu size={14} className="text-blue-400" />
                  </div>
                </div>
                <div className="mt-2 text-xl font-semibold text-white">32%</div>
                <div className="w-full bg-gray-700/50 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div className="bg-blue-500 h-full rounded-full" style={{ width: '32%' }}></div>
                </div>
              </div>
              
              <div className="bg-gray-800/70 rounded-lg p-4 border border-gray-700/50">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-400">메모리 사용률</div>
                  <div className="p-1.5 bg-purple-900/30 rounded-md">
                    <FiHardDrive size={14} className="text-purple-400" />
                  </div>
                </div>
                <div className="mt-2 text-xl font-semibold text-white">64%</div>
                <div className="w-full bg-gray-700/50 h-1.5 rounded-full mt-2 overflow-hidden">
                  <div className="bg-purple-500 h-full rounded-full" style={{ width: '64%' }}></div>
                </div>
              </div>
              
              <div className="bg-gray-800/70 rounded-lg p-4 border border-gray-700/50">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-400">총 문서 수</div>
                  <div className="p-1.5 bg-emerald-900/30 rounded-md">
                    <FiFileText size={14} className="text-emerald-400" />
                  </div>
                </div>
                <div className="mt-2 text-xl font-semibold text-white">1,250</div>
                <div className="text-xs text-emerald-400/80 mt-2 flex items-center">
                  <FiTrendingUp size={12} className="mr-1" />
                  <span>지난 주 대비 12% 증가</span>
                </div>
              </div>
              
              <div className="bg-gray-800/70 rounded-lg p-4 border border-gray-700/50">
                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-400">오늘 질의 수</div>
                  <div className="p-1.5 bg-amber-900/30 rounded-md">
                    <FiMessageCircle size={14} className="text-amber-400" />
                  </div>
                </div>
                <div className="mt-2 text-xl font-semibold text-white">487</div>
                <div className="text-xs text-amber-400/80 mt-2 flex items-center">
                  <FiTrendingUp size={12} className="mr-1" />
                  <span>어제 대비 8% 증가</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* 푸터 */}
      <div className="mt-auto py-4 px-6 border-t border-gray-800 text-center text-xs text-gray-500">
        <p>© {new Date().getFullYear()} RAG 챗봇 관리자 시스템 | 버전 1.0.0</p>
      </div>
    </div>
  );
};

export default AdminMainMenu; 