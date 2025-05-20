import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import ModernSidebar from './ModernSidebar';
import TopBar from './TopBar';

/**
 * Modern SaaS Dashboard 스타일의 전체 애플리케이션 레이아웃 컴포넌트
 * 사이드바와 메인 콘텐츠 영역을 관리합니다.
 */
const Layout = ({ 
  children,
  conversations = [],
  activeConversationId,
  onNewConversation,
  onDeleteConversation,
  onSelectConversation,
  onRenameConversation,
  onTogglePinConversation,
  onToggleTheme,
  isDarkMode,
  onToggleMode,
  currentMode = 'chat',
  onDeleteAllConversations,
  recentQueries = [],
  dbSchema = {},
  dashboardStats = {},
  userName = "사용자"
}) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  
  // 반응형 처리
  useEffect(() => {
    const checkIsMobile = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile && sidebarOpen) setSidebarOpen(false);
      else if (!mobile && !sidebarOpen) setSidebarOpen(true);
    };
    
    checkIsMobile();
    window.addEventListener('resize', checkIsMobile);
    return () => window.removeEventListener('resize', checkIsMobile);
  }, []);
  
  // 햄버거 메뉴 클릭 시 사이드바 토글
  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };
  
  // 사이드바 접기/펼치기 토글
  const toggleSidebarCollapse = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  return (
    <div className="flex h-screen bg-dashboard-background text-dashboard-text overflow-hidden">
      {/* 사이드바 - 모바일에서는 오버레이로 표시 */}
      <ModernSidebar 
        collapsed={sidebarCollapsed}
        setCollapsed={setSidebarCollapsed}
        isOpen={sidebarOpen}
        setIsOpen={setSidebarOpen}
        isMobile={isMobile}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onNewConversation={onNewConversation}
        onSelectConversation={onSelectConversation}
        onRenameConversation={onRenameConversation}
        onDeleteConversation={onDeleteConversation}
        onTogglePinConversation={onTogglePinConversation}
        onDeleteAllConversations={onDeleteAllConversations}
        currentMode={currentMode}
        recentQueries={recentQueries}
        dbSchema={dbSchema}
        dashboardStats={dashboardStats}
      />
      
      {/* 메인 콘텐츠 영역 */}
      <motion.main 
        className="flex flex-col flex-1 h-full overflow-hidden"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.2 }}
      >
        {/* 상단 내비게이션 바 */}
        <TopBar 
          toggleSidebar={toggleSidebar}
          sidebarOpen={sidebarOpen}
          currentMode={currentMode}
          setMode={onToggleMode}
          userName={userName}
          isDarkMode={isDarkMode}
        />
        
        {/* 메인 콘텐츠 */}
        <div className="flex-1 overflow-hidden">
          {children}
        </div>
      </motion.main>
      
      {/* 모바일 사이드바 오버레이 */}
      {isMobile && sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 z-10"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
};

export default Layout; 