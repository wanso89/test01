import React, { useRef, useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatContainer from './components/ChatContainer';
import { FiChevronLeft, FiChevronRight, FiSettings, FiRefreshCcw, FiTrash2, FiUser, FiBell, FiLock, FiUnlock } from 'react-icons/fi';

const SIDEBAR_WIDTH = 220;
const SIDEBAR_MIN = 60;
const SIDEBAR_MAX = 400;

function App() {
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_WIDTH);
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768); // md: 이상에서 기본적으로 열림
  const [model, setModel] = useState('GPT-4');
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [showSettingsDropdown, setShowSettingsDropdown] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [userName, setUserName] = useState('사용자');
  const [scrollLocked, setScrollLocked] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const isResizing = useRef(false);

  // 새로고침 후 activeConversationId 초기화
  useEffect(() => {
    const savedConversations = localStorage.getItem('conversations');
    const savedActiveId = localStorage.getItem('activeConversationId');
    let initialConvs = [];
    if (savedConversations) {
      try {
        initialConvs = JSON.parse(savedConversations);
      } catch (e) {
        console.error("Error parsing conversations from localStorage:", e);
        localStorage.removeItem('conversations');
      }
    }
    // 👇 conversations가 0개면 새 대화 생성
    if (initialConvs.length === 0) {
      const now = new Date();
      const newConv = {
        id: Date.now(),
        title: `대화 1`,
        timestamp: now.toLocaleString(),
        messages: [{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }]
      };
      initialConvs = [newConv];
      setConversations(initialConvs);
      setActiveConversationId(newConv.id);
      localStorage.setItem('conversations', JSON.stringify(initialConvs));
      localStorage.setItem('activeConversationId', JSON.stringify(newConv.id));
    } else {
      setConversations(initialConvs);
      let initialActiveId = null;
      if (savedActiveId) {
        try {
          initialActiveId = JSON.parse(savedActiveId);
          if (!initialConvs.some(conv => conv.id === initialActiveId)) {
            initialActiveId = initialConvs[initialConvs.length - 1].id;
          }
        } catch (e) {
          initialActiveId = initialConvs[initialConvs.length - 1].id;
        }
      } else {
        initialActiveId = initialConvs[initialConvs.length - 1].id;
      }
      setActiveConversationId(initialActiveId);
    }
  }, []);

  // activeConversationId 변경 시 localStorage에 저장
  useEffect(() => {
    if (activeConversationId) {
      localStorage.setItem('activeConversationId', JSON.stringify(activeConversationId));
    }
  }, [activeConversationId]);

  // conversations 변경 시 localStorage 동기화
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('conversations', JSON.stringify(conversations));
    }
  }, [conversations]);

  const createNewConversation = () => {
    const now = new Date();
    const newConv = {
      id: Date.now(),
      title: `대화 ${conversations.length + 1}`,
      timestamp: now.toLocaleString(),
      messages: [{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }]
    };
    const updatedConvs = conversations.length > 0 ? [...conversations, newConv] : [newConv];
    setConversations(updatedConvs);
    setActiveConversationId(newConv.id);
  };

  // 새 대화 생성
  const handleNewConversation = () => {
    createNewConversation();
  };

  // 대화 삭제
  const handleDeleteConversation = (id) => {
    setConversations(prev => {
      const updated = prev.filter(conv => conv.id !== id);
      let newActive = activeConversationId;
      if (id === activeConversationId) {
        newActive = updated.length > 0 ? updated[updated.length - 1].id : null;
        setActiveConversationId(newActive);
      }
      // 👇 conversations가 0개가 되면 새 대화 자동 생성
      if (updated.length === 0) {
        const now = new Date();
        const newConv = {
          id: Date.now(),
          title: '대화 1',
          timestamp: now.toLocaleString(),
          messages: [{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }]
        };
        setActiveConversationId(newConv.id);
        return [newConv];
      }
      return updated;
    });
  };

  // 대화 선택
  const handleSelectConversation = (id) => {
    setActiveConversationId(id);
  };

  // 메시지 업데이트
  const handleUpdateMessages = (updatedMessages) => {
    setConversations(prev =>
      prev.map(conv =>
        conv.id === activeConversationId
          ? { ...conv, messages: updatedMessages }
          : conv
      )
    );
  };

  // 현재 활성 대화의 메시지
  const currentMessages =
    conversations.find(conv => conv.id === activeConversationId)?.messages ||
    [{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }];

  // 반응형: 화면 크기 변경 시 사이드바 상태 업데이트
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      } else {
        setSidebarOpen(true);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 드래그 핸들러
  const handleMouseDown = (e) => {
    isResizing.current = true;
    document.body.style.cursor = 'col-resize';
    e.preventDefault();
  };
  const handleMouseMove = (e) => {
    if (isResizing.current && sidebarOpen) {
      let newWidth = Math.max(SIDEBAR_MIN, Math.min(SIDEBAR_MAX, e.clientX));
      setSidebarWidth(newWidth);
    }
  };
  const handleMouseUp = () => {
    if (isResizing.current) {
      isResizing.current = false;
      document.body.style.cursor = 'default';
    }
  };
  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  });

  // 토글(여닫이) 버튼
  const handleToggleSidebar = () => {
    setSidebarOpen((open) => !open);
  };

  // 모델 선택 드롭다운 토글
  const handleToggleModelDropdown = () => {
    setShowModelDropdown(!showModelDropdown);
    setShowSettingsDropdown(false);
  };

  // 설정 드롭다운 토글
  const handleToggleSettingsDropdown = () => {
    setShowSettingsDropdown(!showSettingsDropdown);
    setShowModelDropdown(false);
  };

  // 모델 선택
  const handleSelectModel = (selectedModel) => {
    setModel(selectedModel);
    setShowModelDropdown(false);
  };

  // Regenerate 기능 (예시)
  const handleRegenerate = () => {
    console.log('Regenerate last response');
    // 실제로는 마지막 메시지 재생성 로직 추가
  };

  // Clear Chat 기능 (예시)
  const handleClearChat = () => {
    console.log('Clear chat history');
    // 실제로는 대화 기록 초기화 로직 추가
  };

  // 알림 설정 토글
  const handleToggleNotifications = () => {
    setNotificationsEnabled(!notificationsEnabled);
  };

  // 스크롤 잠금 토글
  const handleToggleScrollLock = () => {
    setScrollLocked(!scrollLocked);
  };

  // 사용자 이름 변경 (예시)
  const handleChangeUserName = (newName) => {
    setUserName(newName);
    setShowSettingsDropdown(false);
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-[#f0f4f8] to-[#e3e9f0] dark:from-gray-900 dark:to-gray-800 transition-colors">
      {/* 사이드바 */}
      <div
        className={`
          relative h-full transition-all duration-300 ease-in-out
          ${sidebarOpen ? 'border-r border-gray-200 dark:border-gray-700' : ''}
          bg-white dark:bg-gray-800
          md:block hidden
        `}
        style={{
          width: sidebarOpen ? sidebarWidth : 0,
          minWidth: sidebarOpen ? SIDEBAR_MIN : 0,
          maxWidth: sidebarOpen ? SIDEBAR_MAX : 0,
          transform: sidebarOpen ? 'translateX(0)' : `translateX(-${sidebarWidth}px)`,
          overflow: 'hidden',
          zIndex: 20,
        }}
      >
        <Sidebar
          collapsed={!sidebarOpen}
          conversations={conversations}
          activeConversationId={activeConversationId}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          onSelectConversation={handleSelectConversation}
        />
        {/* 닫기 버튼 */}
        {sidebarOpen && (
          <button
            onClick={handleToggleSidebar}
            className="absolute -right-4 top-1/2 z-30 transform -translate-y-1/2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-full shadow p-1 hover:bg-blue-100 dark:hover:bg-blue-900 transition"
            style={{ width: 28, height: 28 }}
            title="사이드바 접기"
          >
            <FiChevronLeft size={20} />
          </button>
        )}
      </div>
      {/* 열기 버튼 (항상 화면 왼쪽에 고정) */}
      {!sidebarOpen && (
        <button
          onClick={handleToggleSidebar}
          className="fixed left-2 top-1/2 z-40 transform -translate-y-1/2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-full shadow p-1 hover:bg-blue-100 dark:hover:bg-blue-900 transition md:block hidden"
          style={{ width: 28, height: 28 }}
          title="사이드바 펼치기"
        >
          <FiChevronRight size={20} />
        </button>
      )}
      {/* 드래그 핸들 */}
      {sidebarOpen && (
        <div
          className="w-2 cursor-col-resize bg-gray-100 dark:bg-gray-700 hover:bg-blue-200 dark:hover:bg-blue-800 transition z-20 absolute right-0 top-0 h-full md:block hidden"
          onMouseDown={handleMouseDown}
          style={{ userSelect: 'none' }}
          title="사이드바 크기 조절"
        />
      )}
      {/* 본문 */}
      <main className="flex-1 flex flex-col border-l border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-900/80 transition-all duration-300 md:w-0 w-full">
        <header className="p-4 border-b bg-white dark:bg-gray-800 flex items-center justify-between sticky top-0 z-10 backdrop-blur-sm">
          <h1 className="text-2xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-indigo-600 font-['Pretendard','Noto Sans KR',sans-serif] drop-shadow">
            RAG 챗봇
          </h1>
          <div className="flex items-center gap-2">
            <button
              onClick={handleRegenerate}
              className="px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-100 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
              title="마지막 응답 재생성"
            >
              <FiRefreshCcw size={18} />
            </button>
            <button
              onClick={handleClearChat}
              className="px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-100 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
              title="대화 초기화"
            >
              <FiTrash2 size={18} />
            </button>
            <button
              onClick={handleToggleModelDropdown}
              className="px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-100 hover:bg-gray-300 dark:hover:bg-gray-600 transition relative"
              title="모델 선택"
            >
              {model}
              {showModelDropdown && (
                <div className="absolute top-full right-0 mt-2 w-48 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg z-20 animate-fade-in">
                  <button
                    onClick={() => handleSelectModel('GPT-4')}
                    className="block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition"
                  >
                    GPT-4
                  </button>
                  <button
                    onClick={() => handleSelectModel('GPT-3.5')}
                    className="block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition"
                  >
                    GPT-3.5
                  </button>
                  <button
                    onClick={() => handleSelectModel('Claude')}
                    className="block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition"
                  >
                    Claude
                  </button>
                </div>
              )}
            </button>
            <button
              onClick={handleToggleSettingsDropdown}
              className="px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-100 hover:bg-gray-300 dark:hover:bg-gray-600 transition relative"
              title="설정"
            >
              <FiSettings size={18} />
              {showSettingsDropdown && (
                <div className="absolute top-full right-0 mt-2 w-48 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg z-20 animate-fade-in">
                  <button
                    onClick={() => console.log('프로필 설정')}
                    className="block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition flex items-center"
                  >
                    <FiUser className="mr-2" size={16} /> 프로필 ({userName})
                  </button>
                  <button
                    onClick={handleToggleNotifications}
                    className="block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition flex items-center"
                  >
                    <FiBell className="mr-2" size={16} /> 알림 {notificationsEnabled ? '켜짐' : '꺼짐'}
                  </button>
                  <button
                    onClick={handleToggleScrollLock}
                    className="block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition flex items-center"
                  >
                    {scrollLocked ? <FiLock className="mr-2" size={16} /> : <FiUnlock className="mr-2" size={16} />}
                    스크롤 {scrollLocked ? '잠금' : '해제'}
                  </button>
                </div>
              )}
            </button>
            <button
              onClick={() => {
                document.documentElement.classList.toggle('dark');
              }}
              className="px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-100 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
              title="다크모드 토글"
            >
              🌙
            </button>
          </div>
        </header>
        <ChatContainer
          key={activeConversationId}
          scrollLocked={scrollLocked}
          activeConversationId={activeConversationId}
          messages={currentMessages}
          onUpdateMessages={handleUpdateMessages}
        />
      </main>
    </div>
  );
}

export default App;
