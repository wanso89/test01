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
  const [userId, setUserId] = useState("user1"); // 임시 사용자 ID, 실제로는 인증 기반 ID 사용
  const [theme, setTheme] = useState('light'); // 테마 상태 (light/dark)
  const [defaultCategory, setDefaultCategory] = useState('메뉴얼'); // 기본 카테고리 상태
  const [showStatsDashboard, setShowStatsDashboard] = useState(false);
  const [statsData, setStatsData] = useState([]);

  // 초기 대화 목록 로드 (로컬 스토리지 또는 백엔드)
  useEffect(() => {
    const savedConversations = localStorage.getItem('conversations');
    const savedActiveId = localStorage.getItem('activeConversationId');
    let initialConvs = [];
    if (savedConversations) {
      try {
        initialConvs = JSON.parse(savedConversations);
        initialConvs = initialConvs.map(conv => ({
          ...conv,
          messages: conv.messages.map(msg => ({
            ...msg,
            sources: msg.sources || []
          }))
        }))
      } catch (e) {
        console.error("Error parsing conversations from localStorage:", e);
        localStorage.removeItem('conversations');
      }
    }
    // conversations가 0개면 새 대화 생성
    if (initialConvs.length === 0) {
      const now = new Date();
      const newConv = {
        id: Date.now().toString(),
        title: `대화 1`,
        timestamp: now.toLocaleString(),
        messages: [{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?', sources: [] }],
        pinned: false
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

  // 초기 설정 로드 (로컬 스토리지 또는 백엔드)
useEffect(() => {
  const savedTheme = localStorage.getItem('theme');
  const savedCategory = localStorage.getItem('defaultCategory');
  if (savedTheme) {
    setTheme(savedTheme);
    document.documentElement.classList.toggle('dark', savedTheme === 'dark');
  }
  if (savedCategory) {
    setDefaultCategory(savedCategory);
  }
  // 백엔드에서 설정 불러오기
  loadUserSettingsFromBackend(userId);
}, [userId]);

// 테마 변경 함수
const handleChangeTheme = (newTheme) => {
  setTheme(newTheme);
  localStorage.setItem('theme', newTheme);
  document.documentElement.classList.toggle('dark', newTheme === 'dark');
  // 백엔드에 설정 저장
  saveUserSettingsToBackend(userId, { theme: newTheme, defaultCategory });
};

// 기본 카테고리 변경 함수
const handleChangeDefaultCategory = (newCategory) => {
  setDefaultCategory(newCategory);
  localStorage.setItem('defaultCategory', newCategory);
  // 백엔드에 설정 저장
  saveUserSettingsToBackend(userId, { theme, defaultCategory: newCategory });
};

// 백엔드에 사용자 설정 저장
const saveUserSettingsToBackend = async (userId, settings) => {
  try {
    const response = await fetch('http://172.10.2.70:8000/api/settings/save', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userId: userId,
        settings: settings
      })
    });
    if (!response.ok) {
      throw new Error(`사용자 설정 저장 실패: ${response.status} ${response.statusText}`);
    }
    console.log("사용자 설정이 백엔드에 저장되었습니다.");
  } catch (err) {
    console.error("사용자 설정 저장 중 오류 발생:", err);
  }
};

// 백엔드에서 사용자 설정 불러오기
const loadUserSettingsFromBackend = async (userId) => {
  try {
    const response = await fetch('http://172.10.2.70:8000/api/settings/load', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userId: userId
      })
    });
    if (!response.ok) {
      throw new Error(`사용자 설정 불러오기 실패: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    if (data.status === "success" && data.settings) {
      if (data.settings.theme) {
        setTheme(data.settings.theme);
        localStorage.setItem('theme', data.settings.theme);
        document.documentElement.classList.toggle('dark', data.settings.theme === 'dark');
      }
      if (data.settings.defaultCategory) {
        setDefaultCategory(data.settings.defaultCategory);
        localStorage.setItem('defaultCategory', data.settings.defaultCategory);
      }
      console.log("사용자 설정이 백엔드에서 불러와졌습니다.");
    } else if (data.status === "not_found") {
      console.log("사용자 설정이 없습니다. 기본 설정을 사용합니다.");
    }
  } catch (err) {
    console.error("사용자 설정 불러오기 중 오류 발생:", err);
  }
};


  // activeConversationId 변경 시 localStorage에 저장
  useEffect(() => {
    if (activeConversationId) {
      localStorage.setItem('activeConversationId', JSON.stringify(activeConversationId));
    }
  }, [activeConversationId]);

  // conversations 변경 시 localStorage 및 백엔드 동기화
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('conversations', JSON.stringify(conversations));
      // 백엔드에도 저장 (활성 대화만 저장 예시)
      if (activeConversationId) {
        const activeConv = conversations.find(conv => conv.id === activeConversationId);
        if (activeConv) {
          saveConversationToBackend(userId, activeConversationId, activeConv.messages);
        }
      }
    }
  }, [conversations, activeConversationId]);

  // 새 대화 생성
  const handleNewConversation = () => {
    const now = new Date();
    const newConv = {
      id: Date.now().toString(),
      title: `대화 ${conversations.length + 1}`,
      timestamp: now.toLocaleString(),
      messages: [{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?', sources: [] }],
      pinned: false
    };
    setConversations(prev => [...prev, newConv]);
    setActiveConversationId(newConv.id);
  };

  // 대화 선택
  const handleSelectConversation = (id) => {
    setActiveConversationId(id);
    // 백엔드에서 대화 불러오기 (필요 시)
    loadConversationFromBackend(userId, id);
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
      // conversations가 0개가 되면 새 대화 자동 생성
      if (updated.length === 0) {
        const now = new Date();
        const newConv = {
          id: Date.now().toString(),
          title: '대화 1',
          timestamp: now.toLocaleString(),
          messages: [{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?', sources: [] }],
          pinned: false
        };
        setActiveConversationId(newConv.id);
        return [newConv];
      }
      return updated;
    });
  };

  // 대화 제목 변경
  const handleRenameConversation = (id, newTitle) => {
    setConversations(prev =>
      prev.map(conv =>
        conv.id === id ? { ...conv, title: newTitle } : conv
      )
    );
  };

  // 대화 즐겨찾기 토글
  const handleTogglePinConversation = (id) => {
    setConversations(prev =>
      prev.map(conv =>
        conv.id === id ? { ...conv, pinned: !conv.pinned } : conv
      )
    );
  };

  // 사용자 행동 기록 함수
const logUserAction = async (action, details = {}) => {
  try {
    const response = await fetch('http://172.10.2.70:8000/api/stats/save', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userId: userId,
        action: action,
        details: details
      })
    });
    if (!response.ok) {
      throw new Error(`통계 저장 실패: ${response.status} ${response.statusText}`);
    }
    console.log(`통계 저장됨: ${action}`);
  } catch (err) {
    console.error("통계 저장 중 오류 발생:", err);
  }
};

// 통계 조회 함수
const fetchStats = async () => {
  try {
    const response = await fetch('http://172.10.2.70:8000/api/stats/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userId: "", // 전체 사용자 조회
        startDate: "", // 모든 기간
        endDate: "" // 모든 기간
      })
    });
    if (!response.ok) {
      throw new Error(`통계 조회 실패: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    if (data.status === "success") {
      setStatsData(data.stats);
      setShowStatsDashboard(true);
      console.log("통계 조회 완료");
    }
  } catch (err) {
    console.error("통계 조회 중 오류 발생:", err);
    alert("통계 조회에 실패했습니다. 서버 연결을 확인해주세요.");
  }
};


  // 메시지 업데이트
  const handleUpdateMessages = (updatedMessages) => {
    setConversations(prev =>
      prev.map(conv =>
        conv.id === activeConversationId ? { ...conv, messages: updatedMessages } : conv
      )
    );
    // 사용자가 질문을 보낸 경우 통계 기록
    if (updatedMessages.length > 0 && updatedMessages[updatedMessages.length - 1].role === 'user') {
      logUserAction('question', { category: defaultCategory, timestamp: new Date().toISOString() });
    }
  };

  // 백엔드에 대화 저장
  const saveConversationToBackend = async (userId, conversationId, messages) => {
    try {
      const response = await fetch('http://172.10.2.70:8000/api/conversations/save', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: userId,
          conversationId: conversationId,
          messages: messages.map(msg => ({
            role: msg.role,
            content: msg.content,
            sources: msg.sources || []
          }))
        })
      });
      if (!response.ok) {
        throw new Error(`대화 저장 실패: ${response.status} ${response.statusText}`);
      }
      console.log("대화가 백엔드에 저장되었습니다.");
    } catch (err) {
      console.error("대화 저장 중 오류 발생:", err);
    }
  };

  // 백엔드에서 대화 불러오기
  const loadConversationFromBackend = async (userId, conversationId) => {
    try {
      const response = await fetch('http://172.10.2.70:8000/api/conversations/load', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: userId,
          conversationId: conversationId
        })
      });
      if (!response.ok) {
        throw new Error(`대화 불러오기 실패: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      if (data.status === "success" && data.conversation) {
        setConversations(prev =>
          prev.map(conv =>
            conv.id === conversationId ? { ...conv, messages: data.conversation.messages } : conv
          )
        );
        console.log("대화가 백엔드에서 불러와졌습니다.");
      }
    } catch (err) {
      console.error("대화 불러오기 중 오류 발생:", err);
    }
  };

  // 현재 활성 대화의 메시지
  const currentMessages =
    conversations.find(conv => conv.id === activeConversationId)?.messages ||
    [{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?', sources: [] }];

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
    <div className="flex h-screen bg-gradient-to-br from-[#f0f4f8] to-[#e3e9f0] dark:from-gray-900 dark:to-gray-800 transition-colors duration-300">
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
          onRenameConversation={handleRenameConversation}
          onTogglePinConversation={handleTogglePinConversation}
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
                  <div
                    onClick={() => console.log('프로필 설정')}
                    className="block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition flex items-center cursor-pointer"
                  >
                    <FiUser className="mr-2" size={16} /> 프로필 ({userName})
                  </div>
                  <div
                    onClick={handleToggleNotifications}
                    className="block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition flex items-center cursor-pointer"
                  >
                    <FiBell className="mr-2" size={16} /> 알림 {notificationsEnabled ? '켜짐' : '꺼짐'}
                  </div>
                  <div
                    onClick={handleToggleScrollLock}
                    className="block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition flex items-center cursor-pointer"
                  >
                    {scrollLocked ? <FiLock className="mr-2" size={16} /> : <FiUnlock className="mr-2" size={16} />}
                    스크롤 {scrollLocked ? '잠금' : '해제'}
                  </div>
                  <div className="border-t border-gray-200 dark:border-gray-700 my-1"></div>
                  <div className="px-3 py-1 text-sm text-gray-500 dark:text-gray-400">테마</div>
                  <div
                    onClick={() => handleChangeTheme('light')}
                    className={`block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition cursor-pointer ${theme === 'light' ? 'bg-blue-100 dark:bg-blue-800' : ''}`}
                  >
                    라이트 모드
                  </div>
                  <div
                    onClick={() => handleChangeTheme('dark')}
                    className={`block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition cursor-pointer ${theme === 'dark' ? 'bg-blue-100 dark:bg-blue-800' : ''}`}
                  >
                    다크 모드
                  </div>
                  <div className="border-t border-gray-200 dark:border-gray-700 my-1"></div>
                  <div className="px-3 py-1 text-sm text-gray-500 dark:text-gray-400">기본 카테고리</div>
                  <div
                    onClick={() => handleChangeDefaultCategory('메뉴얼')}
                    className={`block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition cursor-pointer ${defaultCategory === '메뉴얼' ? 'bg-blue-100 dark:bg-blue-800' : ''}`}
                  >
                    메뉴얼
                  </div>
                  <div
                    onClick={() => handleChangeDefaultCategory('기술문서')}
                    className={`block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition cursor-pointer ${defaultCategory === '기술문서' ? 'bg-blue-100 dark:bg-blue-800' : ''}`}
                  >
                    기술문서
                  </div>
                  <div
                    onClick={() => handleChangeDefaultCategory('기타')}
                    className={`block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition cursor-pointer ${defaultCategory === '기타' ? 'bg-blue-100 dark:bg-blue-800' : ''}`}
                  >
                    기타
                  </div>
                  <div className="border-t border-gray-200 dark:border-gray-700 my-1"></div>
                  <div
                    onClick={fetchStats}
                    className="block w-full text-left px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition cursor-pointer"
                  >
                    통계 대시보드
                  </div>
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
              {showStatsDashboard && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center min-h-screen z-50 p-4 animate-fade-in">
                  <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-4xl w-full max-h-[80vh] overflow-y-auto shadow-2xl animate-slide-up">
                    <h3 className="text-lg font-bold text-gray-800 dark:text-white mb-2">통계 대시보드</h3>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                      사용자 행동 및 챗봇 사용 통계
                    </div>
                    <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded border border-gray-200 dark:border-gray-600 text-gray-800 dark:text-gray-200">
                      {statsData.length > 0 ? (
                        <div>
                          <h4 className="text-md font-semibold mb-2">최근 활동 ({statsData.length}건)</h4>
                          <ul className="space-y-2">
                            {statsData.slice(-10).reverse().map((stat, idx) => (
                              <li key={idx} className="border-b border-gray-200 dark:border-gray-600 pb-2">
                                <span className="font-medium">{stat.action}</span> - 사용자: {stat.userId}
                                <div className="text-xs text-gray-500 dark:text-gray-400">
                                  {stat.timestamp} | 세부: {JSON.stringify(stat.details)}
                                </div>
                              </li>
                            ))}
                          </ul>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">최근 10건 표시 중...</p>
                        </div>
                      ) : (
                        <p>통계 데이터가 없습니다.</p>
                      )}
                    </div>
                    <button
                      onClick={() => setShowStatsDashboard(false)}
                      className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
                    >
                      닫기
                    </button>
                  </div>
                </div>
              )}
              </div>
                );
              }

export default App;
