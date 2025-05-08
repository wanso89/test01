import React, { useRef, useState, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import ChatContainer from "./components/ChatContainer";
import {
  FiChevronLeft,
  FiChevronRight,
  FiSettings,
  FiRefreshCcw,
  FiTrash2,
  FiUser,
  FiBell,
  FiLock,
  FiUnlock,
  FiSearch,
  FiSun,
  FiMoon,
  FiPlus,
  FiMenu,
  FiMessageSquare,
  FiLayout,
  FiChevronDown,
  FiAlertCircle,
  FiX
} from "react-icons/fi";
import { FiServer } from "react-icons/fi";

const SIDEBAR_WIDTH = 280;
const SIDEBAR_MIN = 60;
const SIDEBAR_MAX = 400;

function App() {
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_WIDTH);
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768); // md: 이상에서 기본적으로 열림
  const [showSettingsDropdown, setShowSettingsDropdown] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [userName, setUserName] = useState("사용자");
  const [scrollLocked, setScrollLocked] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const isResizing = useRef(false);
  const [userId, setUserId] = useState("user1"); // 임시 사용자 ID, 실제로는 인증 기반 ID 사용
  const [theme, setTheme] = useState("light"); // 테마 상태 (light/dark)
  const [defaultCategory, setDefaultCategory] = useState("메뉴얼"); // 기본 카테고리 상태
  const [showStatsDashboard, setShowStatsDashboard] = useState(false);
  const [statsData, setStatsData] = useState([]);
  const [saveError, setSaveError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentMessages, setCurrentMessages] = useState([
    {
      role: "assistant",
      content: "안녕하세요! 무엇을 도와드릴까요?",
      sources: [],
    },
  ]);
  const [filteredMessages, setFilteredMessages] = useState(currentMessages);

  // 초기 대화 목록 로드 (로컬 스토리지 또는 백엔드)
  useEffect(() => {
    const savedConversations = localStorage.getItem("conversations");
    const savedActiveId = localStorage.getItem("activeConversationId");
    let initialConvs = [];
    if (savedConversations) {
      try {
        initialConvs = JSON.parse(savedConversations);
        initialConvs = initialConvs.map((conv) => ({
          ...conv,
          messages: conv.messages.map((msg) => ({
            ...msg,
            sources: msg.sources || [],
          })),
        }));
      } catch (e) {
        console.error("Error parsing conversations from localStorage:", e);
        localStorage.removeItem("conversations");
      }
    }
    // conversations가 0개면 새 대화 생성
    if (initialConvs.length === 0) {
      const now = new Date();
      const newConv = {
        id: Date.now().toString(),
        title: `대화 1`,
        timestamp: now.toLocaleString(),
        messages: [
          {
            role: "assistant",
            content: "안녕하세요! 무엇을 도와드릴까요?",
            sources: [],
          },
        ],
        pinned: false,
      };
      initialConvs = [newConv];
      setConversations(initialConvs);
      setActiveConversationId(newConv.id);
      localStorage.setItem("conversations", JSON.stringify(initialConvs));
      localStorage.setItem("activeConversationId", JSON.stringify(newConv.id));
    } else {
      setConversations(initialConvs);
      let initialActiveId = null;
      if (savedActiveId) {
        try {
          initialActiveId = JSON.parse(savedActiveId);
          if (!initialConvs.some((conv) => conv.id === initialActiveId)) {
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
    const savedTheme = localStorage.getItem("theme");
    const savedCategory = localStorage.getItem("defaultCategory");
    if (savedTheme) {
      setTheme(savedTheme);
      document.documentElement.classList.toggle("dark", savedTheme === "dark");
    }
    if (savedCategory) {
      setDefaultCategory(savedCategory);
    }
    // 백엔드에서 설정 불러오기
    loadUserSettingsFromBackend(userId);
  }, [userId]);

  // 검색어 변경 핸들러
  const handleSearchChange = (e) => {
    setSearchTerm(e.target.value);
  };

  // searchTerm 또는 currentMessages가 변경될 때 필터링 로직 적용
  useEffect(() => {
    if (searchTerm && currentMessages.length > 0) {
      const filtered = currentMessages.filter(
        (msg) =>
          typeof msg.content === "string" &&
          msg.content.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredMessages(filtered);
    } else {
      setFilteredMessages(currentMessages);
    }
  }, [searchTerm, currentMessages]);

  const isDark = theme === "dark";

  const handleToggleTheme = () => {
    const newTheme = isDark ? "light" : "dark";
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
    document.documentElement.classList.toggle("dark", newTheme === "dark");
    saveUserSettingsToBackend(userId, { theme: newTheme, defaultCategory });
  };

  // 테마 변경 함수
  const handleChangeTheme = (newTheme) => {
    setTheme(newTheme);
    localStorage.setItem("theme", newTheme);
    document.documentElement.classList.toggle("dark", newTheme === "dark");
    // 백엔드에 설정 저장
    saveUserSettingsToBackend(userId, { theme: newTheme, defaultCategory });
  };

  // 기본 카테고리 변경 함수
  const handleChangeDefaultCategory = (newCategory) => {
    setDefaultCategory(newCategory);
    localStorage.setItem("defaultCategory", newCategory);
    // 백엔드에 설정 저장
    saveUserSettingsToBackend(userId, { theme, defaultCategory: newCategory });
  };

  // 백엔드에 사용자 설정 저장
  const saveUserSettingsToBackend = async (userId, settings) => {
    try {
      const response = await fetch(
        "http://172.10.2.70:8000/api/settings/save",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            userId: userId,
            settings: settings,
          }),
        }
      );
      if (!response.ok) {
        throw new Error(
          `사용자 설정 저장 실패: ${response.status} ${response.statusText}`
        );
      }
      console.log("사용자 설정이 백엔드에 저장되었습니다.");
    } catch (err) {
      console.error("사용자 설정 저장 중 오류 발생:", err);
    }
  };

  // 백엔드에서 사용자 설정 불러오기
  const loadUserSettingsFromBackend = async (userId) => {
    try {
      const response = await fetch(
        "http://172.10.2.70:8000/api/settings/load",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            userId: userId,
          }),
        }
      );
      if (!response.ok) {
        throw new Error(
          `사용자 설정 불러오기 실패: ${response.status} ${response.statusText}`
        );
      }
      const data = await response.json();
      if (data.status === "success" && data.settings) {
        if (data.settings.theme) {
          setTheme(data.settings.theme);
          localStorage.setItem("theme", data.settings.theme);
          document.documentElement.classList.toggle(
            "dark",
            data.settings.theme === "dark"
          );
        }
        if (data.settings.defaultCategory) {
          setDefaultCategory(data.settings.defaultCategory);
          localStorage.setItem(
            "defaultCategory",
            data.settings.defaultCategory
          );
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
      localStorage.setItem(
        "activeConversationId",
        JSON.stringify(activeConversationId)
      );
    }
  }, [activeConversationId]);

  // conversations 변경 시 localStorage 및 백엔드 동기화
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem("conversations", JSON.stringify(conversations));
      // 백엔드에도 저장 (활성 대화만 저장 예시)
      if (activeConversationId) {
        const activeConv = conversations.find(
          (conv) => conv.id === activeConversationId
        );
        if (activeConv) {
          saveConversationToBackend(
            userId,
            activeConversationId,
            activeConv.messages
          );
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
      messages: [
        {
          role: "assistant",
          content: "안녕하세요! 무엇을 도와드릴까요?",
          sources: [],
        },
      ],
      pinned: false,
    };
    setConversations((prev) => [...prev, newConv]);
    setActiveConversationId(newConv.id);
    // 새 대화 생성 시 검색어 초기화
    setSearchTerm("");
  };

  // 대화 선택
  const handleSelectConversation = (id) => {
    setActiveConversationId(id);
    // 백엔드에서 대화 불러오기 (필요 시)
    loadConversationFromBackend(userId, id);
    // 모바일에서 대화 선택 시 사이드바 닫기
    if (window.innerWidth < 768) {
      setSidebarOpen(false);
    }
  };

  // 대화 삭제
  const handleDeleteConversation = (id) => {
    setConversations((prev) => {
      const updated = prev.filter((conv) => conv.id !== id);
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
          title: "대화 1",
          timestamp: now.toLocaleString(),
          messages: [
            {
              role: "assistant",
              content: "안녕하세요! 무엇을 도와드릴까요?",
              sources: [],
            },
          ],
          pinned: false,
        };
        setActiveConversationId(newConv.id);
        return [newConv];
      }
      return updated;
    });
  };

  // 대화 제목 변경
  const handleRenameConversation = (id, newTitle) => {
    setConversations((prev) =>
      prev.map((conv) => (conv.id === id ? { ...conv, title: newTitle } : conv))
    );
  };

  // 대화 즐겨찾기 토글
  const handleTogglePinConversation = (id) => {
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === id ? { ...conv, pinned: !conv.pinned } : conv
      )
    );
  };

  // 사용자 행동 기록 함수
  const logUserAction = async (action, details = {}) => {
    try {
      const response = await fetch("http://172.10.2.70:8000/api/stats/save", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          userId: userId,
          action: action,
          details: details,
        }),
      });
      if (!response.ok) {
        throw new Error(
          `통계 저장 실패: ${response.status} ${response.statusText}`
        );
      }
      console.log(`통계 저장됨: ${action}`);
    } catch (err) {
      console.error("통계 저장 중 오류 발생:", err);
    }
  };

  // 통계 조회 함수
  const fetchStats = async () => {
    try {
      const response = await fetch("http://172.10.2.70:8000/api/stats/query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          userId: "", // 전체 사용자 조회
          startDate: "", // 모든 기간
          endDate: "", // 모든 기간
        }),
      });
      if (!response.ok) {
        throw new Error(
          `통계 조회 실패: ${response.status} ${response.statusText}`
        );
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
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === activeConversationId
          ? { ...conv, messages: updatedMessages }
          : conv
      )
    );
    // 사용자가 질문을 보낸 경우 통계 기록
    if (
      updatedMessages.length > 0 &&
      updatedMessages[updatedMessages.length - 1].role === "user"
    ) {
      logUserAction("question", {
        category: defaultCategory,
        timestamp: new Date().toISOString(),
      });
    }
  };

  // 백엔드에 대화 저장
  const saveConversationToBackend = async (
    userId,
    conversationId,
    messages
  ) => {
    try {
      const response = await fetch(
        "http://172.10.2.70:8000/api/conversations/save",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            userId: userId,
            conversationId: conversationId,
            messages: messages.map((msg) => ({
              role: msg.role,
              content: msg.content,
              sources: msg.sources || [],
            })),
          }),
        }
      );
      if (!response.ok) {
        throw new Error(
          `대화 저장 실패: ${response.status} ${response.statusText}`
        );
      }
      console.log("대화가 백엔드에 저장되었습니다.");
      setSaveError(null); // 성공 시 오류 메시지 초기화
    } catch (err) {
      console.error("대화 저장 중 오류 발생:", err);
      setSaveError("대화 저장에 실패했습니다. 로컬에만 저장됩니다."); // 오류 메시지 설정
      // 로컬 저장은 이미 conversations 변경 시 useEffect에서 보장됨
      setTimeout(() => setSaveError(null), 5000); // 5초 후 알림 사라짐
    }
  };

  // 백엔드에서 대화 불러오기
  const loadConversationFromBackend = async (userId, conversationId) => {
    try {
      const response = await fetch(
        "http://172.10.2.70:8000/api/conversations/load",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            userId: userId,
            conversationId: conversationId,
          }),
        }
      );
      if (!response.ok) {
        throw new Error(
          `대화 불러오기 실패: ${response.status} ${response.statusText}`
        );
      }
      const data = await response.json();
      if (data.status === "success" && data.conversation) {
        setConversations((prev) =>
          prev.map((conv) =>
            conv.id === conversationId
              ? { ...conv, messages: data.conversation.messages }
              : conv
          )
        );
        console.log("대화가 백엔드에서 불러와졌습니다.");
      }
    } catch (err) {
      console.error("대화 불러오기 중 오류 발생:", err);
    }
  };

  // conversations와 activeConversationId가 초기화된 경우에만 메시지 가져오기
  useEffect(() => {
    if (conversations.length > 0 && activeConversationId) {
      const activeConv = conversations.find(
        (conv) => conv.id === activeConversationId
      );
      if (activeConv) {
        setCurrentMessages(activeConv.messages);
      }
    } else {
      setCurrentMessages([
        {
          role: "assistant",
          content: "안녕하세요! 무엇을 도와드릴까요?",
          sources: [],
        },
      ]);
    }
  }, [conversations, activeConversationId]);

  // 반응형: 화면 크기 변경 시 사이드바 상태 업데이트
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 768 && sidebarOpen) {
        setSidebarOpen(false);
      } else if (window.innerWidth >= 1200 && !sidebarOpen) {
        setSidebarOpen(true);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [sidebarOpen]);

  // 드래그 핸들러
  const handleMouseDown = (e) => {
    isResizing.current = true;
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", handleMouseUp);
  };
  const handleMouseMove = (e) => {
    if (!isResizing.current) return;
    const newWidth = Math.max(
      SIDEBAR_MIN,
      Math.min(SIDEBAR_MAX, e.clientX)
    );
    setSidebarWidth(newWidth);
  };
  const handleMouseUp = () => {
    isResizing.current = false;
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", handleMouseUp);
  };

  // 토글(여닫이) 버튼
  const handleToggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  // 설정 드롭다운 토글
  const handleToggleSettingsDropdown = () => {
    setShowSettingsDropdown(!showSettingsDropdown);
  };

  // Clear Chat 기능 (예시)
  const handleClearChat = () => {
    if (window.confirm("대화 내용을 모두 삭제하시겠습니까?")) {
      const updatedConversations = conversations.map((c) =>
        c.id === activeConversationId
          ? {
              ...c,
              messages: [
                {
                  role: "assistant",
                  content: "안녕하세요! 무엇을 도와드릴까요?",
                  sources: [],
                },
              ],
            }
          : c
      );
      setConversations(updatedConversations);
    }
  };

  // 알림 설정 토글
  const handleToggleNotifications = () => {
    setNotificationsEnabled(!notificationsEnabled);
  };

  // 스크롤 잠금 토글
  const handleToggleScrollLock = () => {
    setScrollLocked(!scrollLocked);
  };

  return (
    <div className={`flex flex-col h-screen ${isDark ? "dark" : ""} bg-gray-50 dark:bg-slate-900`}>
      {/* 헤더 */}
      <header className="border-b border-slate-200 dark:border-slate-800 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-slate-900 dark:to-slate-800 shadow-sm z-10 relative">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <button
              className="p-2 rounded-lg hover:bg-slate-100/70 dark:hover:bg-slate-800/80 text-slate-700 dark:text-slate-300 md:hidden transition-colors"
              onClick={handleToggleSidebar}
              aria-label="메뉴 토글"
            >
              <FiMenu size={20} />
            </button>
            <div className="flex items-center gap-2.5">
              <FiServer className="text-blue-500 dark:text-blue-400" size={22} />
              <h1 className="font-semibold text-xl bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">RAG 챗봇</h1>
            </div>
          </div>
          
          <div className="flex items-center gap-1">
            <button
              className="p-2 rounded-lg hover:bg-slate-100/70 dark:hover:bg-slate-800/80 text-slate-700 dark:text-slate-300 transition-colors"
              onClick={handleToggleTheme}
              aria-label={isDark ? "라이트 모드로 전환" : "다크 모드로 전환"}
            >
              {isDark ? <FiSun size={18} /> : <FiMoon size={18} />}
            </button>
            
            <button
              className="hidden md:flex items-center justify-center p-2 rounded-lg hover:bg-slate-100/70 dark:hover:bg-slate-800/80 
                text-slate-700 dark:text-slate-300 transition-colors"
              onClick={handleNewConversation}
              aria-label="새 대화"
            >
              <FiPlus size={18} />
            </button>
            
            <button
              className="p-2 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 
                text-slate-700 dark:text-slate-300 hover:text-red-600 dark:hover:text-red-400 transition-colors"
              onClick={handleClearChat}
              aria-label="대화 초기화"
            >
              <FiTrash2 size={18} />
            </button>
          </div>
        </div>
      </header>

      {/* 메인 컨테이너 */}
      <div className="flex flex-1 overflow-hidden">
        {/* 사이드바 */}
        {sidebarOpen && (
          <div
            className="flex-shrink-0 h-full"
            style={{ width: `${sidebarWidth}px` }}
          >
            <Sidebar
              collapsed={false}
              conversations={conversations}
              activeConversationId={activeConversationId}
              onNewConversation={handleNewConversation}
              onDeleteConversation={handleDeleteConversation}
              onSelectConversation={handleSelectConversation}
              onRenameConversation={handleRenameConversation}
              onTogglePinConversation={handleTogglePinConversation}
            />
          </div>
        )}

        {/* 드래그 핸들 */}
        {sidebarOpen && (
          <div
            className="w-1 h-full cursor-col-resize bg-slate-200 dark:bg-slate-800 hover:bg-blue-500 dark:hover:bg-blue-600 custom-transition-colors"
            onMouseDown={handleMouseDown}
          ></div>
        )}

        {/* 메인 채팅 영역 */}
        <div className="flex-1 flex flex-col bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-950">
          {/* 모바일 토글 버튼 */}
          {!sidebarOpen && (
            <button
              className="fixed bottom-4 left-4 z-10 p-3 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white shadow-lg custom-transition-all md:hidden"
              onClick={handleToggleSidebar}
            >
              <FiMessageSquare size={20} />
            </button>
          )}

          {/* 채팅 컨테이너 */}
          <ChatContainer
            scrollLocked={scrollLocked}
            activeConversationId={activeConversationId}
            messages={currentMessages}
            filteredMessages={filteredMessages}
            searchTerm={searchTerm}
            onUpdateMessages={handleUpdateMessages}
          />
        </div>
      </div>
      
      {/* 에러 메시지 표시 */}
      {saveError && (
        <div className="fixed bottom-4 right-4 bg-red-100 dark:bg-red-900/30 border border-red-200 dark:border-red-800 
          text-red-800 dark:text-red-200 px-4 py-3 rounded-lg shadow-lg animate-fade-in">
          <div className="flex items-start">
            <FiAlertCircle className="mt-0.5 mr-2 flex-shrink-0" />
            <div>
              <p className="font-medium">저장 오류</p>
              <p className="text-sm">{saveError}</p>
            </div>
            <button 
              className="ml-3 text-red-700 dark:text-red-300 hover:text-red-900 dark:hover:text-red-100"
              onClick={() => setSaveError(null)}
            >
              <FiX size={18} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
