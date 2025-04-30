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
} from "react-icons/fi";

const SIDEBAR_WIDTH = 220;
const SIDEBAR_MIN = 60;
const SIDEBAR_MAX = 400;

function App() {
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_WIDTH);
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768); // md: 이상에서 기본적으로 열림
  const [model, setModel] = useState("GPT-4");
  const [showModelDropdown, setShowModelDropdown] = useState(false);
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
  };

  // 대화 선택
  const handleSelectConversation = (id) => {
    setActiveConversationId(id);
    // 백엔드에서 대화 불러오기 (필요 시)
    loadConversationFromBackend(userId, id);
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
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      } else {
        setSidebarOpen(true);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // 드래그 핸들러
  const handleMouseDown = (e) => {
    isResizing.current = true;
    document.body.style.cursor = "col-resize";
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
      document.body.style.cursor = "default";
    }
  };
  useEffect(() => {
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
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
    console.log("Regenerate last response");
    // 실제로는 마지막 메시지 재생성 로직 추가
  };

  // Clear Chat 기능 (예시)
  const handleClearChat = () => {
    console.log("Clear chat history");
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
          ${sidebarOpen ? "border-r border-gray-200 dark:border-gray-700" : ""}
          bg-white dark:bg-gray-800
          md:block hidden
        `}
        style={{
          width: sidebarOpen ? sidebarWidth : 0,
          minWidth: sidebarOpen ? SIDEBAR_MIN : 0,
          maxWidth: sidebarOpen ? SIDEBAR_MAX : 0,
          transform: sidebarOpen
            ? "translateX(0)"
            : `translateX(-${sidebarWidth}px)`,
          overflow: "hidden",
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
          style={{ userSelect: "none" }}
          title="사이드바 크기 조절"
        />
      )}
      {/* 본문 */}
      <main className="flex-1 flex flex-col border-l border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-900/80 transition-all duration-300 md:w-0 w-full">
        <header className="p-4 border-b bg-white dark:bg-gray-800 flex items-center justify-between sticky top-0 z-10 backdrop-blur-sm">
          <h1 className="text-2xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-500 to-indigo-600 font-['Pretendard','Noto Sans KR',sans-serif] drop-shadow">
            쓰리에스소프트 챗봇 테스트 ㅇㅇㅇㅇ
          </h1>
          <div className="flex items-center gap-2">
            <div className="relative w-full max-w-xs">
              <input
                type="text"
                value={searchTerm}
                onChange={handleSearchChange}
                placeholder="메시지 검색..."
                className="
      w-full pl-10 pr-1 py-2
      text-sm font-medium
      text-gray-800 placeholder-gray-400
      dark:text-white dark:placeholder-gray-500
      bg-white/60 dark:bg-gray-800/70
      backdrop-blur-md
      border border-gray-200 dark:border-gray-600
      rounded-xl
      shadow-[inset_0_1px_2px_rgba(0,0,0,0.05),0_2px_6px_rgba(0,0,0,0.08)]
      hover:shadow-[inset_0_1px_3px_rgba(0,0,0,0.08),0_4px_12px_rgba(0,0,0,0.1)]
      focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
      transition-all duration-200 ease-in-out
    "
              />
              <FiSearch
                className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500"
                size={16}
              />
            </div>

            <button
              onClick={handleToggleTheme}
              className={`
    group relative flex items-center justify-center
    w-10 h-10 rounded-full
    bg-white dark:bg-gray-800
    border border-gray-200 dark:border-gray-600
    shadow-md hover:shadow-lg
    transition-all duration-300 ease-in-out
  `}
              title="다크모드 토글"
            >
              <span
                className={`
    absolute inset-0 flex items-center justify-center
    transition-all duration-300 ease-in-out
    ${
      isDark ? "opacity-0 scale-75 rotate-90" : "opacity-100 scale-100 rotate-0"
    }
  `}
              >
                <FiSun className="text-yellow-500" size={18} />
              </span>

              <span
                className={`
    absolute inset-0 flex items-center justify-center
    transition-all duration-300 ease-in-out
    ${
      isDark
        ? "opacity-100 scale-100 rotate-0"
        : "opacity-0 scale-75 -rotate-90"
    }
  `}
              >
                <FiMoon className="text-blue-300" size={18} />
              </span>
            </button>
          </div>
        </header>
        <ChatContainer
          key={activeConversationId}
          scrollLocked={scrollLocked}
          activeConversationId={activeConversationId}
          messages={currentMessages}
          onUpdateMessages={handleUpdateMessages}
          filteredMessages={filteredMessages}
          searchTerm={searchTerm}
        />
      </main>
      {saveError && (
        <div className="fixed bottom-4 right-4 bg-red-500 text-white px-4 py-2 rounded shadow-lg z-50 animate-fade-in">
          {saveError}
        </div>
      )}
    </div>
  );
}

export default App;
