import React, { useRef, useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import ChatContainer from "./components/ChatContainer";
import {
  FiChevronLeft,
  FiChevronRight,
  FiMessageSquare,
  FiPlus,
  FiX,
  FiLoader,
  FiCheckCircle,
  FiFile
} from "react-icons/fi";

const SIDEBAR_WIDTH = 280;
const SIDEBAR_MIN = 240;
const SIDEBAR_MAX = 400;

// 임베딩 알림 오버레이 컴포넌트 추가
const EmbeddingOverlay = ({ isActive, status, files }) => {
  if (!isActive && !status) return null;
  
  const isCompleted = status === '완료';
  
  return (
    <div className="fixed inset-0 bg-gray-900/90 backdrop-blur-sm flex flex-col items-center justify-center z-[100] animate-fade-in">
      <div className="max-w-lg w-full px-6 py-8 rounded-2xl bg-gray-800/70 backdrop-blur-md text-center space-y-6">
        <div className="relative mx-auto w-24 h-24">
          <div className="absolute inset-0 rounded-full border-4 border-indigo-500/20"></div>
          
          {!isCompleted ? (
          <div className="absolute inset-0 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin"></div>
          ) : (
            <div className="absolute inset-0 rounded-full border-4 border-green-500 flex items-center justify-center animate-pulse">
              <FiCheckCircle className="text-green-400" size={36} />
            </div>
          )}
          
          <div className="absolute inset-4 rounded-full bg-indigo-500/20 flex items-center justify-center">
            <FiFile className={isCompleted ? "text-green-200" : "text-indigo-200"} size={24} />
          </div>
        </div>
        
        <div className="space-y-2">
          <h3 className="text-xl font-bold text-white">
            {isCompleted ? "파일 임베딩 완료!" : (status || "파일 임베딩 처리 중...")}
          </h3>
          <p className="text-gray-300 text-sm">
            {isCompleted ? (
              <>파일이 성공적으로 임베딩되었습니다.<br/>이제 챗봇과의 대화에 활용할 수 있습니다.</>
            ) : (
              <>이 과정은 파일 크기와 내용에 따라 몇 분 정도 소요될 수 있습니다.<br/>
              임베딩이 완료될 때까지 기다려주세요.</>
            )}
          </p>
        </div>
        
        {files && files.length > 0 && !isCompleted && (
          <div className="bg-gray-900/50 rounded-xl p-4 max-h-40 overflow-y-auto">
            <p className="text-gray-400 text-xs mb-2">{files.length}개 파일 처리 중:</p>
            <div className="space-y-1.5">
              {files.map((file, index) => (
                <div key={index} className="flex items-center">
                  <div className="w-2 h-2 bg-indigo-500 rounded-full mr-2"></div>
                  <span className="text-gray-200 text-sm truncate">{file.name}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {isCompleted && (
          <button 
            className="mt-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
            onClick={() => { window.dispatchEvent(new CustomEvent('refreshIndexedFiles')); }}
          >
            파일 목록 보기
          </button>
        )}
      </div>
    </div>
  );
};

function App() {
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_WIDTH);
  const [sidebarOpen, setSidebarOpen] = useState(window.innerWidth >= 768); // md: 이상에서 기본적으로 열림
  const [userName, setUserName] = useState("사용자");
  const [scrollLocked, setScrollLocked] = useState(false);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const isResizing = useRef(false);
  const [userId, setUserId] = useState("user1"); // 임시 사용자 ID, 실제로는 인증 기반 ID 사용
  const [theme, setTheme] = useState(() => {
    // 항상 다크모드를 기본값으로 설정
    localStorage.setItem("theme", "dark");
    return "dark";
  });
  const [defaultCategory, setDefaultCategory] = useState("메뉴얼"); // 기본 카테고리 상태
  const [saveError, setSaveError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [currentMessages, setCurrentMessages] = useState([
    {
      role: "assistant",
      content: "안녕하세요! 무엇을 도와드릴까요?",
      sources: [],
      timestamp: new Date().getTime(),
    },
  ]);
  const [filteredMessages, setFilteredMessages] = useState(currentMessages);
  // 임베딩 상태 관련 변수 추가
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [embeddingStatus, setEmbeddingStatus] = useState(null);
  const [embeddedFiles, setEmbeddedFiles] = useState([]);

  // 테마 변경 함수
  const toggleTheme = useCallback(() => {
    // 다크모드만 유지하도록 수정
    setTheme("dark");
    localStorage.setItem("theme", "dark");
    document.documentElement.classList.add("dark");
  }, []);
  
  // 테마 적용
  useEffect(() => {
    // 항상 다크모드 적용
    document.documentElement.classList.add("dark");
  }, [theme]);

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

  // 초기 설정 로드 (로컬 스토리지 또는 백엔드) 수정
  useEffect(() => {
    // 다크모드로 항상 설정
    const savedTheme = "dark";
    const savedCategory = localStorage.getItem("defaultCategory");
    setTheme(savedTheme);
    document.documentElement.classList.add("dark");
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

  // 인덱싱된 파일 목록 새로고침을 위한 이벤트 리스너 추가
  useEffect(() => {
    const handleRefreshFiles = () => {
      setFileManagerOpen(true);
    };
    
    window.addEventListener('refreshIndexedFiles', handleRefreshFiles);
    
    return () => {
      window.removeEventListener('refreshIndexedFiles', handleRefreshFiles);
    };
  }, []);

  // 파일 목록 표시 상태
  const [fileManagerOpen, setFileManagerOpen] = useState(false);

  // 파일 업로드 완료 핸들러 추가
  const handleUploadSuccess = (files) => {
    console.log('업로드 성공:', files);
    setIsEmbedding(true);
    setEmbeddedFiles(files);

    // 5초 후 임베딩 완료 처리 (실제로는 서버에서 완료 신호를 받아야 함)
    setTimeout(() => {
      setIsEmbedding(false);
      // 임베딩 완료 상태를 설정하고 UI로 표시
      setEmbeddingStatus('완료');
      // 5초 후 임베딩 상태 초기화
      setTimeout(() => {
        setEmbeddingStatus(null);
        setEmbeddedFiles([]);
      }, 5000);
    }, 5000);
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
            "light",
            data.settings.theme === "light"
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

  // 대화 메시지 변경 시 자동 저장
  useEffect(() => {
    // 대화가 변경되면 로컬 스토리지에 자동 저장
    if (conversations.length > 0) {
      localStorage.setItem("conversations", JSON.stringify(conversations));
      // 활성화된 대화에 대해서만 백엔드에 저장
      if (activeConversationId) {
        const activeConv = conversations.find(
          (c) => c.id === activeConversationId
        );
        if (activeConv && activeConv.messages.length > 0) {
          console.log("로컬 스토리지에 대화 저장 완료");
          // 백엔드 저장 비활성화 (422 오류 문제 해결을 위함)
          // saveConversationToBackend(userId, activeConversationId, activeConv.messages);
        }
      }
    }
  }, [conversations, activeConversationId, userId]);

  // 새 대화 생성
  const handleNewConversation = (topic, category) => {
    // 새 대화 ID 생성 (UUID 형식)
    const newConversationId = `conversation-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    
    // 첫 시스템 메시지 설정 (대화 컨텍스트 정보)
    let contextMessage = "안녕하세요! 무엇을 도와드릴까요?";
    
    // 새 대화 생성
    const newConversation = {
      id: newConversationId,
      messages: [
        {
          role: "assistant",
          content: contextMessage,
          timestamp: Date.now()
        }
      ],
      title: topic || "새 대화",
      created_at: new Date().toISOString()
    };
    
    // 대화 목록 및 활성 대화 업데이트
    setConversations(prev => [newConversation, ...prev]);
    setActiveConversationId(newConversationId);
    
    // 검색어 초기화
    setSearchTerm('');
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
    // 확인 창 없이 바로 삭제 처리
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

  // 메시지 업데이트
  const handleUpdateMessages = (updatedMessages) => {
    setConversations((prev) =>
      prev.map((conv) =>
        conv.id === activeConversationId
          ? { ...conv, messages: updatedMessages }
          : conv
      )
    );
  };

  return (
    <div className="h-screen overflow-hidden bg-gray-900 dark:bg-gray-900 text-gray-300 dark:text-gray-300 transition-colors duration-200">
      {/* Sidebar */}
      <div
        className={`fixed left-0 top-0 h-full z-10 transition-all duration-300 ${
          sidebarOpen
            ? `w-[${sidebarWidth}px] translate-x-0`
            : "w-0 -translate-x-full"
        }`}
        style={{ width: sidebarOpen ? `${sidebarWidth}px` : 0 }}
      >
        <Sidebar
          conversations={conversations}
          activeConversationId={activeConversationId}
          onNewConversation={handleNewConversation}
          onDeleteConversation={handleDeleteConversation}
          onSelectConversation={handleSelectConversation}
          onRenameConversation={handleRenameConversation}
          onTogglePinConversation={handleTogglePinConversation}
          onToggleTheme={toggleTheme}
          isDarkMode={theme === "dark"}
        />
      </div>

      {/* Sidebar 드래그 리사이즈 핸들 */}
      {sidebarOpen && (
        <div
          className="fixed left-0 top-0 h-full z-20 w-1 cursor-ew-resize flex items-center justify-center"
          style={{ left: `${sidebarWidth}px` }}
          onMouseDown={handleMouseDown}
        >
          <div className="h-8 w-1 bg-gray-900 rounded-full"></div>
        </div>
      )}

      {/* 메인 콘텐츠 */}
      <div
        className="h-full transition-all duration-300"
        style={{
          marginLeft: sidebarOpen ? `${sidebarWidth}px` : 0,
        }}
      >
        {/* 사이드바 토글 버튼 */}
        <button
          className="fixed top-4 left-4 z-30 p-2 rounded-lg bg-gray-800 text-white hover:bg-gray-700 shadow-lg"
          onClick={handleToggleSidebar}
        >
          {sidebarOpen ? (
            <FiChevronLeft size={20} />
          ) : (
            <FiChevronRight size={20} />
          )}
        </button>

        {/* 대화 컨테이너 */}
        <ChatContainer
          scrollLocked={scrollLocked}
          activeConversationId={activeConversationId}
          messages={currentMessages}
          searchTerm={searchTerm}
          filteredMessages={filteredMessages}
          onUpdateMessages={handleUpdateMessages}
          isEmbedding={isEmbedding}
          onUploadSuccess={handleUploadSuccess}
          onNewConversation={handleNewConversation}
          fileManagerOpen={fileManagerOpen}
          setFileManagerOpen={setFileManagerOpen}
        />
      </div>

      {/* 임베딩 알림 오버레이 */}
      <EmbeddingOverlay 
        isActive={isEmbedding} 
        status={embeddingStatus} 
        files={embeddedFiles}
      />
    </div>
  );
}

export default App;
