import React, { useRef, useState, useEffect, useCallback } from "react";
import Sidebar from "./components/Sidebar";
import ChatContainer from "./components/ChatContainer";
import SQLQueryPage from "./components/SQLQueryPage";
import {
  FiChevronLeft,
  FiChevronRight,
  FiMessageSquare,
  FiPlus,
  FiX,
  FiLoader,
  FiCheckCircle,
  FiFile,
  FiMenu,
  FiAlignLeft,
  FiDatabase,
  FiAlertCircle
} from "react-icons/fi";

const SIDEBAR_WIDTH = 320;
const SIDEBAR_MIN = 280;
const SIDEBAR_MAX = 450;

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
  
  // SQL 쿼리 페이지 표시 여부 상태 추가
  const [showSQLPage, setShowSQLPage] = useState(false);
  // 모드 상태 추가 (chat 또는 sql)
  const [mode, setMode] = useState('chat');
  // 파일 매니저 상태 추가
  const [fileManagerOpen, setFileManagerOpen] = useState(false);

  console.log('===============================================');
  console.log('앱 컴포넌트 초기화');
  console.log('showSQLPage 초기값:', showSQLPage);
  console.log('mode 초기값:', mode);
  
  // 브라우저 콘솔에서 직접 상태를 변경할 수 있는 디버깅 함수 추가
  useEffect(() => {
    // 전역 함수로 등록
    window.setAppMode = (newMode) => {
      if (newMode === 'sql' || newMode === 'chat') {
        console.log('콘솔에서 모드 변경:', newMode);
        setMode(newMode);
        return true;
      }
      console.error('유효하지 않은 모드:', newMode, '(sql 또는 chat만 가능)');
      return false;
    };

    window.toggleAppMode = () => {
      const newMode = mode === 'chat' ? 'sql' : 'chat';
      console.log('콘솔에서 모드 토글:', mode, '->', newMode);
      setMode(newMode);
      return newMode;
    };
    
    window.debugAppState = () => {
      console.log('현재 앱 상태:');
      console.log('- 현재 모드:', mode);
      console.log('- showSQLPage:', showSQLPage);
      console.log('- sidebarOpen:', sidebarOpen);
      console.log('- activeConversationId:', activeConversationId);
      console.log('- isEmbedding:', isEmbedding);
      console.log('- theme:', theme);
    };
    
    console.log('디버깅 함수가 콘솔에 등록되었습니다. 사용법:');
    console.log('window.setAppMode("chat" 또는 "sql") - 모드 직접 설정');
    console.log('window.toggleAppMode() - 모드 전환');
    console.log('window.debugAppState() - 현재 앱 상태 출력');
    
    return () => {
      // 정리 함수
      delete window.setAppMode;
      delete window.toggleAppMode;
      delete window.debugAppState;
    };
  }, [mode, showSQLPage, sidebarOpen, activeConversationId, isEmbedding, theme]);
  
  // mode 상태가 변경될 때 toast 메시지 표시
  useEffect(() => {
    console.log('모드 변경됨:', mode);
    // 모드 변경 시 토스트 메시지 표시
    if (mode === 'sql') {
      showModeChangeToast('SQL 질의 모드로 전환했습니다');
    } else if (mode === 'chat') {
      showModeChangeToast('챗봇 모드로 전환했습니다');
    }
  }, [mode]);

  // showSQLPage 상태 변경 감지 useEffect 추가
  useEffect(() => {
    console.log('showSQLPage 상태 변경됨:', showSQLPage);
  }, [showSQLPage]);

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
    const handleRefreshFilesEvent = () => {
      setFileManagerOpen(true);
    };
    
    window.addEventListener('refreshIndexedFiles', handleRefreshFilesEvent);
    
    return () => {
      window.removeEventListener('refreshIndexedFiles', handleRefreshFilesEvent);
    };
  }, []);

  // 파일 목록 새로고침 함수 추가
  const handleRefreshFiles = () => {
    setFileManagerOpen(true);
    console.log('파일 목록 새로고침');
  };

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
      // 백엔드 연결 시도 전 로컬 기본값 설정
      const savedTheme = localStorage.getItem("theme") || "dark";
      const savedDefaultCategory = localStorage.getItem("defaultCategory") || "메뉴얼";
      
      // 로컬 저장소의 값 적용
      setTheme(savedTheme);
      setDefaultCategory(savedDefaultCategory);
      document.documentElement.classList.toggle("light", savedTheme === "light");
      
      console.log("로컬 설정 적용됨:", { theme: savedTheme, defaultCategory: savedDefaultCategory });
      
      // 백엔드 요청 시도 (타임아웃 설정)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3초 타임아웃
      
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
          signal: controller.signal
        }
      );
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        console.warn(`사용자 설정 불러오기 실패: ${response.status} ${response.statusText}`);
        return; // 이미 로컬 설정이 적용되었으므로 종료
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
        console.log("사용자 설정이 없습니다. 로컬 설정을 유지합니다.");
      }
    } catch (err) {
      // 오류 발생 시 콘솔에 경고만 표시하고 앱은 계속 실행
      console.warn("사용자 설정 불러오기 중 오류 발생:", err.message);
      console.log("로컬 설정을 사용합니다.");
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
      try {
        // 직렬화 전에 DOM 요소 참조를 제거하기 위해 순수 객체만 추출
        const cleanConversations = conversations.map(conv => ({
          id: conv.id,
          title: conv.title,
          timestamp: conv.timestamp,
          pinned: !!conv.pinned,
          messages: conv.messages.map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp || Date.now(),
            sources: Array.isArray(msg.sources) ? msg.sources : []
          }))
        }));
        
        localStorage.setItem("conversations", JSON.stringify(cleanConversations));
        console.log("로컬 스토리지에 대화 저장 완료");
      } catch (error) {
        console.error("대화 저장 중 오류 발생:", error);
      }
      
      // 활성화된 대화에 대해서만 백엔드에 저장
      if (activeConversationId) {
        const activeConv = conversations.find(
          (c) => c.id === activeConversationId
        );
        if (activeConv && activeConv.messages.length > 0) {
          // 백엔드 저장 비활성화 (422 오류 문제 해결을 위함)
          // saveConversationToBackend(userId, activeConversationId, activeConv.messages);
        }
      }
    }
  }, [conversations, activeConversationId, userId]);

  // 전체 대화 삭제 기능 추가
  const deleteAllConversations = () => {
    try {
      // 로컬 상태 초기화
      setConversations([]);
      setActiveConversationId(null);
      setCurrentMessages([]);
      
      // 로컬 스토리지 데이터 삭제
      localStorage.removeItem('conversations');
      localStorage.removeItem('activeConversationId');
      
      // 백엔드에 요청 (옵션)
      fetch("http://172.10.2.70:8000/api/conversations/delete-all", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ userId: "user123" }),
      }).catch(err => {
        console.warn("백엔드에 전체 삭제 요청 실패:", err);
        // 백엔드 실패해도 로컬에서는 삭제된 상태로 유지
      });
      
      // 성공 메시지 표시
      showModeChangeToast("모든 대화가 삭제되었습니다");
      
      // 새 대화 시작 - 항상 '대화 1'로 시작
      handleNewConversation(null, defaultCategory, true);
    } catch (err) {
      console.error("대화 전체 삭제 중 오류 발생:", err);
    }
  };

  // 새 대화 생성
  const handleNewConversation = (topic, category, forceFirst = false) => {
    try {
      console.log('새 대화 시작...', topic, category);
      
      // 초기 제목 설정
      const initialTitle = topic || `대화 ${conversations.length + 1}`;
      
      // 고유 ID 생성
      const newId = `conv_${Date.now()}`;
      
      // 새 대화 객체 생성 (구조 개선 - 메타데이터 추가)
      const newConversation = {
        id: newId,
        title: initialTitle,
        messages: [
          // 시스템 시작 메시지 추가하여 챗봇이 먼저 인사하도록 함
          {
            role: "assistant",
            content: "안녕하세요! 무엇을 도와드릴까요?",
            sources: [],
            timestamp: Date.now(),
          }
        ],
        timestamp: Date.now(),
        category: category || defaultCategory,
        pinned: false,
        metadata: {
          messageCount: 1,
          firstMessageTimestamp: Date.now(),
          lastActivity: Date.now()
        }
      };
      
      // 새 대화 추가 (맨 앞 또는 맨 뒤)
      if (forceFirst) {
        // 가장 최근 대화로 추가 (맨 앞)
        setConversations(prevConversations => [
          newConversation,
          ...prevConversations
        ]);
      } else {
        // 새 대화 추가 (맨 뒤)
        setConversations(prevConversations => [
          ...prevConversations,
          newConversation
        ]);
      }
      
      // 새 대화를 현재 활성 대화로 설정
      setActiveConversationId(newId);
      
      // 메시지 초기화
      setCurrentMessages([
        // 시스템 시작 메시지 추가하여 챗봇이 먼저 인사하도록 함
        {
          role: "assistant",
          content: "안녕하세요! 무엇을 도와드릴까요?",
          sources: [],
          timestamp: Date.now(),
        }
      ]);
      
      // 검색어 초기화
      setSearchTerm("");
      
      // 사이드바 모바일에서 자동으로 닫기
      if (window.innerWidth < 768) {
        setSidebarOpen(false);
      }
      
      return newId;
    } catch (error) {
      console.error('새 대화 생성 중 오류 발생:', error);
      return null;
    }
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
    
    // 대화창 전환 시 스크롤을 최신 메시지로 강력하게 이동시키기 위한 다중 이벤트 발생
    
    // 1. 즉시 이벤트 발생
    window.dispatchEvent(new CustomEvent('chatScrollToBottom'));
    
    // 2. 약간의 지연 후 이벤트 발생 (DOM 업데이트 기다림)
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('chatScrollToBottom'));
    }, 50);
    
    // 3. 대화 내용 로드 완료 후 이벤트 발생 (비동기 작업 완료 대기)
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('chatScrollToBottom'));
    }, 300);
    
    // 4. 렌더링 완료 보장을 위한 추가 이벤트
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('chatScrollToBottom'));
    }, 800);
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
      // 백엔드 요청 시도 (타임아웃 설정)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3초 타임아웃
      
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
          signal: controller.signal
        }
      );
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        console.warn(`대화 저장 실패: ${response.status} ${response.statusText}`);
        setSaveError("대화 저장에 실패했습니다. 로컬에만 저장됩니다.");
        setTimeout(() => setSaveError(null), 5000);
        return;
      }
      
      console.log("대화가 백엔드에 저장되었습니다.");
      setSaveError(null); // 성공 시 오류 메시지 초기화
    } catch (err) {
      console.warn("대화 저장 중 오류 발생:", err.message);
      setSaveError("대화 저장에 실패했습니다. 로컬에만 저장됩니다."); // 오류 메시지 설정
      // 로컬 저장은 이미 conversations 변경 시 useEffect에서 보장됨
      setTimeout(() => setSaveError(null), 5000); // 5초 후 알림 사라짐
    }
  };

  // 백엔드에서 대화 불러오기
  const loadConversationFromBackend = async (userId, conversationId) => {
    try {
      // 백엔드 요청 시도 (타임아웃 설정)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000); // 3초 타임아웃
      
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
          signal: controller.signal
        }
      );
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        console.warn(`대화 불러오기 실패: ${response.status} ${response.statusText}`);
        return; // 로컬 데이터 유지
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
      console.warn("대화 불러오기 중 오류 발생:", err.message);
      console.log("로컬 대화 데이터를 사용합니다.");
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
    if (!activeConversationId) return;
    
    // 대화 업데이트
    setConversations((prev) => {
      const updated = prev.map((conv) => {
        if (conv.id === activeConversationId) {
          // 기존 대화 업데이트
          return { ...conv, messages: updatedMessages, timestamp: new Date().getTime() };
        }
        return conv;
      });
      return updated;
    });
    setCurrentMessages(updatedMessages);
    
    // 자동 제목 생성 로직: 사용자 메시지가 있을 때 모든 제목에 대해 동작
    const userMessages = updatedMessages.filter(msg => msg.role === "user");
    
    // 첫 번째 사용자 메시지가 추가됐을 때만 자동 제목 생성을 수행
    const activeConv = conversations.find(c => c.id === activeConversationId);
    const prevUserMessages = activeConv?.messages?.filter(msg => msg.role === "user") || [];
    
    if (userMessages.length > 0 && prevUserMessages.length === 0) {
      console.log("첫 질문 감지: 자동 제목 생성 시도");
      // 비동기로 제목 생성 API 호출
      generateTitleForConversation(activeConversationId, updatedMessages);
    }
  };

  // 대화 제목 자동 생성 함수
  const generateTitleForConversation = async (conversationId, messages) => {
    try {
      console.log("제목 생성 API 호출 - 메시지:", messages);
      
      // 백엔드 요청 시도 (타임아웃 설정)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000); // 5초 타임아웃
      
      const response = await fetch("http://172.10.2.70:8000/api/generate-title", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ messages }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        console.warn(`제목 생성 실패: ${response.status} ${response.statusText}`);
        // 백엔드 API 호출 실패 시 대화 메시지 기반으로 제목 생성
        generateFallbackTitle(conversationId, messages);
        return;
      }
      
      const data = await response.json();
      console.log("제목 생성 응답:", data);
      
      if (data.status === "success" && data.title) {
        // 백엔드에서 제공한 제목으로 업데이트
        handleRenameConversation(conversationId, data.title);
      } else {
        // 실패하면 폴백 제목 생성
        generateFallbackTitle(conversationId, messages);
      }
    } catch (err) {
      console.warn("제목 생성 중 오류 발생:", err.message);
      // 오류 발생 시 폴백 제목 생성
      generateFallbackTitle(conversationId, messages);
    }
  };

  // 오류 시 폴백 제목 생성 함수
  const generateFallbackTitle = (conversationId, messages) => {
    try {
      // 첫 사용자 메시지를 기반으로 제목 생성
      const userMessage = messages.find(msg => msg.role === "user");
      
      if (userMessage) {
        // 첫 질문의 처음 15자를 추출하고 말줄임표 추가
        let title = userMessage.content.trim().substring(0, 15);
        if (userMessage.content.length > 15) {
          title += "...";
        }
        
        // 제목이 너무 짧으면 기본 제목 사용
        if (title.length < 5) {
          const defaultTitle = `대화 ${new Date().toLocaleDateString('ko-KR')}`;
          handleRenameConversation(conversationId, defaultTitle);
        } else {
          // 추출한 제목으로 업데이트
          handleRenameConversation(conversationId, title);
        }
      }
    } catch (error) {
      console.warn("폴백 제목 생성 실패:", error);
      // 최종 폴백: 현재 날짜/시간 기반 제목
      const timestamp = new Date().toLocaleDateString('ko-KR', { 
        month: 'short', 
        day: 'numeric', 
        hour: '2-digit', 
        minute: '2-digit' 
      });
      handleRenameConversation(conversationId, `대화 ${timestamp}`);
    }
  };

  // 모드 전환 함수 - 챗봇 <-> SQL 질의 모드 전환
  const onToggleMode = (mode) => {
    console.log('onToggleMode 호출됨:', mode, '현재 상태:', showSQLPage);
    
    try {
      if (mode === 'sql') {
        console.log('SQL 모드로 전환 중...');
        if (!showSQLPage) {  // 현재 SQL 모드가 아닐 때만 전환
          setShowSQLPage(true);
          // 모드 전환 토스트 메시지 표시
          showModeChangeToast('SQL 질의 모드로 전환했습니다');
        }
      } else if (mode === 'chat') {
        console.log('챗봇 모드로 전환 중...');
        if (showSQLPage) {  // 현재 SQL 모드일 때만 전환
          setShowSQLPage(false);
          // 모드 전환 토스트 메시지 표시
          showModeChangeToast('챗봇 모드로 전환했습니다');
        }
      } else {
        // 모드가 지정되지 않은 경우 토글
        const newMode = !showSQLPage;
        console.log('모드 토글 중...', newMode ? 'SQL로' : '챗봇으로');
        setShowSQLPage(newMode);
        // 모드 전환 토스트 메시지 표시
        showModeChangeToast(newMode ? 'SQL 질의 모드로 전환했습니다' : '챗봇 모드로 전환했습니다');
      }
      console.log('모드 전환 요청 완료. 현재 showSQLPage 상태:', showSQLPage, '(실제 변경은 리렌더링 후 적용됨)');
    } catch (error) {
      console.error('모드 전환 중 오류 발생:', error);
    }
  };

  // 모드 전환 토스트 메시지 표시 함수
  const showModeChangeToast = (message) => {
    console.log('토스트 메시지 표시:', message);
    
    // 기존 토스트가 있으면 제거
    const existingToast = document.getElementById('mode-switch-toast');
    if (existingToast) {
      document.body.removeChild(existingToast);
    }
    
    // 새 토스트 생성
    const toast = document.createElement('div');
    toast.id = 'mode-switch-toast';
    toast.className = 'fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-gradient-to-r from-indigo-600 to-indigo-800 text-white px-5 py-3 rounded-xl shadow-2xl z-[9999] flex items-center gap-3 transition-all duration-150 pointer-events-none';
    toast.style.opacity = '0';
    
    // 토스트 내용 생성
    const iconDiv = document.createElement('div');
    iconDiv.className = 'w-10 h-10 rounded-full bg-white/20 flex items-center justify-center';
    
    // 모드에 따라 다른 아이콘 표시
    const isSqlMode = message.includes('SQL');
    
    const iconSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    iconSvg.setAttribute('viewBox', '0 0 24 24');
    iconSvg.setAttribute('width', '24');
    iconSvg.setAttribute('height', '24');
    iconSvg.setAttribute('fill', 'none');
    iconSvg.setAttribute('stroke', 'currentColor');
    iconSvg.setAttribute('stroke-width', '2');
    iconSvg.setAttribute('stroke-linecap', 'round');
    iconSvg.setAttribute('stroke-linejoin', 'round');
    
    // SQL 또는 챗봇 아이콘 패스 설정
    if (isSqlMode) {
      // Database 아이콘
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', 'M12 2a8 8 0 0 0-8 8v12c0 .6.4 1 1 1h14c.6 0 1-.4 1-1V10a8 8 0 0 0-8-8zm0 0v8m-8 2h16m-8 2v8');
      iconSvg.appendChild(path);
    } else {
      // Message 아이콘
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', 'M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z');
      iconSvg.appendChild(path);
    }
    
    iconDiv.appendChild(iconSvg);
    toast.appendChild(iconDiv);
    
    const messageSpan = document.createElement('span');
    messageSpan.className = 'font-medium';
    messageSpan.textContent = message;
    toast.appendChild(messageSpan);
    
    // 토스트에 애니메이션 효과 추가
    const animDiv = document.createElement('div');
    animDiv.className = 'absolute bottom-0 left-0 h-1 bg-white/30 rounded-b-xl';
    animDiv.style.width = '100%';
    animDiv.style.animation = 'toast-timer 400ms linear forwards';
    toast.appendChild(animDiv);
    
    // 토스트 애니메이션 키프레임 추가
    const styleElement = document.createElement('style');
    styleElement.textContent = `
      @keyframes toast-timer {
        from { width: 100%; }
        to { width: 0%; }
      }
    `;
    document.head.appendChild(styleElement);
    
    // 토스트를 body에 추가
    document.body.appendChild(toast);
    
    // 토스트 표시 애니메이션
    setTimeout(() => {
      toast.style.opacity = "1";
      console.log('토스트 표시됨');
    }, 10);
    
    // 토스트 제거 (0.4초 후)
    setTimeout(() => {
      toast.style.opacity = "0";
      console.log('토스트 숨김 처리');
      setTimeout(() => {
        if (document.body.contains(toast)) {
          document.body.removeChild(toast);
          console.log('토스트 제거됨');
        }
        if (document.head.contains(styleElement)) {
          document.head.removeChild(styleElement);
        }
      }, 150);
    }, 400);
  };

  return (
    <div className="flex flex-col md:flex-row h-screen bg-gray-900 text-gray-100 overflow-hidden relative">
      {/* 임베딩 처리 오버레이 */}
      <EmbeddingOverlay 
        isActive={isEmbedding} 
        status={embeddingStatus} 
        files={embeddedFiles}
      />
      
      {/* 사이드바 */}
      <div
        className={`absolute md:relative inset-y-0 left-0 z-20 transform ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        } md:translate-x-0 transition-transform duration-300 ease-in-out flex-shrink-0 w-${sidebarWidth}px flex flex-col border-r border-gray-800 h-full bg-gray-900`}
        style={{ width: `${sidebarWidth}px` }}
      >
        <Sidebar
          conversations={conversations}
          activeConversationId={activeConversationId}
          onSelectConversation={handleSelectConversation}
          onNewConversation={handleNewConversation}
          onRenameConversation={handleRenameConversation}
          onDeleteConversation={handleDeleteConversation}
          onTogglePinConversation={handleTogglePinConversation}
          onToggleTheme={toggleTheme}
          isDarkMode={theme === "dark"}
          onToggleMode={setMode}
          currentMode={mode}
          onDeleteAllConversations={deleteAllConversations}
        />
        <div
          className="absolute top-0 -right-3 h-full w-3 cursor-ew-resize z-10"
          onMouseDown={handleMouseDown}
        ></div>
      </div>

      {/* 채팅 컨테이너 또는 SQL 쿼리 페이지 */}
      <div className="flex-grow relative overflow-hidden">
        {console.log('렌더링 시점의 mode 값:', mode)}
        {mode === 'sql' ? (
          // SQL 쿼리 페이지 렌더링
          <div className="w-full h-full">
            {console.log('SQL 페이지 렌더링 중...')}
            <SQLQueryPage 
              setMode={setMode} 
              key="sql-page-component" 
            />
          </div>
        ) : (
          // 챗봇 컨테이너 렌더링
          <div className="w-full h-full">
            {console.log('챗봇 컨테이너 렌더링 중...')}
            {!sidebarOpen && (
              <button
                className="absolute left-4 top-4 z-10 md:hidden p-2 rounded-full bg-gray-800 hover:bg-gray-700 text-gray-300 transition-colors"
                onClick={handleToggleSidebar}
              >
                <FiMenu size={20} />
              </button>
            )}
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
              sidebarOpen={sidebarOpen}
              setMode={setMode}
              key="chat-container-component"
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
