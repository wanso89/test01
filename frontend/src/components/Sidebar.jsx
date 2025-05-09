import { useState, useEffect, useMemo, useRef } from "react";
import {
  FiChevronRight,
  FiSearch,
  FiPlus,
  FiMessageSquare,
  FiStar,
  FiEdit2,
  FiTrash2,
  FiX,
  FiClock,
  FiChevronDown,
  FiMoreVertical,
  FiEdit3,
  FiCheck,
  FiArrowRight,
  FiGithub,
  FiInfo,
  FiHelpCircle,
  FiMoon,
  FiSun
} from "react-icons/fi";

function Sidebar({
  collapsed = false,
  conversations = [],
  activeConversationId,
  onNewConversation,
  onDeleteConversation,
  onSelectConversation,
  onRenameConversation,
  onTogglePinConversation,
  onToggleTheme,
  isDarkMode,
}) {
  const [searchQuery, setSearchQuery] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState("");
  const [dropdownMenu, setDropdownMenu] = useState(null);

  // 날짜 포맷팅 함수
  const formatDate = (timestamp) => {
    if (!timestamp) return '';
    
    // timestamp가 숫자 또는 문자열인 경우 처리
    let dateObj;
    if (typeof timestamp === 'number' || !isNaN(parseInt(timestamp))) {
      dateObj = new Date(timestamp);
    } else if (typeof timestamp === 'string') {
      // ISO 형식 문자열 날짜 처리
      dateObj = new Date(timestamp);
    } else {
      return '';
    }
    
    // 올바른 날짜가 아닌 경우 빈 문자열 반환
    if (isNaN(dateObj.getTime())) return '';
    
    const now = new Date();
    const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    // 오늘이면 시간만 표시
    if (dateObj >= today) {
      return dateObj.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: true });
    }
    // 어제면 '어제'로 표시
    else if (dateObj >= yesterday) {
      return '어제';
    }
    // 올해면 월/일만 표시
    else if (dateObj.getFullYear() === now.getFullYear()) {
      return `${dateObj.getMonth() + 1}/${dateObj.getDate()}`;
    }
    // 작년 이전이면 연/월/일 표시
    else {
      return `${dateObj.getFullYear()}/${dateObj.getMonth() + 1}/${dateObj.getDate()}`;
    }
  };

  // 타이틀 편집 시작
  const handleTitleDoubleClick = (conv) => {
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };

  // 타이틀 변경 이벤트
  const handleTitleChange = (e) => {
    setEditTitle(e.target.value);
  };

  // 타이틀 편집 완료 (Enter 또는 blur)
  const handleTitleBlurOrEnter = (conv) => {
    if (editTitle.trim() !== "") {
      onRenameConversation(conv.id, editTitle.trim());
    }
    setEditingId(null);
  };

  // 모바일 메뉴 토글
  const toggleDropdownMenu = (id) => {
    setDropdownMenu(dropdownMenu === id ? null : id);
  };

  // 클릭 외부 감지
  useEffect(() => {
    const handleClickOutside = () => {
      setDropdownMenu(null);
    };
    document.addEventListener("click", handleClickOutside);
    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, []);

  // 대화 검색 필터링
  const filteredConversations = useMemo(() => {
    return conversations
      .filter((conv) =>
        conv.title.toLowerCase().includes(searchQuery.toLowerCase())
      )
      .sort((a, b) => {
        // 고정된 대화 우선
        if (a.pinned && !b.pinned) return -1;
        if (!a.pinned && b.pinned) return 1;
        // 그 다음 최신순
        return new Date(b.timestamp) - new Date(a.timestamp);
      });
  }, [conversations, searchQuery]);

  return (
    <div className="flex flex-col h-full overflow-hidden bg-gray-900">
      {/* 헤더 */}
      <div className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-600 to-indigo-500 flex items-center justify-center shadow-md">
              <FiMessageSquare size={18} className="text-white" />
            </div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-500 to-indigo-400 bg-clip-text text-transparent">
              쓰리에스소프트
            </h1>
          </div>
          <button
            onClick={onNewConversation}
            className="p-2 bg-gradient-to-r from-indigo-100/10 to-indigo-50/10 rounded-xl text-indigo-400 hover:from-indigo-200/20 hover:to-indigo-100/20 transition-all duration-300 shadow-sm hover:shadow transform hover:scale-105 focus:outline-none"
          >
            <FiPlus size={20} />
          </button>
        </div>
      </div>

      {/* 검색 입력 필드 추가 */}
      <div className="px-4 pb-2">
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="대화 검색..."
            className="w-full bg-gray-800/50 border border-gray-700 rounded-lg py-2 pl-9 pr-3 text-sm text-gray-300 placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
          />
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <FiSearch className="h-4 w-4 text-gray-500" />
          </div>
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-500 hover:text-gray-300"
            >
              <FiX className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>

      {/* 대화 목록 */}
      <div className="flex-1 overflow-hidden px-3 relative">
        {/* 장식 요소 - 시각적인 흥미 추가 */}
        <div 
          className="absolute -right-8 top-20 w-16 h-16 rounded-full bg-indigo-200 opacity-20"
          style={{ filter: 'blur(20px)' }}
        ></div>
        <div 
          className="absolute -left-8 bottom-40 w-20 h-20 rounded-full bg-indigo-300 opacity-10"
          style={{ filter: 'blur(25px)' }}
        ></div>
        
        {/* 스크롤 영역 */}
        <div 
          className="overflow-y-auto h-full pr-1 custom-scrollbar relative z-10"
          style={{
            maskImage: 'linear-gradient(to bottom, transparent, black 10px, black calc(100% - 10px), transparent)',
            WebkitMaskImage: 'linear-gradient(to bottom, transparent, black 10px, black calc(100% - 10px), transparent)'
          }}
        >
          {/* 핀된 대화 목록 */}
          {filteredConversations.some(conv => conv.pinned) && (
            <div className="mb-4">
              <div className="flex items-center px-2 py-1.5 text-xs text-indigo-400 font-medium">
                <FiStar size={12} className="mr-1.5" />
                <span>핀 고정</span>
              </div>
              <div className="space-y-1.5 mt-1.5">
                {filteredConversations.filter(conv => conv.pinned).map((conversation, index) => (
                  <ConversationItem
                    key={conversation.id}
                    conversation={conversation}
                    isActive={conversation.id === activeConversationId}
                    onClick={() => onSelectConversation(conversation.id)}
                    onRename={(newTitle) => onRenameConversation(conversation.id, newTitle)}
                    onDelete={() => onDeleteConversation(conversation.id)}
                    onTogglePin={() => onTogglePinConversation(conversation.id)}
                    animationDelay={index * 0.05}
                  />
                ))}
              </div>
                  </div>
                )}

          {/* 최근 대화 목록 */}
          <div>
            <div className="flex items-center px-2 py-1.5 text-xs text-gray-500 font-medium">
              <FiClock size={12} className="mr-1.5" />
              <span>최근 대화</span>
            </div>
            <div className="space-y-1.5 mt-1.5">
              {filteredConversations.length === 0 ? (
                <div className="px-3 py-6 text-center">
                  <div className="text-gray-400 text-sm">대화 기록이 없습니다</div>
                  <button
                    onClick={onNewConversation}
                    className="mt-2 px-3 py-1.5 bg-indigo-900/30 text-indigo-400 text-sm rounded-lg hover:bg-indigo-800/40 transition-colors inline-flex items-center"
                  >
                    <FiPlus size={14} className="mr-1" />
                    새 대화 시작하기
                  </button>
                </div>
              ) : (
                filteredConversations.filter(conv => !conv.pinned).map((conversation, index) => (
                  <ConversationItem
                    key={conversation.id}
                    conversation={conversation}
                    isActive={conversation.id === activeConversationId}
                    onClick={() => onSelectConversation(conversation.id)}
                    onRename={(newTitle) => onRenameConversation(conversation.id, newTitle)}
                    onDelete={() => onDeleteConversation(conversation.id)}
                    onTogglePin={() => onTogglePinConversation(conversation.id)}
                    animationDelay={index * 0.05}
                  />
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 푸터 - 수정된 부분 (다크모드, 도움말 제거) */}
      <div className="p-3 bg-gray-900">
        <div className="flex flex-col space-y-3">
          {/* 버전 정보만 남김 */}
          <div className="mt-2 flex items-center justify-center text-xs text-gray-500">
            <span>3Ssoft Chatbot v1</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// 대화 항목 컴포넌트 업데이트
function ConversationItem({ conversation, isActive, onClick, onRename, onDelete, onTogglePin, animationDelay }) {
  const [isEditing, setIsEditing] = useState(false);
  const [newTitle, setNewTitle] = useState(conversation.title);
  const inputRef = useRef(null);

  // 타이틀 편집 시작
  const startEditing = () => {
    setIsEditing(true);
  };

  // 타이틀 편집 완료 (Enter 또는 blur)
  const handleSubmit = (e) => {
    e.preventDefault();
    if (newTitle.trim() !== "") {
      onRename(newTitle.trim());
    }
    setIsEditing(false);
  };

  // 타이틀 편집 취소
  const cancelEditing = () => {
    setNewTitle(conversation.title);
    setIsEditing(false);
  };

  return (
    <div 
      className={`relative group transition-all duration-200 ${isEditing ? 'z-20' : 'z-10'}`}
      style={{ 
        animation: 'fade-in-right 0.3s ease-out forwards',
        animationDelay: `${animationDelay}s`
      }}
    >
      <div
        className={`flex items-center px-3 py-2.5 rounded-xl cursor-pointer transition-all duration-300 ${
          isActive 
            ? 'bg-gradient-to-r from-indigo-600 to-indigo-500 text-white shadow-md' 
            : 'hover:bg-gray-800/70 text-gray-300'
        }`}
        onClick={isEditing ? undefined : onClick}
      >
        <div className="mr-2 text-sm">
          <FiMessageSquare size={16} className={isActive ? 'text-white' : 'text-indigo-400'} />
        </div>
        
        {isEditing ? (
          <form onSubmit={handleSubmit} className="flex-1 flex">
            <input
              type="text"
              ref={inputRef}
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 shadow-sm text-white"
              autoFocus
            />
            <div className="flex ml-1">
              <button
                type="submit"
                className="p-1 bg-green-800/70 text-green-400 rounded-lg hover:bg-green-700/70"
              >
                <FiCheck size={16} />
              </button>
              <button
                type="button"
                onClick={cancelEditing}
                className="p-1 bg-gray-800/70 text-gray-400 rounded-lg ml-1 hover:bg-gray-700/70"
              >
                <FiX size={16} />
              </button>
            </div>
          </form>
        ) : (
          <>
            <div className="flex-1 truncate text-sm">
              {conversation.title}
            </div>
            
            <div className={`flex space-x-1 ${isActive ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'} transition-opacity`}>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onTogglePin();
                }}
                className={`p-1 rounded-lg ${isActive ? 'text-white hover:bg-white/20' : 'text-gray-400 hover:bg-gray-700'} transition-colors`}
                title={conversation.pinned ? "핀 제거" : "대화 핀 고정"}
              >
                <FiStar size={14} className={conversation.pinned ? "fill-current" : ""} />
              </button>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  startEditing();
                }}
                className={`p-1 rounded-lg ${isActive ? 'text-white hover:bg-white/20' : 'text-gray-400 hover:bg-gray-700'} transition-colors`}
                title="대화 이름 변경"
              >
                <FiEdit2 size={14} />
              </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                  onDelete(); // 확인 창 없이 바로 삭제
                      }}
                className={`p-1 rounded-lg ${isActive ? 'text-white hover:bg-white/20' : 'text-gray-400 hover:bg-gray-700'} transition-colors`}
                      title="대화 삭제"
                    >
                      <FiTrash2 size={14} />
                    </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export default Sidebar;
