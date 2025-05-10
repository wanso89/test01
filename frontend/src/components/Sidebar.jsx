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
    setEditTitle(typeof conv.title === 'string' ? conv.title : '');
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
      .filter((conv) => {
        // title이 문자열이 아닌 경우에 대한 안전한 처리
        const title = typeof conv.title === 'string' ? conv.title : '';
        return title.toLowerCase().includes(searchQuery.toLowerCase());
      })
      .sort((a, b) => {
        // 고정된 대화 우선
        if (a.pinned && !b.pinned) return -1;
        if (!a.pinned && b.pinned) return 1;
        // 그 다음 최신순
        return new Date(b.timestamp || 0) - new Date(a.timestamp || 0);
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
          
          {/* 새 대화 시작 버튼 - 현재 대화가 없을 때만 보이도록 수정 */}
          {conversations.length === 0 && (
            <button
              onClick={onNewConversation}
              className="p-2 bg-gradient-to-r from-indigo-600 to-indigo-500 rounded-xl text-white hover:from-indigo-700 hover:to-indigo-600 transition-all duration-300 shadow-sm hover:shadow transform hover:scale-105 focus:outline-none"
              title="새 대화 시작"
            >
              <FiPlus size={20} />
            </button>
          )}
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

      {/* 푸터 - 다크모드 버튼과 도움말 제거 */}
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
  const [newTitle, setNewTitle] = useState(typeof conversation.title === 'string' ? conversation.title : '');
  const [showMenu, setShowMenu] = useState(false);
  const inputRef = useRef(null);
  const menuRef = useRef(null);

  // 타이틀 편집 시작
  const startEditing = () => {
    setIsEditing(true);
    setShowMenu(false); // 메뉴 닫기
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
    setNewTitle(typeof conversation.title === 'string' ? conversation.title : '');
    setIsEditing(false);
  };
  
  // 메뉴 토글
  const toggleMenu = (e) => {
    e.stopPropagation();
    setShowMenu(!showMenu);
  };
  
  // 메뉴 외부 클릭 감지
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setShowMenu(false);
      }
    };
    
    if (showMenu) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showMenu]);

  return (
    <div 
      className={`relative group transition-all duration-200 ${isEditing || showMenu ? 'z-20' : 'z-10'}`}
      style={{ 
        animation: 'fade-in-right 0.3s ease-out forwards',
        animationDelay: `${animationDelay}s`
      }}
    >
      {isEditing ? (
        // 이름 변경 모드 - 세로로 배치하여 공간 확보
        <div className="p-2 bg-gray-800 rounded-xl border border-gray-700 shadow-lg">
          <form onSubmit={handleSubmit} className="flex flex-col space-y-2">
            <input
              type="text"
              ref={inputRef}
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-2 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400 shadow-sm text-white"
              autoFocus
              placeholder="대화 이름 입력..."
            />
            <div className="flex space-x-2">
              <button
                type="submit"
                className="flex-1 flex items-center justify-center gap-1 py-1.5 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg transition-colors"
              >
                <FiCheck size={14} />
                <span className="text-xs font-medium">확인</span>
              </button>
              <button
                type="button"
                onClick={cancelEditing}
                className="flex-1 flex items-center justify-center gap-1 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-lg transition-colors"
              >
                <FiX size={14} />
                <span className="text-xs font-medium">취소</span>
              </button>
            </div>
          </form>
        </div>
      ) : (
        // 일반 모드
        <div
          className={`flex items-center px-3 py-2.5 rounded-xl cursor-pointer transition-all duration-300 ${
            isActive 
              ? 'bg-gradient-to-r from-indigo-600 to-indigo-500 text-white shadow-md' 
              : 'hover:bg-gray-800/70 text-gray-300'
          }`}
          onClick={onClick}
        >
          <div className="mr-2 text-sm">
            <FiMessageSquare size={16} className={isActive ? 'text-white' : 'text-indigo-400'} />
          </div>
          
          {/* 대화 제목 영역 - 더 넓게 설정 */}
          <div className="flex-1 truncate text-sm mr-2">
            {typeof conversation.title === 'string' ? conversation.title : '제목 없음'}
            {/* 핀 고정된 경우 작은 핀 아이콘 표시 */}
            {conversation.pinned && (
              <span className="ml-1 inline-flex items-center">
                <FiStar size={10} className="fill-current text-yellow-400" />
              </span>
            )}
          </div>
          
          {/* 세로 점 메뉴 버튼 */}
          <div className="relative" ref={menuRef}>
            <button
              onClick={toggleMenu}
              className={`p-1 rounded-lg ${isActive ? 'text-white hover:bg-white/20' : 'text-gray-400 hover:bg-gray-700'} transition-colors`}
              title="메뉴"
            >
              <FiMoreVertical size={16} />
            </button>
            
            {/* 드롭다운 메뉴 - z-index 높임, 위치 조정 */}
            {showMenu && (
              <div className="fixed right-4 mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-lg py-1 w-36 z-50">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onTogglePin();
                    setShowMenu(false);
                  }}
                  className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 flex items-center"
                >
                  <FiStar size={14} className={`mr-2 ${conversation.pinned ? "fill-current text-yellow-400" : "text-gray-400"}`} />
                  {conversation.pinned ? "고정 해제" : "대화 고정"}
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    startEditing();
                    setShowMenu(false);
                  }}
                  className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 flex items-center"
                >
                  <FiEdit2 size={14} className="mr-2 text-gray-400" />
                  이름 변경
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete();
                    setShowMenu(false);
                  }}
                  className="w-full text-left px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 flex items-center"
                >
                  <FiTrash2 size={14} className="mr-2 text-gray-400" />
                  대화 삭제
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default Sidebar;
