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
  FiSun,
  FiMenu,
  FiChevronUp
} from "react-icons/fi";
import { LOGO_IMAGE, createLogoIcon } from "../assets/3ssoft-logo.js";

// 로고 컴포넌트 - 이미지 로드 상태를 관리
const Logo = () => {
  const [imageLoaded, setImageLoaded] = useState(true);
  const [isHovered, setIsHovered] = useState(false);
  
  const handleImageError = () => {
    // 콘솔 오류 메시지 제거
    setImageLoaded(false);
  };
  
  // 외부 이미지가 로드되지 않을 경우 대체 URL
  const fallbackLogoUrl = "https://3ssoft.co.kr/wp-content/uploads/2023/06/cropped-logo-300x104.png";
  
  return (
    <div 
      className="flex flex-col items-center justify-center w-full"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {imageLoaded ? (
        <div className="flex flex-col items-center">
          <img 
            src={LOGO_IMAGE || fallbackLogoUrl}
            alt="3S소프트 로고" 
            className={`w-32 h-auto ${isHovered ? 'brightness-110' : ''} transition-all duration-200`}
            onError={handleImageError}
            style={{ objectFit: 'contain' }}
          />
          <p className="text-[8px] text-gray-400 mt-0 text-center tracking-tight opacity-80" style={{ marginTop: '-3px', letterSpacing: '-0.02em', paddingLeft: '15px' }}>
            Server consulting, Solution & Systems software
          </p>
        </div>
      ) : (
        <div className="flex flex-col items-center">
          <div className="flex items-center justify-center">
            <div 
              className="h-8 w-8 flex items-center justify-center mr-2"
              dangerouslySetInnerHTML={{ __html: createLogoIcon('#0B3C71', 32) }} 
            />
            <div className="flex items-baseline">
              <span className="font-bold text-xl tracking-tight" style={{ fontFamily: "'Arial', 'Helvetica', sans-serif", color: '#0A2F65', letterSpacing: '-0.02em' }}>3S</span>
              <span className="font-bold text-xl tracking-tight" style={{ fontFamily: "'Arial', 'Helvetica', sans-serif", color: '#4A4A4A', letterSpacing: '-0.02em' }}>소프트</span>
            </div>
          </div>
          <p className="text-[8px] text-gray-400 mt-0 text-center tracking-tight opacity-80" style={{ marginTop: '1px', letterSpacing: '-0.02em', paddingLeft: '18px' }}>
            Server consulting, Solution & Systems software
          </p>
        </div>
      )}
    </div>
  );
};

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
      {/* 브랜딩 헤더 - 로고만 표시 */}
      <div className="h-16 py-2 px-4 flex items-center justify-center border-b border-gray-800">
        <Logo />
      </div>

      {/* 새 대화 버튼 */}
      <div className="px-4 py-2.5 border-b border-gray-800/50">
        <button
          onClick={onNewConversation}
          className="flex items-center gap-1.5 px-3 py-1.5 text-gray-300 hover:text-white bg-transparent hover:bg-gray-800/70 rounded-md transition-all duration-200 text-sm ml-auto float-right"
        >
          <FiPlus size={15} className="text-gray-400" />
          <span>새 대화</span>
        </button>
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

          {/* 최근 대화 목록 - 접기/펼치기 기능 추가 */}
          <ConversationGroup
            title="최근 대화"
            icon={<FiClock size={12} className="mr-1.5" />}
            conversations={filteredConversations.filter(conv => !conv.pinned)}
            activeConversationId={activeConversationId}
            onSelect={onSelectConversation}
            onRename={onRenameConversation}
            onDelete={onDeleteConversation}
            onTogglePin={onTogglePinConversation}
            maxVisible={6}
          />
        </div>
      </div>

      {/* 푸터 - 유용한 정보 추가 */}
      <div className="p-4 bg-gray-800/30 rounded-lg mx-3 mb-3 shadow-inner">
        <div className="flex flex-col">
          {/* 유용한 단축키 */}
          <div>
            <div className="text-xs font-medium text-indigo-400 mb-2.5 flex items-center">
              <FiHelpCircle size={12} className="mr-1.5" />
              <span>유용한 단축키</span>
            </div>
            <div className="grid grid-cols-2 gap-y-2 gap-x-3 text-xs text-gray-300">
              <div className="flex items-center">
                <span className="inline-flex items-center justify-center min-w-[24px] h-5 px-1 mr-1.5 bg-gray-700/70 rounded text-indigo-300 font-mono">↑</span>
                <span>이전 질문</span>
              </div>
              <div className="flex items-center">
                <span className="inline-flex items-center justify-center min-w-[24px] h-5 px-1 mr-1.5 bg-gray-700/70 rounded text-indigo-300 font-mono">↓</span>
                <span>다음 질문</span>
              </div>
              <div className="flex items-center">
                <span className="inline-flex items-center justify-center min-w-[24px] h-5 px-1 mr-1.5 bg-gray-700/70 rounded text-indigo-300 font-mono">Tab</span>
                <span>자동 완성</span>
              </div>
              <div className="flex items-center">
                <span className="inline-flex items-center justify-center min-w-[24px] h-5 px-1 mr-1.5 bg-gray-700/70 rounded text-indigo-300 font-mono">Esc</span>
                <span>입력 취소</span>
              </div>
            </div>
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

// 대화 그룹 컴포넌트 - 접기/펼치기 기능을 가진 대화 목록
function ConversationGroup({ title, icon, conversations, activeConversationId, onSelect, onRename, onDelete, onTogglePin, maxVisible = 6 }) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [showAll, setShowAll] = useState(false);
  
  // 보여줄 대화 목록 결정
  const visibleConversations = showAll
    ? conversations
    : conversations.slice(0, maxVisible);
    
  // 더 보기 버튼이 필요한지 여부
  const needsMoreButton = conversations.length > maxVisible;
  
  // 대화 목록이 비어있으면 빈 UI 반환
  if (conversations.length === 0) {
    return (
      <div className="mb-3">
        <div className="flex items-center justify-between px-2 py-1.5">
          <div className="flex items-center text-xs text-gray-500 font-medium">
            {icon}
            <span>{title}</span>
          </div>
        </div>
        <div className="px-3 py-6 text-center">
          <div className="text-gray-400 text-sm">대화 기록이 없습니다</div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="mb-3">
      {/* 그룹 헤더 - 접기/펼치기 토글 버튼 */}
      <div className="flex items-center justify-between px-2 py-1.5 cursor-pointer" onClick={() => setIsCollapsed(!isCollapsed)}>
        <div className="flex items-center text-xs text-gray-500 font-medium">
          {icon}
          <span>{title}</span>
        </div>
        <button className="text-gray-500 hover:text-gray-300 p-0.5">
          {isCollapsed ? <FiChevronRight size={14} /> : <FiChevronDown size={14} />}
        </button>
      </div>
      
      {/* 대화 목록 */}
      {!isCollapsed && (
        <div className="space-y-1.5 mt-1.5">
          {/* 대화 항목들 */}
          {visibleConversations.map((conversation, index) => (
            <ConversationItem
              key={conversation.id}
              conversation={conversation}
              isActive={conversation.id === activeConversationId}
              onClick={() => onSelect(conversation.id)}
              onRename={(newTitle) => onRename(conversation.id, newTitle)}
              onDelete={() => onDelete(conversation.id)}
              onTogglePin={() => onTogglePin(conversation.id)}
              animationDelay={index * 0.05}
            />
          ))}
          
          {/* 더 보기/접기 버튼 */}
          {needsMoreButton && (
            <button
              onClick={() => setShowAll(!showAll)}
              className="w-full flex items-center justify-center py-1.5 px-3 mt-2 text-xs text-gray-400 hover:text-gray-300 hover:bg-gray-800/50 rounded-lg transition-colors"
            >
              {showAll ? (
                <>
                  <FiChevronUp size={14} className="mr-1.5" />
                  <span>접기</span>
                </>
              ) : (
                <>
                  <FiChevronDown size={14} className="mr-1.5" />
                  <span>더 보기 ({conversations.length - maxVisible})</span>
                </>
              )}
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default Sidebar;
