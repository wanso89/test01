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
  FiChevronUp,
  FiDatabase,
  FiTerminal,
  FiCode,
  FiMessageCircle,
  FiServer,
  FiCommand,
  FiAlertCircle,
  FiTrash
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
  onToggleMode,
  currentMode = 'chat', // 현재 선택된 모드 prop 추가
  onDeleteAllConversations // 전체 대화 삭제 추가
}) {
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState("");
  const [dropdownMenu, setDropdownMenu] = useState(null);
  // 대화 목록 접기/펼치기 상태
  const [showAllConversations, setShowAllConversations] = useState(false);

  // 전체 대화 삭제 모달 상태
  const [showDeleteAllModal, setShowDeleteAllModal] = useState(false);

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

  // 대화 필터링 및 정렬
  const sortedConversations = useMemo(() => {
    return [...conversations].sort((a, b) => {
      // 고정된 대화 우선
      if (a.pinned && !b.pinned) return -1;
      if (!a.pinned && b.pinned) return 1;
      // 그 다음 최신순
      return new Date(b.timestamp || 0) - new Date(a.timestamp || 0);
    });
  }, [conversations]);

  // 고정된 대화와 일반 대화 분리
  const pinnedConversations = useMemo(() => {
    return sortedConversations.filter(conv => conv.pinned);
  }, [sortedConversations]);

  const unpinnedConversations = useMemo(() => {
    return sortedConversations.filter(conv => !conv.pinned);
  }, [sortedConversations]);

  // 표시할 일반 대화 수 제한 (9개)
  const MAX_VISIBLE_CONVERSATIONS = 9;
  const visibleUnpinnedConversations = showAllConversations 
    ? unpinnedConversations 
    : unpinnedConversations.slice(0, MAX_VISIBLE_CONVERSATIONS);

  return (
    <div className="flex flex-col h-full overflow-hidden bg-gray-900">
      {/* 브랜딩 헤더 - 로고만 표시 */}
      <div className="h-16 py-2 px-4 flex items-center justify-center border-b border-gray-800">
        <Logo />
      </div>

      {/* 새 대화 버튼 & 전체 삭제 버튼 */}
      <div className="px-4 py-2.5 border-b border-gray-800/50">
        <div className="flex justify-between items-center">
          <button
            onClick={() => setShowDeleteAllModal(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-red-400 hover:text-red-300 bg-transparent hover:bg-red-950/50 rounded-md transition-all duration-200 text-sm"
            title="모든 대화 삭제"
          >
            <FiTrash size={15} />
            <span>전체 삭제</span>
          </button>
          
          <button
            onClick={onNewConversation}
            className="flex items-center gap-1.5 px-3 py-1.5 text-gray-300 hover:text-white bg-transparent hover:bg-gray-800/70 rounded-md transition-all duration-200 text-sm"
            title="새 대화 시작하기"
          >
            <FiPlus size={15} className="text-gray-400" />
            <span>새 대화</span>
          </button>
        </div>
      </div>

      {/* 대화 목록 스크롤 영역 */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden py-2 custom-scrollbar">
        {/* 대화 목록이 비어있을 때 */}
        {conversations.length === 0 ? (
          <div className="px-4 py-4 text-center text-gray-500 text-sm">
            대화 내역이 없습니다.
          </div>
        ) : (
          <div className="space-y-4">
            {/* 핀 고정된 대화 목록 */}
            {pinnedConversations.length > 0 && (
              <div className="mb-2">
                <div className="flex items-center px-3 py-1.5 text-xs text-indigo-400 font-medium">
                  <FiStar size={12} className="mr-1.5" />
                  <span>핀 고정</span>
                </div>
                <div className="space-y-1.5 mt-1.5 px-2">
                  {pinnedConversations.map((conv) => (
                    <ConversationItem
                      key={conv.id}
                      conversation={conv}
                      isActive={conv.id === activeConversationId}
                      onClick={() => onSelectConversation(conv.id)}
                      onRename={(newTitle) => onRenameConversation(conv.id, newTitle)}
                      onDelete={() => onDeleteConversation(conv.id)}
                      onTogglePin={() => onTogglePinConversation(conv.id)}
                      animationDelay={0}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* 일반 대화 목록 */}
            <div>
              <div className="flex items-center justify-between px-3 py-1.5">
                <div className="flex items-center text-xs text-gray-500 font-medium">
                  <FiClock size={12} className="mr-1.5" />
                  <span>최근 대화</span>
                </div>
                {unpinnedConversations.length > MAX_VISIBLE_CONVERSATIONS && (
                  <button 
                    onClick={() => setShowAllConversations(!showAllConversations)}
                    className="text-xs text-gray-500 hover:text-gray-300 flex items-center"
                  >
                    {showAllConversations ? (
                      <>
                        <FiChevronUp size={14} className="mr-1" />
                        <span>접기</span>
                      </>
                    ) : (
                      <>
                        <FiChevronDown size={14} className="mr-1" />
                        <span>더보기 ({unpinnedConversations.length - MAX_VISIBLE_CONVERSATIONS})</span>
                      </>
                    )}
                  </button>
                )}
              </div>
              <div className="space-y-1.5 mt-1.5 px-2">
                {visibleUnpinnedConversations.map((conv) => (
                  <ConversationItem
                    key={conv.id}
                    conversation={conv}
                    isActive={conv.id === activeConversationId}
                    onClick={() => onSelectConversation(conv.id)}
                    onRename={(newTitle) => onRenameConversation(conv.id, newTitle)}
                    onDelete={() => onDeleteConversation(conv.id)}
                    onTogglePin={() => onTogglePinConversation(conv.id)}
                    animationDelay={0}
                  />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 전체 대화 삭제 확인 모달 */}
      {showDeleteAllModal && (
        <div className="fixed inset-0 bg-gray-900/90 backdrop-blur-sm flex items-center justify-center z-[70] animate-fade-in">
          <div className="bg-gray-800 rounded-xl p-5 max-w-md w-full border border-gray-700 shadow-2xl animate-slide-up">
            <div className="flex items-start mb-4">
              <div className="flex-shrink-0 text-red-500 mr-3">
                <FiAlertCircle size={24} />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white mb-2">
                  모든 대화 삭제
                </h3>
                <p className="text-gray-300 text-sm">
                  모든 대화 내역이 영구적으로 삭제됩니다. 이 작업은 되돌릴 수 없습니다.
                </p>
                <p className="mt-2 text-red-400 text-xs">
                  정말 모든 대화를 삭제하시겠습니까?
                </p>
              </div>
            </div>
            <div className="flex justify-end gap-3 mt-4">
              <button
                onClick={() => setShowDeleteAllModal(false)}
                className="px-4 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
              >
                취소
              </button>
              <button
                onClick={() => {
                  if (onDeleteAllConversations) {
                    onDeleteAllConversations();
                  }
                  setShowDeleteAllModal(false);
                }}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2"
              >
                <FiTrash size={16} />
                삭제 확인
              </button>
            </div>
          </div>
        </div>
      )}
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
              <div className="absolute right-0 mt-1 bg-gray-800 border border-gray-700 rounded-lg shadow-lg py-1 w-36 z-50">
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
