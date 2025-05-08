import { useState, useEffect } from 'react';
import { 
  FiPlus, FiTrash2, FiStar, FiChevronDown, FiSearch, 
  FiMessageSquare, FiX, FiEdit2, FiCheck, FiMenu, FiServer
} from 'react-icons/fi';

function Sidebar({
  collapsed,
  conversations,
  activeConversationId,
  onNewConversation,
  onDeleteConversation,
  onSelectConversation,
  onRenameConversation,
  onTogglePinConversation
}) {
  // 제목 편집 상태
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  // 검색 상태
  const [searchTerm, setSearchTerm] = useState('');
  // 더보기 상태 (대화 목록이 많을 경우 제한적으로 표시)
  const [showMore, setShowMore] = useState(false);
  const visibleLimit = 15; // 초기 표시 대화 수 제한 증가
  // 그룹화된 대화 목록 (날짜별)
  const [groupedConversations, setGroupedConversations] = useState({});

  // 즐겨찾기 고정: pinned=true인 대화가 상단
  const sortedConversations = [...conversations].sort((a, b) => {
    if (a.pinned === b.pinned) {
      // 최신순 정렬
      return new Date(b.timestamp) - new Date(a.timestamp);
    }
    return a.pinned ? -1 : 1;
  });

  // 검색어 필터링
  const filteredConversations = sortedConversations.filter(conv =>
    conv.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // 표시할 대화 목록 (더보기 여부에 따라 제한)
  const visibleConversations = showMore ? filteredConversations : filteredConversations.slice(0, visibleLimit);

  // 날짜별로 대화 그룹화
  useEffect(() => {
    if (!filteredConversations.length) return;
    
    const grouped = filteredConversations.reduce((acc, conv) => {
      // 오늘, 어제, 이번 주, 이번 달, 더 오래된 대화 등으로 분류
      const date = new Date(conv.timestamp);
      const now = new Date();
      const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
      const yesterday = new Date(today);
      yesterday.setDate(yesterday.getDate() - 1);
      
      let group = '이전 대화';
      
      if (date >= today) {
        group = '오늘';
      } else if (date >= yesterday) {
        group = '어제';
      } else if (date.getMonth() === now.getMonth() && date.getFullYear() === now.getFullYear()) {
        group = '이번 달';
      }
      
      if (!acc[group]) {
        acc[group] = [];
      }
      acc[group].push(conv);
      return acc;
    }, {});
    
    setGroupedConversations(grouped);
  }, [filteredConversations]);

  // 제목 더블클릭 → 편집모드
  const handleTitleDoubleClick = (conv) => {
    setEditingId(conv.id);
    setEditTitle(conv.title);
  };
  
  const handleTitleChange = (e) => setEditTitle(e.target.value);
  
  const handleTitleBlurOrEnter = (conv) => {
    if (editTitle.trim() && editTitle !== conv.title) {
      onRenameConversation(conv.id, editTitle.trim());
    }
    setEditingId(null);
  };

  // 검색어 입력 핸들러
  const handleSearchChange = (e) => setSearchTerm(e.target.value);
  const clearSearch = () => setSearchTerm('');

  return (
    <div className={`flex flex-col h-full ${collapsed ? 'w-16' : 'w-72'} custom-transition-all duration-300 
      border-r border-slate-200 dark:border-slate-800 
      bg-gradient-to-b from-gray-50 to-slate-100 dark:from-slate-900 dark:to-slate-950
      shadow-md`}>
      
      {/* 헤더 섹션 */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-200 dark:border-slate-800 
        bg-white dark:bg-slate-900 shadow-sm">
        {!collapsed && (
          <div className="flex items-center gap-2">
            <FiServer className="text-blue-500 dark:text-blue-400" size={20} />
            <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-200">RAG 챗봇</h2>
          </div>
        )}
        
        <button 
          onClick={onNewConversation}
          className="flex items-center justify-center h-9 w-9 rounded-lg 
            bg-blue-500 hover:bg-blue-600 text-white shadow-sm custom-transition-colors"
          aria-label="새 대화"
        >
          <FiPlus className="w-5 h-5" />
        </button>
      </div>
      
      {/* 검색 필드 */}
      {!collapsed && (
        <div className="px-4 py-3 border-b border-slate-200 dark:border-slate-800">
          <div className="relative">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <FiSearch className="h-4 w-4 text-slate-400" />
            </div>
            <input
              type="text"
              value={searchTerm}
              onChange={handleSearchChange}
              placeholder="대화 검색..."
              className="w-full pl-10 pr-10 py-2 text-sm border border-slate-200 dark:border-slate-700 
                rounded-lg bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200
                placeholder-slate-400 shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            {searchTerm && (
              <button 
                onClick={clearSearch}
                className="absolute inset-y-0 right-0 pr-3 flex items-center text-slate-400 hover:text-slate-600 dark:hover:text-slate-300"
              >
                <FiX className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      )}
      
      {/* 대화 목록 */}
      <nav className="flex-1 px-2 py-2 overflow-y-auto custom-scrollbar space-y-1">
        {filteredConversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center py-10">
            <div className="w-16 h-16 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center mb-4">
              <FiMessageSquare className="w-7 h-7 text-slate-400 dark:text-slate-500" />
            </div>
            <p className="text-slate-500 dark:text-slate-400 text-sm px-4">
              {searchTerm ? "검색 결과가 없습니다." : "대화가 없습니다. 새 대화를 시작하세요."}
            </p>
            {!searchTerm && (
              <button
                onClick={onNewConversation}
                className="mt-4 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm font-medium custom-transition-colors"
              >
                <span className="flex items-center gap-2">
                  <FiPlus size={16} />
                  새 대화 시작
                </span>
              </button>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {!collapsed && Object.entries(groupedConversations).map(([group, convs]) => (
              <div key={group} className="space-y-1 mb-3">
                <h3 className="text-xs font-medium text-slate-500 dark:text-slate-400 px-3 py-2 uppercase tracking-wider">{group}</h3>
                {convs.slice(0, showMore ? undefined : visibleLimit).map(conv => (
                  <div
                    key={conv.id}
                    className={`
                      flex items-center gap-2 px-3 py-2.5 rounded-lg cursor-pointer custom-transition-all 
                      group relative overflow-hidden
                      ${activeConversationId === conv.id
                        ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-900 dark:text-blue-100 font-medium'
                        : 'hover:bg-slate-200/70 dark:hover:bg-slate-800/60 text-slate-800 dark:text-slate-200'}
                    `}
                    onClick={() => onSelectConversation(conv.id)}
                  >
                    <div className="flex-shrink-0">
                      {conv.pinned ? 
                        <FiStar size={18} className={`${activeConversationId === conv.id ? 'text-yellow-500' : 'text-yellow-500'} fill-current`} /> : 
                        <FiMessageSquare size={18} className={activeConversationId === conv.id ? 'text-blue-600 dark:text-blue-400' : 'text-slate-400 dark:text-slate-500'} />
                      }
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      {editingId === conv.id ? (
                        <input
                          className="w-full px-2 py-1 text-sm rounded-md border border-blue-300 dark:border-blue-700 
                            bg-white dark:bg-slate-800 text-slate-800 dark:text-slate-200 
                            focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm"
                          value={editTitle}
                          autoFocus
                          onChange={handleTitleChange}
                          onBlur={() => handleTitleBlurOrEnter(conv)}
                          onKeyDown={e => {
                            if (e.key === 'Enter') handleTitleBlurOrEnter(conv);
                            if (e.key === 'Escape') setEditingId(null);
                          }}
                        />
                      ) : (
                        <>
                          <div
                            className="text-sm truncate"
                            onDoubleClick={() => handleTitleDoubleClick(conv)}
                            tabIndex={0}
                            title={conv.title}
                          >
                            {conv.title}
                          </div>
                          
                          {/* 타임스탬프 */}
                          <div className="text-xs text-slate-500 dark:text-slate-400 mt-0.5 truncate">
                            {new Date(conv.timestamp).toLocaleDateString()}
                          </div>
                        </>
                      )}
                    </div>
                    
                    {/* 액션 버튼들 - 투명 배경에서 호버 시 보이게 */}
                    <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 
                      absolute right-2 custom-transition-opacity">
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          handleTitleDoubleClick(conv);
                        }}
                        className="p-1.5 rounded-md text-slate-500 hover:text-slate-700 
                          hover:bg-slate-300/50 dark:hover:bg-slate-700/50 dark:hover:text-slate-300 custom-transition-colors"
                        aria-label="대화 이름 변경"
                      >
                        <FiEdit2 size={14} />
                      </button>
                      
                      <button
                        onClick={e => {
                          e.stopPropagation();
                          onTogglePinConversation(conv.id);
                        }}
                        className={`p-1.5 rounded-md hover:bg-slate-300/50 dark:hover:bg-slate-700/50 custom-transition-colors
                          ${conv.pinned ? 'text-yellow-500' : 'text-slate-500 hover:text-yellow-500 dark:hover:text-yellow-400'}`}
                        aria-label={conv.pinned ? "즐겨찾기 해제" : "즐겨찾기"}
                      >
                        <FiStar size={14} className={conv.pinned ? 'fill-current' : ''} />
                      </button>
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (window.confirm('이 대화를 삭제하시겠습니까?')) {
                            onDeleteConversation(conv.id);
                          }
                        }}
                        className="p-1.5 rounded-md text-slate-500 hover:text-red-500 
                          hover:bg-red-100 dark:hover:bg-red-900/30 dark:hover:text-red-400 custom-transition-colors"
                        aria-label="대화 삭제"
                      >
                        <FiTrash2 size={14} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ))}
            
            {/* 간소화된 버전 (모바일/접힌 상태) */}
            {collapsed && visibleConversations.map(conv => (
              <div
                key={conv.id}
                className={`
                  flex justify-center p-2.5 mb-1 rounded-lg cursor-pointer custom-transition-colors
                  ${activeConversationId === conv.id
                    ? 'bg-blue-100 dark:bg-blue-900/30'
                    : 'hover:bg-slate-200 dark:hover:bg-slate-800/70'}
                `}
                onClick={() => onSelectConversation(conv.id)}
                title={conv.title}
              >
                {conv.pinned ? 
                  <FiStar size={20} className="text-yellow-500 fill-current" /> : 
                  <FiMessageSquare size={20} className={activeConversationId === conv.id 
                    ? 'text-blue-600 dark:text-blue-400' 
                    : 'text-slate-500 dark:text-slate-400'} />
                }
              </div>
            ))}
          </div>
        )}
      </nav>
      
      {/* 더보기 버튼 */}
      {!collapsed && filteredConversations.length > visibleLimit && !showMore && (
        <div className="p-3 border-t border-slate-200 dark:border-slate-800">
          <button
            onClick={() => setShowMore(true)}
            className="w-full py-2 px-4 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 
              text-slate-600 dark:text-slate-300 text-sm font-medium rounded-lg 
              custom-transition-colors flex items-center justify-center shadow-sm"
          >
            <FiChevronDown size={16} className="mr-1 text-slate-500" />
            더보기 ({filteredConversations.length - visibleLimit}개)
          </button>
        </div>
      )}
    </div>
  );
}

export default Sidebar;
