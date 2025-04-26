import { useState } from 'react';
import { FiPlus, FiTrash2, FiStar } from 'react-icons/fi';

function Sidebar({
  collapsed,
  conversations,
  activeConversationId,
  onNewConversation,
  onDeleteConversation,
  onSelectConversation,
  onRenameConversation,
  onTogglePinConversation // ⭐️ 추가!
}) {
  // 제목 편집 상태
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState('');
  // 검색 상태
  const [searchTerm, setSearchTerm] = useState('');

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

  // ⭐️ 즐겨찾기 고정: pinned=true인 대화가 상단
  const sortedConversations = [...conversations].sort((a, b) => {
    if (a.pinned === b.pinned) {
      // 최신순 정렬
      return b.timestamp.localeCompare(a.timestamp);
    }
    return a.pinned ? -1 : 1;
  });

  // 검색어 필터링
  const filteredConversations = sortedConversations.filter(conv =>
    conv.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className={`flex flex-col h-full ${collapsed ? 'items-center' : ''} transition-all`}>
      <div className={`p-4 border-b w-full ${collapsed ? 'text-center' : ''}`}>
        <h2 className={`font-bold text-lg text-gray-700 dark:text-white ${collapsed ? 'text-xs' : ''}`}>대화</h2>
        {!collapsed && (
          <input
            type="text"
            value={searchTerm}
            onChange={handleSearchChange}
            placeholder="대화 검색..."
            className="mt-2 w-full px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 transition"
            style={{ fontSize: '14px' }}
          />
        )}
      </div>
      <nav className={`flex-1 p-4 w-full space-y-2 ${collapsed ? 'px-1' : ''} overflow-y-auto`}>
        {filteredConversations.length === 0 ? (
          <div className="text-gray-400 text-sm text-center mt-8">검색 결과가 없습니다.</div>
        ) : (
          filteredConversations.map(conv => (
            <div
              key={conv.id}
              className={`
                flex justify-between items-center p-2 rounded-xl cursor-pointer transition
                ${activeConversationId === conv.id
                  ? 'bg-gradient-to-r from-blue-100 to-blue-200 dark:from-blue-900 dark:to-blue-800 text-blue-800 dark:text-blue-200 font-semibold'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'}
              `}
              onClick={() => onSelectConversation(conv.id)}
              title={collapsed ? conv.title : ''}
            >
              <div className="flex-1 overflow-hidden">
                {editingId === conv.id ? (
                  <input
                    className="text-sm font-bold w-full px-1 py-0.5 rounded border focus:outline-none"
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
                  <div
                    className={`text-sm ${collapsed ? 'text-xs truncate' : ''} font-bold cursor-pointer`}
                    onDoubleClick={() => handleTitleDoubleClick(conv)}
                    tabIndex={0}
                    title="더블클릭해서 제목 편집"
                  >
                    {conv.title}
                  </div>
                )}
                {!collapsed && <div className="text-xs text-gray-500 dark:text-gray-400 truncate">{conv.timestamp}</div>}
              </div>
              {/* ⭐️ 즐겨찾기 버튼 */}
              <button
                onClick={e => {
                  e.stopPropagation();
                  onTogglePinConversation(conv.id);
                }}
                className={`mr-1 ${conv.pinned ? 'text-yellow-400' : 'text-gray-400'} hover:text-yellow-500 transition`}
                title={conv.pinned ? "즐겨찾기 해제" : "즐겨찾기"}
                tabIndex={-1}
                style={{ background: 'none', border: 'none' }}
              >
                <FiStar fill={conv.pinned ? '#facc15' : 'none'} size={18} />
              </button>
              {/* 삭제 버튼 */}
              {!collapsed && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(conv.id);
                  }}
                  className="ml-2 text-gray-500 dark:text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition"
                  title="대화 삭제"
                >
                  <FiTrash2 size={14} />
                </button>
              )}
            </div>
          ))
        )}
      </nav>
      <div className={`p-4 border-t w-full ${collapsed ? 'text-center' : ''}`}>
        <button
          onClick={onNewConversation}
          className={`w-full py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition flex items-center justify-center ${collapsed ? 'text-xs px-1 py-1' : ''}`}
        >
          <FiPlus className="mr-1" /> {collapsed ? '' : '새 대화'}
        </button>
      </div>
    </div>
  );
}
export default Sidebar;
