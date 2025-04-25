import { useState, useEffect } from 'react';
import { FiPlus, FiTrash2 } from 'react-icons/fi';

function Sidebar({ collapsed }) {
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);

  useEffect(() => {
    // localStorage에서 대화 기록 불러오기
    const savedConversations = localStorage.getItem('conversations');
    if (savedConversations) {
      setConversations(JSON.parse(savedConversations));
    }
  }, []);

  useEffect(() => {
    // 대화 기록 저장
    localStorage.setItem('conversations', JSON.stringify(conversations));
  }, [conversations]);

  const handleNewConversation = () => {
    const now = new Date();
    const newConv = {
      id: Date.now(),
      title: `대화 ${conversations.length + 1}`,
      timestamp: now.toLocaleString(),
      messages: []
    };
    setConversations(prev => [...prev, newConv]);
    setActiveConversation(newConv.id);
  };

  const handleDeleteConversation = (id) => {
    setConversations(prev => prev.filter(conv => conv.id !== id));
    if (activeConversation === id) {
      setActiveConversation(null);
    }
  };

  const handleSelectConversation = (id) => {
    setActiveConversation(id);
    // 실제로는 해당 대화의 메시지를 로드하는 로직 추가
    console.log(`Selected conversation: ${id}`);
  };

  return (
    <div className={`flex flex-col h-full ${collapsed ? 'items-center' : ''} transition-all`}>
      <div className={`p-4 border-b w-full ${collapsed ? 'text-center' : ''}`}>
        <h2 className={`font-bold text-lg text-gray-700 dark:text-white ${collapsed ? 'text-xs' : ''}`}>대화</h2>
      </div>
      <nav className={`flex-1 p-4 w-full space-y-2 ${collapsed ? 'px-1' : ''} overflow-y-auto`}>
        {conversations.map(conv => (
          <div
            key={conv.id}
            className={`
              flex justify-between items-center p-2 rounded-xl cursor-pointer transition
              ${activeConversation === conv.id
                ? 'bg-gradient-to-r from-blue-100 to-blue-200 dark:from-blue-900 dark:to-blue-800 text-blue-800 dark:text-blue-200 font-semibold'
                : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'}
            `}
            onClick={() => handleSelectConversation(conv.id)}
            title={collapsed ? conv.title : ''}
          >
            <div className="flex-1 overflow-hidden">
              <div className={`text-sm ${collapsed ? 'text-xs truncate' : ''}`}>{conv.title}</div>
              {!collapsed && <div className="text-xs text-gray-500 dark:text-gray-400 truncate">{conv.timestamp}</div>}
            </div>
            {!collapsed && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDeleteConversation(conv.id);
                }}
                className="ml-2 text-gray-500 dark:text-gray-400 hover:text-red-500 dark:hover:text-red-400 transition"
                title="대화 삭제"
              >
                <FiTrash2 size={14} />
              </button>
            )}
          </div>
        ))}
      </nav>
      <div className={`p-4 border-t w-full ${collapsed ? 'text-center' : ''}`}>
        <button
          onClick={handleNewConversation}
          className={`w-full py-2 bg-blue-600 text-white rounded-xl hover:bg-blue-700 transition flex items-center justify-center ${collapsed ? 'text-xs px-1 py-1' : ''}`}
        >
          <FiPlus className="mr-1" /> {collapsed ? '' : '새 대화'}
        </button>
      </div>
    </div>
  );
}
export default Sidebar;
