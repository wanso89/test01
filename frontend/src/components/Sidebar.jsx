import { useState, useEffect } from 'react';
import { FiPlus, FiTrash2 } from 'react-icons/fi';

function Sidebar({ collapsed, onSelectConversation }) {
  const [conversations, setConversations] = useState([]);
  const [activeConversation, setActiveConversation] = useState(null);

  useEffect(() => {
    // localStorage에서 대화 기록 불러오기
    const savedConversations = localStorage.getItem('conversations');
    if (savedConversations) {
      try {
        const parsedConvs = JSON.parse(savedConversations);
        setConversations(parsedConvs);
        // 새로고침 후 마지막 활성 대화 세션 설정
        if (parsedConvs.length > 0) {
          const lastConvId = parsedConvs[parsedConvs.length - 1].id;
          setActiveConversation(lastConvId);
          if (onSelectConversation) {
            onSelectConversation(lastConvId);
          }
        }
      } catch (e) {
        console.error("Error parsing conversations from localStorage:", e);
        localStorage.removeItem('conversations'); // 손상된 데이터 삭제
        setConversations([]);
      }
    }
  }, [onSelectConversation]);

  useEffect(() => {
    // 대화 기록 저장
    if (conversations.length > 0) {
      localStorage.setItem('conversations', JSON.stringify(conversations));
    }
  }, [conversations]);

  // handleNewConversation 함수 수정
  const handleNewConversation = () => {
    const now = new Date();
    const newConv = {
      id: Date.now(),
      title: `대화 ${conversations.length + 1}`,
      timestamp: now.toLocaleString(),
      messages: [{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }]
    };
    
    // 기존 대화 목록에 새 대화 추가
    const updatedConversations = [...conversations, newConv];
    setConversations(updatedConversations);
    
    // 새 대화를 활성화
    setActiveConversation(newConv.id);
    if (onSelectConversation) {
      onSelectConversation(newConv.id);
    }
    
    // localStorage에 저장
    localStorage.setItem('conversations', JSON.stringify(updatedConversations));
  };


  const handleDeleteConversation = (id) => {
    const updatedConversations = conversations.filter(conv => conv.id !== id);
    setConversations(updatedConversations);
    localStorage.setItem('conversations', JSON.stringify(updatedConversations));
    
    if (activeConversation === id) {
      const newActiveId = updatedConversations.length > 0 ? updatedConversations[updatedConversations.length - 1].id : null;
      setActiveConversation(newActiveId);
      if (onSelectConversation) {
        onSelectConversation(newActiveId);
      }
    }
  };

  const handleSelectConversation = (id) => {
    setActiveConversation(id);
    if (onSelectConversation) {
      onSelectConversation(id);
    }
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
