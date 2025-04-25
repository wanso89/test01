import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { useRef, useEffect, useState, useMemo } from 'react';
import { FiLoader } from 'react-icons/fi';

function ChatContainer({ scrollLocked, activeConversationId }) {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // 대화 세션 변경 시 메시지 로드 (localStorage에서)
  useEffect(() => {
    if (activeConversationId) {
      const savedConversations = localStorage.getItem('conversations');
      if (savedConversations) {
        try {
          const convs = JSON.parse(savedConversations);
          const activeConv = convs.find(conv => conv.id === activeConversationId);
          if (activeConv && activeConv.messages && activeConv.messages.length > 0) {
            setMessages(activeConv.messages);
          } else {
            setMessages([{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }]);
          }
        } catch (e) {
          console.error("Error parsing conversations from localStorage:", e);
          setMessages([{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }]);
        }
      } else {
        setMessages([{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }]);
      }
    } else {
      setMessages([{ role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }]);
    }
  }, [activeConversationId]);

  // 메시지 추가 시 스크롤 처리
  useEffect(() => {
    if (!scrollLocked) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, scrollLocked]);

  // 메시지 전송 핸들러 (백엔드 연동 준비)
  const handleSend = async (msg) => {
    if (!activeConversationId) return;
    
    const newMessage = { role: 'user', content: msg };
    setMessages(prev => {
      const updatedMessages = [...prev, newMessage];
      // localStorage에 저장 (대화 세션에 연결)
      if (activeConversationId) {
        const savedConversations = localStorage.getItem('conversations');
        if (savedConversations) {
          try {
            const convs = JSON.parse(savedConversations);
            const updatedConvs = convs.map(conv => {
              if (conv.id === activeConversationId) {
                return { ...conv, messages: updatedMessages };
              }
              return conv;
            });
            localStorage.setItem('conversations', JSON.stringify(updatedConvs));
          } catch (e) {
            console.error("Error updating conversations in localStorage:", e);
          }
        }
      }
      return updatedMessages;
    });

    // 로딩 상태 표시
    setIsLoading(true);
    setError(null);

    try {
      // 실제 백엔드 API 호출 (주석 해제 후 사용)
      /*
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, conversationId: activeConversationId })
      });
      if (!response.ok) {
        throw new Error('API 호출 실패');
      }
      const data = await response.json();
      const aiResponse = { role: 'assistant', content: data.response, sources: data.sources || [] };
      */
      // 임시 더미 응답 (백엔드 연동 전)
      const aiResponse = { role: 'assistant', content: `응답: "${msg}"에 대한 답변입니다.`, sources: [] };
      await new Promise(resolve => setTimeout(resolve, 1000)); // 1초 지연 시뮬레이션

      setMessages(prev => {
        const updatedMessages = [...prev, aiResponse];
        // localStorage에 저장
        if (activeConversationId) {
          const savedConversations = localStorage.getItem('conversations');
          if (savedConversations) {
            try {
              const convs = JSON.parse(savedConversations);
              const updatedConvs = convs.map(conv => {
                if (conv.id === activeConversationId) {
                  return { ...conv, messages: updatedMessages };
                }
                return conv;
              });
              localStorage.setItem('conversations', JSON.stringify(updatedConvs));
            } catch (e) {
              console.error("Error updating conversations in localStorage:", e);
            }
          }
        }
        return updatedMessages;
      });
    } catch (err) {
      setError('응답을 가져오는 중 오류가 발생했습니다. 다시 시도해주세요.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  // 메시지 렌더링 성능 최적화
  const memoizedMessages = useMemo(() => {
    return messages.map((msg, i) => (
      <div
        key={i}
        className="animate-fade-in-up opacity-0"
        style={{
          animation: 'fade-in-up 0.3s ease-out forwards',
          animationDelay: `${i * 0.1}s`
        }}
      >
        <ChatMessage message={msg} />
      </div>
    ));
  }, [messages]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-6 space-y-4 bg-gradient-to-br from-transparent to-white/50 dark:from-transparent dark:to-gray-900/50"
      >
        {memoizedMessages}
        {isLoading && (
          <div className="flex justify-center items-center py-2">
            <FiLoader className="animate-spin text-blue-500 dark:text-blue-400" size={20} />
            <span className="ml-2 text-gray-600 dark:text-gray-400 text-sm">응답을 기다리는 중...</span>
          </div>
        )}
        {error && (
          <div className="flex justify-center items-center py-2 text-red-500 dark:text-red-400 text-sm">
            {error}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="sticky bottom-0 z-10 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        <ChatInput onSend={handleSend} disabled={isLoading} />
      </div>
    </div>
  );
}
export default ChatContainer;
