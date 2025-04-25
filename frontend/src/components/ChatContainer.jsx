import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { useRef, useEffect, useState } from 'react';

function ChatContainer({ scrollLocked }) {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }
  ]);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    if (!scrollLocked) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, scrollLocked]);

  const handleSend = (msg) => {
    setMessages(prev => [...prev, { role: 'user', content: msg }]);
    // 실제로는 백엔드 API 호출 후 응답 추가
    // 예시로 간단히 응답 추가
    setTimeout(() => {
      setMessages(prev => [...prev, { role: 'assistant', content: `응답: "${msg}"에 대한 답변입니다.` }]);
    }, 1000);
  };

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-6 space-y-4 bg-gradient-to-br from-transparent to-white/50 dark:from-transparent dark:to-gray-900/50"
      >
        {messages.map((msg, i) => (
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
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="sticky bottom-0 z-10 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        <ChatInput onSend={handleSend} />
      </div>
    </div>
  );
}
export default ChatContainer;
