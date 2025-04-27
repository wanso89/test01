import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { useRef, useEffect, useState, useMemo } from 'react';
import { FiLoader, FiSend } from 'react-icons/fi';

function ChatContainer({
  scrollLocked,
  activeConversationId,
  messages,
  searchTerm,
  filteredMessages,
  onUpdateMessages
}) {
  const [localMessages, setLocalMessages] = useState(messages);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const inputRef = useRef(null);

  // 입력창 자동 포커스
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);
 

  // messages 또는 activeConversationId가 변경되면 localMessages 업데이트
  useEffect(() => {
    setLocalMessages(messages);
  }, [messages, activeConversationId]);

  // 메시지 추가 시 스크롤 처리
  useEffect(() => {
    if (!scrollLocked) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [localMessages, scrollLocked]);

  // 메시지 전송 핸들러 (FastAPI 백엔드 연동)
  const handleSend = async (msg) => {
    if (!activeConversationId || isSending) return;

    setIsSending(true);
    const newMessage = { role: 'user', content: msg };
    const updatedMessagesWithUser = [...localMessages, newMessage];
    setLocalMessages(updatedMessagesWithUser);
    onUpdateMessages(updatedMessagesWithUser);

    // 로딩 상태 표시
    setIsLoading(true);
    setError(null);

    try {
      // FastAPI 백엔드와 통신
      const response = await fetch('http://172.10.2.70:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: msg,
          category: '메뉴얼', // 카테고리 설정 (필요 시 동적 변경)
          history: updatedMessagesWithUser.map(m => ({
            role: m.role,
            content: m.content
          }))
        })
      });

      if (!response.ok) {
        throw new Error(`FastAPI 호출 실패: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      const aiResponse = { 
        role: 'assistant', 
        content: data.bot_response || '응답을 받아왔습니다.',
        sources: data.sources || []
      };
      const updatedMessagesWithAI = [...updatedMessagesWithUser, aiResponse];
      setLocalMessages(updatedMessagesWithAI);
      onUpdateMessages(updatedMessagesWithAI);
    } catch (err) {
      setError(`응답을 가져오는 중 오류가 발생했습니다: ${err.message}. FastAPI 서버가 실행 중인지 확인해주세요.`);
      console.error(err);
    } finally {
      setIsLoading(false);
      setIsSending(false);
    }
  };

  // 검색 결과 표시 및 하이라이트 (간단한 CSS 스타일로 강조)
  const highlightText = (text, term) => {
    if (!term) return text;
    const regex = new RegExp(`(${term})`, 'gi');
    const parts = text.split(regex);
    return parts.map((part, index) =>
      part.toLowerCase() === term.toLowerCase() ? (
        <span key={index} className="bg-yellow-200 dark:bg-yellow-700 text-black dark:text-white">
          {part}
        </span>
      ) : (
        part
      )
    );
  };

  // 메시지 렌더링 성능 최적화
  const memoizedMessages = useMemo(() => {
    return (searchTerm ? filteredMessages : localMessages).map((msg, i) => (
      <div
        key={i}
        className="animate-fade-in-up opacity-0"
        style={{
          animation: 'fade-in-up 0.3s ease-out forwards',
          animationDelay: `${i * 0.1}s`
        }}
      >
        <ChatMessage
          message={msg} // 하이라이트 적용 없이 원본 메시지 전달
        />
      </div>
    ));
  }, [localMessages, filteredMessages, searchTerm]);

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
        <ChatInput onSend={handleSend} disabled={isSending || isLoading} />
      </div>
    </div>
  );
}
export default ChatContainer;
