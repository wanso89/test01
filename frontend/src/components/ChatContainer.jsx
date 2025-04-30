import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import { useRef, useEffect, useState, useMemo } from "react";
import { FiLoader, FiArrowUp, FiType } from "react-icons/fi";

function ChatContainer({
  scrollLocked,
  activeConversationId,
  messages,
  searchTerm,
  filteredMessages,
  onUpdateMessages,
}) {
  const [localMessages, setLocalMessages] = useState(messages);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const inputRef = useRef(null);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [isTyping, setIsTyping] = useState(false); // 입력 중 상태 추가

  // 입력창 자동 포커스
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  // "입력 중" 메세지 출력 시 스크롤 이동
  useEffect(() => {
    if (isTyping && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [isTyping]);

  // messages 또는 activeConversationId가 변경되면 localMessages 업데이트
  useEffect(() => {
    setLocalMessages(messages);
  }, [messages, activeConversationId]);

  // 메시지 추가 시 스크롤 처리
  useEffect(() => {
    if (!scrollLocked) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [localMessages, scrollLocked]);

  // 검색어가 지워질 때 스크롤을 맨 아래로 이동
  useEffect(() => {
    if (!searchTerm) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [searchTerm]);

  // 스크롤 위치 감지하여 '맨 위로' 버튼 표시 여부 결정
  const handleScroll = () => {
    const container = chatContainerRef.current;
    if (container) {
      const scrollTop = container.scrollTop;
      const scrollHeight = container.scrollHeight;
      const visibleHeight = container.clientHeight;
      // 스크롤이 맨 아래에 가까울 때는 버튼 숨김
      setShowScrollTop(scrollTop < scrollHeight - visibleHeight - 100);
    }
  };

  useEffect(() => {
    const container = chatContainerRef.current;
    if (container) {
      container.addEventListener("scroll", handleScroll);
      handleScroll(); // 초기 스크롤 위치 확인
      return () => container.removeEventListener("scroll", handleScroll);
    }
  }, []);

  // '맨 위로' 버튼 클릭 핸들러
  const scrollToTop = () => {
    chatContainerRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  };

  // 입력 중 상태 핸들러
  const handleTyping = (typing) => {
    setIsTyping(typing);
  };

  // 메시지 전송 핸들러 (FastAPI 백엔드 연동)
  const handleSend = async (msg) => {
    if (!activeConversationId || isSending) return;

    setIsSending(true);
    const newMessage = {
      role: "user",
      content: msg,
      reactions: {},
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
    };
    const updatedMessagesWithUser = [...localMessages, newMessage];
    setLocalMessages(updatedMessagesWithUser);
    onUpdateMessages(updatedMessagesWithUser);

    // 로딩 상태 표시
    setIsLoading(true);
    setError(null);

    try {
      // FastAPI 백엔드와 통신
      const response = await fetch("http://172.10.2.70:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: msg,
          category: "메뉴얼", // 카테고리 설정 (필요 시 동적 변경)
          history: updatedMessagesWithUser.map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
      });

      if (!response.ok) {
        throw new Error(
          `FastAPI 호출 실패: ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();
      const aiResponse = {
        role: "assistant",
        content: data.bot_response || "응답을 받아왔습니다.",
        sources: data.sources || [],
        reactions: {},
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      };
      const updatedMessagesWithAI = [...updatedMessagesWithUser, aiResponse];
      setLocalMessages(updatedMessagesWithAI);
      onUpdateMessages(updatedMessagesWithAI);
    } catch (err) {
      setError(
        `응답을 가져오는 중 오류가 발생했습니다: ${err.message}. FastAPI 서버가 실행 중인지 확인해주세요.`
      );
      console.error(err);
    } finally {
      setIsLoading(false);
      setIsSending(false);
    }
  };

  // 메시지 렌더링 성능 최적화
  const memoizedMessages = useMemo(() => {
    return (searchTerm ? filteredMessages : localMessages).map((msg, i) => (
      <div
        key={msg.id || i}
        className="animate-fade-in-up opacity-0"
        style={{
          animation: "fade-in-up 0.3s ease-out forwards",
          animationDelay: `${i * 0.1}s`,
        }}
      >
        <ChatMessage message={msg} searchTerm={searchTerm || ""} />
      </div>
    ));
  }, [localMessages, filteredMessages, searchTerm]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-6 space-y-4 bg-gradient-to-br from-transparent to-white/50 dark:from-transparent dark:to-gray-900/50 relative"
      >
        {memoizedMessages}
        {isLoading && (
          <div className="flex justify-center items-center py-2">
            <FiLoader
              className="animate-spin text-blue-500 dark:text-blue-400"
              size={20}
            />
            <span className="ml-2 text-gray-600 dark:text-gray-400 text-sm">
              응답을 기다리는 중...
            </span>
          </div>
        )}
        {isTyping && !isLoading && !isSending && (
          <div className="flex justify-center items-center py-2">
            <FiType
              className="animate-pulse text-gray-500 dark:text-gray-400"
              size={20}
            />
            <span className="ml-2 text-gray-600 dark:text-gray-400 text-sm">
              입력 중...
            </span>
          </div>
        )}
        {error && (
          <div className="flex justify-center items-center py-2 text-red-500 dark:text-red-400 text-sm">
            {error}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      {showScrollTop && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-16 right-6 bg-blue-600 text-white rounded-full p-2 shadow-lg hover:bg-blue-700 transition z-20 animate-fade-in"
          title="맨 위로 이동"
        >
          <FiArrowUp size={20} />
        </button>
      )}
      <div className="sticky bottom-0 z-10 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        <ChatInput
          onSend={handleSend}
          disabled={isSending || isLoading}
          onTyping={handleTyping}
        />
      </div>
    </div>
  );
}
export default ChatContainer;
