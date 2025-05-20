import React, { useState, useRef, useEffect, useCallback } from 'react';
import { FiSend, FiMessageCircle, FiLoader, FiThumbsUp, FiThumbsDown, FiCopy, FiUser, FiCornerDownRight, FiFileText, FiAlertCircle, FiMessageSquare } from 'react-icons/fi';
import { Transition } from '@headlessui/react';

// 사용자 메시지 컴포넌트
const UserMessage = ({ content, timestamp }) => {
  return (
    <div className="flex justify-end mb-4 animate-fade-in-up">
      <div className="flex items-start max-w-[85%] sm:max-w-[75%]">
        <div className="order-2 ml-2 flex-shrink-0 mt-1">
          <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white shadow-md">
            <FiUser className="w-4 h-4" />
          </div>
        </div>
        <div className="order-1 px-4 py-3 rounded-t-xl rounded-bl-xl bg-indigo-600 text-white shadow-md">
          <div className="prose prose-sm prose-invert break-words">
            {content}
          </div>
          {timestamp && (
            <div className="text-xs text-indigo-200 mt-1 text-right">
              {new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// 챗봇 메시지 컴포넌트
const BotMessage = ({ content, sources, isLoading, onThumbsUp, onThumbsDown, onCopy, timestamp }) => {
  const [isTyping, setIsTyping] = useState(isLoading);
  const [displayedContent, setDisplayedContent] = useState(isLoading ? '' : content);
  
  // 타이핑 효과
  useEffect(() => {
    if (isLoading) {
      setIsTyping(true);
      setDisplayedContent('');
    } else {
      if (content) {
        setIsTyping(false);
        setDisplayedContent(content);
      }
    }
  }, [isLoading, content]);

  return (
    <div className="flex mb-4 animate-fade-in-up">
      <div className="flex items-start max-w-[85%] sm:max-w-[75%]">
        <div className="mr-2 flex-shrink-0 mt-1">
          <div className="w-8 h-8 rounded-full bg-neutral-700 flex items-center justify-center text-white shadow-md">
            <FiMessageCircle className="w-4 h-4" />
          </div>
        </div>
        <div className="px-4 py-3 rounded-t-xl rounded-br-xl bg-neutral-800 text-white shadow-md">
          <div className="prose prose-sm prose-invert break-words min-w-[240px]">
            {isTyping ? (
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '200ms' }}></div>
                <div className="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style={{ animationDelay: '400ms' }}></div>
              </div>
            ) : (
              <>
                {displayedContent}
                {sources && sources.length > 0 && (
                  <div className="mt-3 pt-2 border-t border-neutral-700">
                    <p className="text-xs text-neutral-400 mb-1">출처:</p>
                    <div className="space-y-1">
                      {sources.map((source, index) => (
                        <div key={index} className="flex items-center text-xs text-neutral-300">
                          <FiFileText className="w-3 h-3 mr-1 text-neutral-400" />
                          <span className="truncate">{source.title || source.source}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
          
          {!isTyping && (
            <div className="mt-2 pt-1 flex items-center justify-between">
              <div className="text-xs text-neutral-400">
                {timestamp && new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </div>
              <div className="flex space-x-1">
                <button 
                  onClick={onCopy} 
                  className="p-1 text-xs rounded-md text-neutral-400 hover:bg-neutral-700 hover:text-white transition-colors"
                  aria-label="내용 복사"
                >
                  <FiCopy className="w-3 h-3" />
                </button>
                <button 
                  onClick={onThumbsUp} 
                  className="p-1 text-xs rounded-md text-neutral-400 hover:bg-neutral-700 hover:text-white transition-colors"
                  aria-label="좋아요"
                >
                  <FiThumbsUp className="w-3 h-3" />
                </button>
                <button 
                  onClick={onThumbsDown}
                  className="p-1 text-xs rounded-md text-neutral-400 hover:bg-neutral-700 hover:text-white transition-colors"
                  aria-label="싫어요"
                >
                  <FiThumbsDown className="w-3 h-3" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// 입력 컴포넌트
const ChatInput = ({ onSend, disabled, placeholder }) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef(null);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (message.trim()) {
      onSend(message);
      setMessage('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;

    const adjustHeight = () => {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 150)}px`;
    };

    textarea.addEventListener('input', adjustHeight);
    adjustHeight();

    return () => {
      textarea.removeEventListener('input', adjustHeight);
    };
  }, [message]);

  return (
    <div className="relative w-full max-w-3xl mx-auto">
      <div className="bg-neutral-800 border border-neutral-700 rounded-xl shadow-lg p-1 flex items-end">
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          className="flex-grow bg-transparent text-white placeholder-neutral-400 resize-none outline-none py-3 px-3 max-h-[150px] min-h-[44px]"
          placeholder={placeholder || "메시지를 입력하세요..."}
          rows={1}
        />
        <button
          onClick={handleSend}
          disabled={!message.trim() || disabled}
          className={`p-2.5 rounded-lg mr-1 mb-1 ${
            !message.trim() || disabled
              ? 'bg-neutral-700 text-neutral-400'
              : 'bg-indigo-600 text-white hover:bg-indigo-700'
          } transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500`}
          aria-label="메시지 전송"
        >
          <FiSend className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

// 대화 컨테이너
const ChatContainer = ({ messages, isLoading }) => {
  const containerRef = useRef(null);

  // 새 메시지가 추가되면 스크롤 아래로
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div 
      ref={containerRef}
      className="flex-1 overflow-y-auto px-4 py-4 custom-scrollbar"
    >
      {messages.length === 0 ? (
        <div className="h-full flex flex-col items-center justify-center text-neutral-400">
          <FiMessageSquare className="w-12 h-12 mb-4 text-neutral-500" />
          <h3 className="text-xl font-semibold mb-2 text-neutral-300">RAG 챗봇과 대화해보세요</h3>
          <p className="text-center max-w-sm mb-6">
            문서를 기반으로 질문하고 답변을 받을 수 있습니다. 지금 시작해보세요!
          </p>
        </div>
      ) : (
        <>
          {messages.map((msg, index) => (
            <div key={index}>
              {msg.role === 'user' ? (
                <UserMessage 
                  content={msg.content} 
                  timestamp={msg.timestamp} 
                />
              ) : (
                <BotMessage
                  content={msg.content}
                  sources={msg.sources}
                  isLoading={index === messages.length - 1 && isLoading}
                  onThumbsUp={() => console.log('Thumbs up for message:', index)}
                  onThumbsDown={() => console.log('Thumbs down for message:', index)}
                  onCopy={() => {
                    navigator.clipboard.writeText(msg.content);
                    // 여기에 복사 완료 토스트 표시 코드 추가 가능
                  }}
                  timestamp={msg.timestamp}
                />
              )}
            </div>
          ))}
          {isLoading && messages[messages.length - 1]?.role === 'user' && (
            <BotMessage isLoading={true} />
          )}
        </>
      )}
    </div>
  );
};

// 로딩 인디케이터 컴포넌트
const LoadingOverlay = ({ active }) => {
  if (!active) return null;
  
  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-neutral-800 rounded-lg p-6 shadow-xl flex items-center space-x-4">
        <div className="relative w-8 h-8">
          <div className="absolute inset-0 rounded-full border-4 border-indigo-500/20"></div>
          <div className="absolute inset-0 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin"></div>
        </div>
        <div className="text-white">문서를 검색하고 있습니다...</div>
      </div>
    </div>
  );
};

// 오류 메시지 컴포넌트
const ErrorMessage = ({ message, onDismiss }) => {
  if (!message) return null;
  
  return (
    <div className="fixed bottom-24 left-1/2 transform -translate-x-1/2 z-50 animate-fade-in-up">
      <div className="bg-red-600 text-white px-4 py-3 rounded-lg shadow-lg flex items-center">
        <FiAlertCircle className="w-5 h-5 mr-2 flex-shrink-0" />
        <span>{message}</span>
        <button 
          onClick={onDismiss} 
          className="ml-4 p-1 rounded hover:bg-red-700 transition-colors"
        >
          <FiCopy className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};

// 메인 채팅 UI 컴포넌트
const ChatUI = ({ initialMessages = [], onSendMessage }) => {
  const [messages, setMessages] = useState(initialMessages);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSendMessage = async (content) => {
    if (!content.trim()) return;
    
    // 새 메시지 객체
    const newUserMessage = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date().toISOString()
    };
    
    // 메시지 추가
    setMessages(prev => [...prev, newUserMessage]);
    setIsLoading(true);
    setError('');
    
    try {
      // 백엔드 API 호출
      const response = await onSendMessage(content);
      
      if (response) {
        // 응답 메시지 추가
        const botMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: response.content || response.text || response.message || '',
          sources: response.sources || [],
          timestamp: new Date().toISOString()
        };
        
        setMessages(prev => [...prev, botMessage]);
      }
    } catch (err) {
      console.error('메시지 전송 오류:', err);
      setError('메시지를 전송하는 중 오류가 발생했습니다. 다시 시도해주세요.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-neutral-900 text-white">
      <LoadingOverlay active={isLoading} />
      
      <ErrorMessage 
        message={error} 
        onDismiss={() => setError('')} 
      />
      
      <main className="flex-1 flex flex-col max-w-5xl w-full mx-auto">
        <ChatContainer messages={messages} isLoading={isLoading} />
        
        <div className="p-4 mt-auto">
          <ChatInput 
            onSend={handleSendMessage} 
            disabled={isLoading} 
            placeholder="무엇이든 물어보세요..." 
          />
          <div className="mt-2 text-center">
            <span className="text-xs text-neutral-500">
              GPT 스타일 챗봇 - RAG 시스템으로 문서 기반 질문에 답변합니다
            </span>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ChatUI; 