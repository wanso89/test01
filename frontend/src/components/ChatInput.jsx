import { useState, useRef, useEffect } from 'react';
import { FiLoader, FiSend } from 'react-icons/fi';

function ChatInput({ onSend, disabled }) {
  const [msg, setMsg] = useState('');
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const textareaRef = useRef(null);
  
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  }, [msg]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (msg.trim() && !disabled) {
        onSend(msg);
        setHistory(prev => [msg, ...prev]); // 입력 히스토리에 추가
        setMsg('');
        setHistoryIndex(-1); // 히스토리 인덱스 초기화
      }
    } else if (e.key === 'ArrowUp' && history.length > 0) {
      e.preventDefault();
      if (historyIndex < history.length - 1) {
        setHistoryIndex(prev => prev + 1);
        setMsg(history[historyIndex + 1]);
      }
    } else if (e.key === 'ArrowDown' && history.length > 0) {
      e.preventDefault();
      if (historyIndex > 0) {
        setHistoryIndex(prev => prev - 1);
        setMsg(history[historyIndex - 1]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setMsg('');
      }
    }
  };

  return (
    <form
      className="flex p-4 border-t bg-white dark:bg-gray-800"
      onSubmit={e => {
        e.preventDefault();
        if (msg.trim() && !disabled) {
          onSend(msg);
          setHistory(prev => [msg, ...prev]); // 입력 히스토리에 추가
          setMsg('');
          setHistoryIndex(-1); // 히스토리 인덱스 초기화
        }
      }}
    >
      <textarea
        ref={textareaRef}
        className="flex-1 p-3 rounded-full border-2 border-gray-200 dark:border-gray-700 focus:ring-2 focus:ring-blue-400 bg-white dark:bg-gray-700 text-gray-800 dark:text-white transition resize-none overflow-hidden"
        placeholder={disabled ? "전송 중..." : "메시지를 입력하세요. Shift+Enter로 줄바꿈, Enter로 전송됩니다."}
        value={msg}
        onChange={e => setMsg(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={1}
        disabled={disabled}
        style={{ minHeight: 40, maxHeight: 120, scrollbarWidth: 'none', msOverflowStyle: 'none' }}
      />
      <button
        type="submit"
        className={`ml-2 px-6 py-2 rounded-full text-white font-bold shadow transition transform duration-200 ${
          disabled 
            ? 'bg-gray-400 dark:bg-gray-600 cursor-not-allowed' 
            : 'bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 hover:scale-105'
        }`}
        disabled={disabled}
      >
        {disabled ? <FiLoader className="animate-spin" size={18} /> : <FiSend size={18} />}
      </button>
    </form>
  );
}
export default ChatInput;
