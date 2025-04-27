import { useState, useRef, useEffect } from 'react';
import { FiLoader, FiSend } from 'react-icons/fi';

function ChatInput({ onSend, disabled }) {
  const [msg, setMsg] = useState('');
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
    if (e.key === 'Enter') {
      if (!e.shiftKey) {
        // 엔터만 눌렀을 때: 메시지 전송
        e.preventDefault();
        if (msg.trim() && !disabled) {
          onSend(msg);
          setMsg('');
        }
      }
      // Shift+Enter: 줄바꿈 (기본 동작 유지, 별도 처리 불필요)
    }
  };

  const handleSendClick = () => {
    if (msg.trim() && !disabled) {
      onSend(msg);
      setMsg('');
    }
  };

  return (
    <form
      className="flex p-4 border-t bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700"
      onSubmit={e => {
        e.preventDefault();
        handleSendClick();
      }}
    >
      <textarea
        ref={textareaRef}
        className="flex-1 p-3 rounded-full border-2 border-gray-200 dark:border-gray-700 focus:ring-2 focus:ring-blue-400 bg-white dark:bg-gray-700 text-gray-800 dark:text-white transition resize-none overflow-hidden duration-200"
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
