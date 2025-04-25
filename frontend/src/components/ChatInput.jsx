import { useState, useRef, useEffect } from 'react';

function ChatInput({ onSend }) {
  const [msg, setMsg] = useState('');
  const textareaRef = useRef(null);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (msg.trim()) {
        onSend(msg);
        setMsg('');
      }
    }
  };

  return (
    <form
      className="flex p-4 border-t bg-white dark:bg-gray-800"
      onSubmit={e => {
        e.preventDefault();
        if (msg.trim()) {
          onSend(msg);
          setMsg('');
        }
      }}
    >
      <textarea
        ref={textareaRef}
        className="flex-1 p-3 rounded-full border-2 border-gray-200 dark:border-gray-700 focus:ring-2 focus:ring-blue-400 bg-white dark:bg-gray-700 text-gray-800 dark:text-white transition resize-none"
        placeholder="메시지를 입력하세요. Shift+Enter로 줄바꿈, Enter로 전송됩니다."
        value={msg}
        onChange={e => setMsg(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={1}
        style={{ minHeight: 40, maxHeight: 120 }}
      />
      <button
        type="submit"
        className="ml-2 px-6 py-2 rounded-full bg-gradient-to-r from-blue-500 to-indigo-500 text-white font-bold shadow hover:from-blue-600 hover:to-indigo-600 hover:scale-105 transition transform duration-200"
      >
        전송
      </button>
    </form>
  );
}
export default ChatInput;
