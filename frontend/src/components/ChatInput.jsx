import { useState, useRef, useEffect } from 'react';
import { FiLoader, FiSend, FiPaperclip, FiX } from 'react-icons/fi';

function ChatInput({ onSend, disabled, onTyping }) {
  const [msg, setMsg] = useState('');
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [suggestions, setSuggestions] = useState([]);
  const [files, setFiles] = useState([]); // 다중 파일을 배열로 관리
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  
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

  // 입력 상태 감지 및 상위 컴포넌트로 전달
  useEffect(() => {
    if (msg.trim().length > 0 && !disabled) {
      onTyping(true); // 입력 중 상태 전달
    } else {
      onTyping(false); // 입력 중 상태 해제
    }
  }, [msg, disabled, onTyping]);

  // 파일 선택 핸들러 (다중 파일 선택 지원)
  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files); // FileList를 배열로 변환
    if (selectedFiles.length > 0) {
      const updatedFiles = selectedFiles.map(file => {
        if (file.type.startsWith('image/')) {
          return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (event) => {
              resolve({ file, preview: event.target.result });
            };
            reader.readAsDataURL(file);
          });
        }
        return { file, preview: null };
      });
      Promise.all(updatedFiles).then(newFiles => {
        setFiles(prevFiles => [...prevFiles, ...newFiles]); // 기존 파일에 추가
      });
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''; // 파일 입력 초기화
    }
  };

  // 입력 중 자동 완성 제안 생성 (히스토리 기반)
  useEffect(() => {
    if (msg.trim().length > 0) {
      const filteredSuggestions = history
        .filter(item => item.toLowerCase().includes(msg.toLowerCase()))
        .slice(0, 3); // 최대 3개 제안
      setSuggestions(filteredSuggestions);
    } else {
      setSuggestions([]);
    }
  }, [msg, history]);

  // 자동 완성 제안 선택 핸들러
  const handleSuggestionSelect = (suggestion) => {
    setMsg(suggestion);
    setSuggestions([]);
    textareaRef.current?.focus();
  };

  // 메시지 전송 함수
  const handleSendMessage = () => {
    if ((msg.trim() || files.length > 0) && !disabled) {
      let content = msg.trim();
      if (files.length > 0) {
        const fileNames = files.map(f => f.file.name).join(', ');
        content = content ? `${content} (첨부파일: ${fileNames})` : `첨부파일: ${fileNames}`;
      }
      onSend(content);
      setHistory(prev => [content, ...prev]); // 입력 히스토리에 추가
      setMsg('');
      setHistoryIndex(-1); // 히스토리 인덱스 초기화
      setSuggestions([]); // 제안 초기화
      setFiles([]); // 파일 초기화
    }
  };

  // 파일 제거 핸들러 (특정 파일만 제거)
  const handleRemoveFile = (indexToRemove) => {
    setFiles(prevFiles => prevFiles.filter((_, index) => index !== indexToRemove));
  };

  // handleKeyDown 함수
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if ((msg.trim() || files.length > 0) && !disabled) {
        handleSendMessage();
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
    } else if (e.key === 'Tab' && suggestions.length > 0) {
      e.preventDefault();
      handleSuggestionSelect(suggestions[0]); // 첫 번째 제안 선택
    }
  };

  return (
    <form
      className="flex p-4 border-t bg-white dark:bg-gray-800"
      onSubmit={e => {
        e.preventDefault();
        handleSendMessage();
      }}
    >
      <div className="flex-1">
        <textarea
          ref={textareaRef}
          className="flex-1 p-3 rounded-full border-2 border-gray-200 dark:border-gray-700 focus:ring-2 focus:ring-blue-400 bg-white dark:bg-gray-700 text-gray-800 dark:text-white transition resize-none overflow-hidden"
          placeholder={disabled ? "전송 중..." : "메시지를 입력하세요. Shift+Enter로 줄바꿈, Enter로 전송됩니다."}
          value={msg}
          onChange={e => setMsg(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={disabled}
          style={{ minHeight: 40, maxHeight: 120, scrollbarWidth: 'none', msOverflowStyle: 'none', width: '100%' }}
        />
        {suggestions.length > 0 && (
          <div className="absolute bottom-16 left-0 w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg z-10 max-h-40 overflow-y-auto">
            {suggestions.map((suggestion, index) => (
              <div
                key={index}
                className="px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition cursor-pointer"
                onClick={() => handleSuggestionSelect(suggestion)}
              >
                {suggestion}
              </div>
            ))}
          </div>
        )}
        {files.length > 0 && (
          <div className="mt-2 max-h-24 overflow-y-auto bg-gray-50 dark:bg-gray-700 rounded border border-gray-200 dark:border-gray-600 p-2">
            {files.map((fileItem, index) => (
              <div key={index} className="flex items-center justify-between mb-1 bg-gray-100 dark:bg-gray-800 rounded px-2 py-1">
                {fileItem.preview ? (
                  <div className="flex items-center">
                    <img src={fileItem.preview} alt="미리보기" className="w-8 h-8 object-cover rounded mr-2" />
                    <span 
                      className="text-gray-800 dark:text-gray-100 text-xs overflow-wrap-break-word" 
                      style={{ overflowWrap: 'break-word' }} 
                      title={fileItem.file.name}
                    >
                      {fileItem.file.name}
                    </span>
                  </div>
                ) : (
                  <span 
                    className="text-gray-800 dark:text-gray-100 text-xs overflow-wrap-break-word" 
                    style={{ overflowWrap: 'break-word' }} 
                    title={fileItem.file.name}
                  >
                    {fileItem.file.name}
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => handleRemoveFile(index)}
                  className="ml-2 text-gray-500 dark:text-gray-400 hover:text-red-500 dark:hover:text-red-400"
                  title="파일 제거"
                >
                  <FiX size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
      <input
        type="file"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={handleFileChange}
        accept="image/*,application/pdf,.doc,.docx"
        multiple // 다중 파일 선택 가능
      />
      <button
        type="button"
        onClick={() => fileInputRef.current.click()}
        className="ml-2 px-3 py-2 rounded-full text-gray-600 dark:text-gray-300 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
        disabled={disabled}
        title="파일 첨부"
        style={{ minWidth: 40, minHeight: 40, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
      >
        <FiPaperclip size={18} />
      </button>
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
