import { useState, useRef, useEffect, forwardRef, useImperativeHandle, useCallback } from 'react';
import { FiLoader, FiSend, FiPaperclip, FiX, FiCheck, FiImage, FiMessageSquare, FiFile, FiSmile } from 'react-icons/fi';
import FileUpload from './FileUpload';

const ChatInput = forwardRef(({ onSend, disabled, onTyping, onUploadSuccess }, ref) => {
  const [message, setMessage] = useState('');
  const [showFileUploadModal, setShowFileUploadModal] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('메뉴얼'); // 기본 카테고리
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const categories = ['메뉴얼', '장애보고서', '기술문서', '기타'];
  const typingTimerRef = useRef(null);
  const [isFocused, setIsFocused] = useState(false);

  useImperativeHandle(ref, () => ({
    focus: () => {
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    },
    clear: () => {
      setMessage('');
    },
  }));

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

  useEffect(() => {
    if (typingTimerRef.current) {
      clearTimeout(typingTimerRef.current);
    }

    // 디바운스 처리 - 300ms 동안 타이핑이 없으면 상태 업데이트
    typingTimerRef.current = setTimeout(() => {
      const isTyping = message.trim().length > 0 && !disabled;
      onTyping?.(isTyping);
    }, 300);

    return () => {
      if (typingTimerRef.current) {
        clearTimeout(typingTimerRef.current);
      }
    };
  }, [message, disabled, onTyping]);

  // 컴포넌트 언마운트 시 타이핑 상태 초기화
  useEffect(() => {
    return () => {
      onTyping?.(false);
    };
  }, [onTyping]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (message.trim() || files.length > 0) {
      // 선택된 카테고리 전달 (문자열 확인)
      onSend(message, selectedCategory || '메뉴얼');
      setMessage('');
      setFiles([]);
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    addFiles(selectedFiles);
  };

  const addFiles = (newFiles) => {
    setFiles(prev => [...prev, ...newFiles]);
  };

  const removeFile = (index) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleFileButtonClick = () => {
    setShowFileUploadModal(true);
  };

  const handleModalClose = () => {
    setShowFileUploadModal(false);
  };

  const handleFileUpload = async (uploadedFiles, category) => {
    try {
      if (uploadedFiles && uploadedFiles.length > 0) {
        addFiles(uploadedFiles);
      }
      
      // 카테고리가 전달된 경우 업데이트
      if (category && typeof category === 'string') {
        setSelectedCategory(category);
      }
      
      // 업로드 성공 콜백 호출
      if (onUploadSuccess) {
        onUploadSuccess(uploadedFiles);
      }
    } catch (error) {
      console.error('파일 업로드 오류:', error);
      alert('파일 업로드 중 오류가 발생했습니다.');
    }
  };

  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
  };

  return (
    <div className="p-3 md:p-4 animate-float-in">
      <div className={`chat-input mx-auto max-w-4xl transition-all duration-300 ${isFocused ? 'shadow-md' : ''}`}>
        {/* 카테고리 선택 영역 추가 */}
        <div className="mb-2 flex justify-start">
          <div className="inline-flex text-xs bg-gray-100 dark:bg-gray-800 rounded-lg overflow-hidden">
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => handleCategoryChange(category)}
                className={`px-3 py-1 transition-colors ${
                  selectedCategory === category 
                    ? 'bg-indigo-500 text-white' 
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>
        
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 p-2 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
            {files.map((file, index) => (
              <div 
                key={index} 
                className="flex items-center gap-1.5 py-1 px-2.5 bg-indigo-100 dark:bg-indigo-900/50 text-indigo-700 dark:text-indigo-300 rounded-full text-sm animate-fade-in-fast shadow-sm"
              >
                <FiFile size={14} />
                <span className="truncate max-w-[150px] font-medium">{file.name}</span>
                <button 
                  onClick={() => removeFile(index)}
                  className="p-0.5 hover:bg-indigo-200 dark:hover:bg-indigo-800 rounded-full transition-colors"
                  title="파일 제거"
                >
                  <FiX size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
        
        <div className="flex items-end gap-2 p-2.5">
          <button
            onClick={handleFileButtonClick}
            className={`p-2.5 rounded-full text-gray-500 hover:text-indigo-600 dark:text-gray-400 dark:hover:text-indigo-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
            disabled={disabled || isUploading}
            title="파일 첨부"
          >
            {isUploading ? (
              <FiLoader className="animate-spin" size={20} />
            ) : (
              <FiPaperclip size={20} />
            )}
          </button>

          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="메시지를 입력하세요..."
              className="w-full resize-none outline-none bg-transparent text-gray-800 dark:text-gray-200 py-2 min-h-[44px] max-h-[150px] placeholder:text-gray-400 dark:placeholder:text-gray-500"
              disabled={disabled}
              rows={1}
            />
            {message.length === 0 && (
              <div className="absolute right-0 bottom-2 text-xs text-gray-400 dark:text-gray-500 pointer-events-none pr-2">
                Enter 키로 전송
              </div>
            )}
          </div>

          <button
            onClick={handleSend}
            className={`p-2.5 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 active:bg-indigo-800 transition-colors ${
              (disabled || (!message.trim() && files.length === 0))
                ? 'opacity-50 cursor-not-allowed'
                : 'shadow-md hover:shadow-lg transform hover:-translate-y-0.5 active:translate-y-0 active:shadow-md'
            }`}
            disabled={disabled || (!message.trim() && files.length === 0)}
            title="메시지 보내기"
          >
            <FiSend size={20} className={message.trim() ? 'animate-appear' : ''} />
          </button>
        </div>
      </div>

      {showFileUploadModal && (
        <FileUpload 
          onClose={handleModalClose} 
          categories={categories}
          onUploadSuccess={handleFileUpload}
          initialCategory={selectedCategory}
        />
      )}

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        multiple
      />
    </div>
  );
});

export default ChatInput;
