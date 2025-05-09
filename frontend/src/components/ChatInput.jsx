import { useState, useRef, useEffect, forwardRef, useImperativeHandle, useCallback } from 'react';
import { FiLoader, FiSend, FiPaperclip, FiX, FiCheck, FiImage, FiMessageSquare, FiFile } from 'react-icons/fi';
import FileUpload from './FileUpload';

const ChatInput = forwardRef(({ onSend, disabled, onTyping }, ref) => {
  const [message, setMessage] = useState('');
  const [showFileUploadModal, setShowFileUploadModal] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const categories = ['메뉴얼', '장애보고서', '기술문서', '기타'];
  const typingTimerRef = useRef(null);

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
      onSend(message, files);
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

  const handleFileUpload = async (uploadedFiles) => {
    try {
      if (uploadedFiles && uploadedFiles.length > 0) {
        addFiles(uploadedFiles);
      }
    } catch (error) {
      console.error('파일 업로드 오류:', error);
      alert('파일 업로드 중 오류가 발생했습니다.');
    }
  };

  return (
    <div className="p-3 md:p-4">
      <div className="chat-input mx-auto max-w-4xl">
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 p-2 border-b border-gray-700">
            {files.map((file, index) => (
              <div 
                key={index} 
                className="flex items-center gap-1.5 py-1 px-2 bg-indigo-900/50 text-indigo-300 rounded-full text-sm"
              >
                <FiFile size={14} />
                <span className="truncate max-w-[140px]">{file.name}</span>
                <button 
                  onClick={() => removeFile(index)}
                  className="p-0.5 hover:bg-indigo-800 rounded-full"
                >
                  <FiX size={14} />
                </button>
              </div>
            ))}
          </div>
        )}
        
        <div className="flex items-end gap-2 p-2">
          <button
            onClick={handleFileButtonClick}
            className={`p-2.5 rounded-full text-indigo-400 hover:bg-gray-800 transition-colors ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
            disabled={disabled || isUploading}
            title="파일 첨부"
          >
            {isUploading ? (
              <FiLoader className="animate-spin" size={20} />
            ) : (
              <FiPaperclip size={20} />
            )}
          </button>

          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="메시지를 입력하세요..."
            className="flex-1 resize-none outline-none bg-transparent text-gray-200 py-2 min-h-[44px] max-h-[150px]"
            disabled={disabled}
            rows={1}
          />

          <button
            onClick={handleSend}
            className={`p-2 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 transition-colors ${
              (disabled || (!message.trim() && files.length === 0))
                ? 'opacity-50 cursor-not-allowed'
                : 'shadow-sm'
            }`}
            disabled={disabled || (!message.trim() && files.length === 0)}
            title="메시지 보내기"
          >
            <FiSend size={20} />
          </button>
        </div>
      </div>

      {showFileUploadModal && (
        <FileUpload 
          onClose={handleModalClose} 
          categories={categories}
          onUploadSuccess={handleFileUpload}
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
