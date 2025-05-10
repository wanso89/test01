import { useState, useRef, useEffect, forwardRef, useImperativeHandle, useCallback } from 'react';
import { FiLoader, FiSend, FiPaperclip, FiX, FiCheck, FiImage, FiMessageSquare, FiFile, FiSmile, FiUploadCloud } from 'react-icons/fi';
import FileUpload from './FileUpload';

const ChatInput = forwardRef(({ onSend, disabled, onTyping, onUploadSuccess, isEmbedding }, ref) => {
  const [message, setMessage] = useState('');
  const [showFileUploadModal, setShowFileUploadModal] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('메뉴얼'); // 기본 카테고리
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const categories = ['메뉴얼', '장애보고서', '기술문서', '기타'];
  const typingTimerRef = useRef(null);
  const [isFocused, setIsFocused] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  // 외부에서 드롭된 파일을 처리하는 메서드
  const handleDroppedFiles = useCallback((droppedFiles) => {
    if (droppedFiles && droppedFiles.length > 0) {
      setFiles(droppedFiles);
      setShowFileUploadModal(true);
    }
  }, []);

  useImperativeHandle(ref, () => ({
    focus: () => {
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    },
    clear: () => {
      setMessage('');
    },
    setMessage: (text) => {
      setMessage(text);
      // 다음 틱에 높이 조정
      setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.style.height = 'auto';
          textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
        }
      }, 0);
    },
    // 외부에서 드롭된 파일을 처리하는 메서드 노출
    handleDroppedFiles
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

  // 드래그 앤 드롭 이벤트 핸들러 설정
  useEffect(() => {
    const dropZone = dropZoneRef.current;
    if (!dropZone) return;
    
    const handleDragOver = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
    };
    
    const handleDragEnter = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(true);
    };
    
    const handleDragLeave = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
    };
    
    const handleDrop = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const droppedFiles = Array.from(e.dataTransfer.files);
        // 파일 업로드 모달로 전환하여 카테고리 선택 가능하게 함
        setFiles(droppedFiles);
        setShowFileUploadModal(true);
      }
    };
    
    // 이벤트 리스너 등록
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragenter', handleDragEnter);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    
    // 클린업 함수로 이벤트 리스너 제거
    return () => {
      dropZone.removeEventListener('dragover', handleDragOver);
      dropZone.removeEventListener('dragenter', handleDragEnter);
      dropZone.removeEventListener('dragleave', handleDragLeave);
      dropZone.removeEventListener('drop', handleDrop);
    };
  }, []);

  // 컴포넌트 언마운트 시 타이핑 상태 초기화
  useEffect(() => {
    return () => {
      onTyping?.(false);
    };
  }, [onTyping]);

  // 임베딩 상태 변화 감지하여 파일 목록 초기화
  useEffect(() => {
    // 임베딩이 끝난 경우(false로 변경된 경우) 파일 목록 초기화
    if (isEmbedding === false) {
      setFiles([]);
    }
  }, [isEmbedding]);

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
      // 카테고리가 전달된 경우 업데이트
      if (category && typeof category === 'string') {
        setSelectedCategory(category);
      }
      
      // 업로드 성공 콜백 호출
      if (onUploadSuccess) {
        onUploadSuccess(uploadedFiles);
      }
      
      // 파일 목록 초기화
      setFiles([]);
    } catch (error) {
      console.error('파일 업로드 오류:', error);
      alert('파일 업로드 중 오류가 발생했습니다.');
    }
  };

  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
  };

  return (
    <div 
      className={`p-3 md:p-4 animate-float-in bg-gray-900 ${isDragging ? 'relative' : ''}`}
      ref={dropZoneRef}
    >
      {/* 드래그 앤 드롭 오버레이 */}
      {isDragging && (
        <div className="absolute inset-0 bg-indigo-900/50 backdrop-blur-sm border-2 border-dashed border-indigo-400 rounded-xl z-10 flex items-center justify-center animate-pulse">
          <div className="text-center p-6 bg-gray-800/70 rounded-xl shadow-lg">
            <FiUploadCloud size={40} className="mx-auto text-indigo-400 mb-3" />
            <p className="text-white font-medium">파일을 여기에 놓으세요</p>
            <p className="text-gray-300 text-sm mt-1">업로드를 시작합니다</p>
          </div>
        </div>
      )}
      
      <div className={`chat-input mx-auto max-w-4xl transition-all duration-300 bg-gray-900 ${isFocused ? 'shadow-sm' : ''}`}>
        {/* 카테고리 선택 영역 추가 */}
        <div className="mb-2 flex justify-start">
          <div className="inline-flex text-xs bg-gray-800 dark:bg-gray-800 rounded-lg overflow-hidden">
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => handleCategoryChange(category)}
                className={`px-3 py-1 transition-colors ${
                  selectedCategory === category 
                    ? 'bg-indigo-500 text-white' 
                    : 'text-gray-400 dark:text-gray-400 hover:bg-gray-700 dark:hover:bg-gray-700'
                }`}
              >
                {category}
              </button>
            ))}
          </div>
        </div>
        
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 p-2 bg-gray-800 dark:bg-gray-800">
            {files.map((file, index) => (
              <div 
                key={index} 
                className="flex items-center gap-1.5 py-1 px-2.5 bg-indigo-900/50 dark:bg-indigo-900/50 text-indigo-300 dark:text-indigo-300 rounded-full text-sm animate-fade-in-fast shadow-sm"
              >
                <FiFile size={14} />
                <span className="truncate max-w-[150px] font-medium">{file.name}</span>
                <button 
                  onClick={() => removeFile(index)}
                  className="p-0.5 hover:bg-indigo-800 dark:hover:bg-indigo-800 rounded-full transition-colors"
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
            className={`p-2.5 rounded-full text-gray-400 hover:text-indigo-400 dark:text-gray-400 dark:hover:text-indigo-400 hover:bg-gray-800 dark:hover:bg-gray-800 transition-colors ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
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
              className="block w-full px-4 py-3 bg-gray-800 dark:bg-gray-800 focus:ring-0 focus:outline-none text-gray-200 dark:text-gray-200 rounded-xl resize-none overflow-hidden transition-shadow duration-200 pr-12"
              disabled={disabled}
              rows={1}
            />
            {/* 타이핑 인디케이터 */}
            {message.trim().length > 0 ? (
              <div className="absolute right-3 bottom-3 text-indigo-500 dark:text-indigo-500 animate-pulse">
                <div className="flex items-center space-x-1">
                  <div className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            ) : (
              <div className="absolute right-3 bottom-3 text-xs text-gray-500 dark:text-gray-500 pointer-events-none">
                Enter 키로 전송
              </div>
            )}
          </div>

          <button
            onClick={handleSend}
            className={`p-2.5 rounded-full bg-indigo-600 text-white hover:bg-indigo-700 active:bg-indigo-800 transition-colors ${
              (disabled || (!message.trim() && files.length === 0))
                ? 'opacity-50 cursor-not-allowed'
                : 'shadow-sm transform hover:-translate-y-0.5 active:translate-y-0'
            }`}
            disabled={disabled || (!message.trim() && files.length === 0)}
            title="메시지 보내기"
          >
            <FiSend size={20} className={message.trim() ? 'animate-appear' : ''} />
          </button>
        </div>
        
        {/* 드래그 앤 드롭 힌트 메시지 */}
        <div className="mt-1 text-center">
          <p className="text-xs text-gray-500">파일을 이곳에 끌어다 놓아도 업로드할 수 있습니다</p>
        </div>
      </div>

      {showFileUploadModal && (
        <FileUpload 
          onClose={handleModalClose} 
          categories={categories}
          onUploadSuccess={handleFileUpload}
          initialCategory={selectedCategory}
          initialFiles={files}
          containerSelector="#chat-content-container"
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
