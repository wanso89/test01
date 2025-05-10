import { useState, useRef, useEffect, forwardRef, useImperativeHandle, useCallback } from 'react';
import { FiLoader, FiSend, FiPaperclip, FiX, FiCheck, FiImage, FiMessageSquare, FiFile, FiSmile, FiUploadCloud } from 'react-icons/fi';
import FileUpload from './FileUpload';

const ChatInput = forwardRef(({ onSend, disabled, onTyping, onUploadSuccess, isEmbedding }, ref) => {
  const [message, setMessage] = useState('');
  const [showFileUploadModal, setShowFileUploadModal] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('메뉴얼'); // 기본 카테고리 (이제 하나만 사용)
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
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

    const handleDragEnter = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!disabled && !isEmbedding) {
        setIsDragging(true);
      }
    };

    const handleDragOver = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!disabled && !isEmbedding) {
        setIsDragging(true);
      }
    };

    const handleDragLeave = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (!e.currentTarget.contains(e.relatedTarget)) {
        setIsDragging(false);
      }
    };

    const handleDrop = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      
      if (disabled || isEmbedding) return;
      
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const droppedFiles = Array.from(e.dataTransfer.files);
        handleDroppedFiles(droppedFiles);
      }
    };

    dropZone.addEventListener('dragenter', handleDragEnter);
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);

    return () => {
      dropZone.removeEventListener('dragenter', handleDragEnter);
      dropZone.removeEventListener('dragover', handleDragOver);
      dropZone.removeEventListener('dragleave', handleDragLeave);
      dropZone.removeEventListener('drop', handleDrop);
    };
  }, [disabled, isEmbedding, handleDroppedFiles]);

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
      // 선택된 카테고리 전달 (항상 '메뉴얼' 사용)
      onSend(message, '메뉴얼');
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
      // 카테고리는 항상 '메뉴얼'로 고정
      setSelectedCategory('메뉴얼');
      
      // 업로드 성공 콜백 호출
      if (onUploadSuccess) {
        onUploadSuccess(uploadedFiles);
      }
      
      setIsUploading(false);
      setShowFileUploadModal(false);
    } catch (error) {
      console.error('파일 업로드 오류:', error);
      alert('파일 업로드 중 오류가 발생했습니다. 다시 시도해주세요.');
      setIsUploading(false);
    }
  };

  // 컴포넌트 렌더링
  return (
    <div ref={dropZoneRef} className={`relative ${isDragging ? 'bg-indigo-900/10 border-2 border-dashed border-indigo-500/50' : ''}`}>
      {/* 파일 업로드 모달 */}
      {showFileUploadModal && (
        <FileUpload
          files={files}
          onClose={handleModalClose}
          onUpload={handleFileUpload}
          onAddFiles={addFiles}
          onRemoveFile={removeFile}
          isLoading={isUploading}
          isEmbedding={isEmbedding}
          showCategories={false} // 카테고리 선택 숨김
          onUploadSuccess={onUploadSuccess}
          categories={['메뉴얼']} // 카테고리 '메뉴얼'로 고정
          initialCategory="메뉴얼"
        />
      )}
      
      {/* 메인 입력창 */}
      <div className={`p-3 bg-gray-850 border border-gray-700/50 rounded-2xl shadow-inner mx-4 mb-4 transition-all duration-200 ${isFocused ? 'bg-gray-800/80' : ''}`}>
        <div className="relative flex items-end">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="메시지를 입력하세요..."
            className="w-full bg-transparent border-none outline-none resize-none text-gray-200 placeholder-gray-500 py-2 pr-24 max-h-[150px] custom-scrollbar"
            disabled={disabled || isEmbedding}
            rows={1}
          />
          
          {/* 버튼 그룹 */}
          <div className="flex space-x-2 absolute right-0 bottom-1">
            {/* 파일 업로드 버튼 - 전송 버튼 옆으로 이동 */}
            <button
              onClick={handleFileButtonClick}
              type="button"
              disabled={disabled || isEmbedding}
              className={`p-2 text-sm rounded-xl transition-colors ${
                disabled || isEmbedding 
                  ? 'text-gray-500 bg-gray-800/40 cursor-not-allowed' 
                  : 'text-gray-300 hover:bg-indigo-600/30 hover:text-indigo-300'
              }`}
              aria-label="파일 첨부"
              title="파일 첨부"
            >
              <FiUploadCloud className="w-5 h-5" />
            </button>
            
            {/* 전송 버튼 */}
            <button
              onClick={handleSend}
              type="button"
              disabled={disabled || (message.trim() === '' && files.length === 0) || isEmbedding}
              className={`p-2 rounded-xl transition-colors ${
                disabled || (message.trim() === '' && files.length === 0) || isEmbedding
                  ? 'text-gray-500 bg-gray-800/40 cursor-not-allowed' 
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white'
              }`}
              aria-label="메시지 보내기"
            >
              {isEmbedding ? (
                <FiLoader className="w-5 h-5 animate-spin" />
              ) : (
                <FiSend className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
        
        {/* 드롭 영역 안내 메시지 - 드래그 중일 때만 표시 */}
        {isDragging && (
          <div className="absolute inset-0 flex items-center justify-center bg-indigo-900/20 rounded-2xl backdrop-blur-sm z-10">
            <div className="text-center p-4 bg-gray-800/80 rounded-xl shadow-lg">
              <FiUploadCloud className="w-10 h-10 mx-auto mb-2 text-indigo-400" />
              <p className="text-gray-200 font-medium">파일을 여기에 놓으세요</p>
              <p className="text-gray-400 text-sm">파일을 자동으로 업로드합니다</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

export default ChatInput;
