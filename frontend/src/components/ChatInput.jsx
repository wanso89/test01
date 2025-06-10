import { useState, useRef, useEffect, forwardRef, useImperativeHandle, useCallback } from 'react';
import { FiLoader, FiSend, FiPaperclip, FiX, FiCheck, FiImage, FiMessageSquare, FiFile, FiUploadCloud, FiStopCircle, FiClock, FiList } from 'react-icons/fi';
import FileUpload from './FileUpload';

const ChatInput = forwardRef(({ onSend, disabled, onTyping, onUploadSuccess, isEmbedding, isStreaming, onStopGeneration }, ref) => {
  const [message, setMessage] = useState('');
  const [showFileUploadModal, setShowFileUploadModal] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [files, setFiles] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('메뉴얼'); // 기본값을 메뉴얼로 설정
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  const typingTimerRef = useRef(null);
  const [isFocused, setIsFocused] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  // 이전 대화 기록 관련 상태 추가
  const [messageHistory, setMessageHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [showHistoryModal, setShowHistoryModal] = useState(false);

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

  // 메시지 히스토리 관리
  useEffect(() => {
    // 로컬 스토리지에서 메시지 히스토리 불러오기
    const savedHistory = localStorage.getItem('messageHistory');
    if (savedHistory) {
      try {
        setMessageHistory(JSON.parse(savedHistory));
      } catch (error) {
        console.error('메시지 히스토리 파싱 오류:', error);
        localStorage.removeItem('messageHistory');
      }
    }
    
    // 응답 완료 후 입력 필드 포커스 이벤트 리스너 추가
    const handleChatInputFocus = () => {
      console.log('ChatInput: 포커스 이벤트 수신됨');
      if (textareaRef.current) {
        // 즉시 포커스 시도
        textareaRef.current.focus();
        
        // 약간의 지연 후 다시 포커스 시도 (DOM 업데이트 후)
        setTimeout(() => {
          if (textareaRef.current) {
            textareaRef.current.focus();
            console.log('ChatInput: 지연 포커스 시도 (100ms)');
          }
        }, 100);
        
        // 더 긴 지연 후 다시 시도 (모든 렌더링 완료 후)
        setTimeout(() => {
          if (textareaRef.current) {
            textareaRef.current.focus();
            console.log('ChatInput: 지연 포커스 시도 (300ms)');
          }
        }, 300);
      }
    };
    
    // 이벤트 리스너 등록
    window.addEventListener('chatInputFocus', handleChatInputFocus);
    
    // 컴포넌트 언마운트 시 이벤트 리스너 제거
    return () => {
      window.removeEventListener('chatInputFocus', handleChatInputFocus);
    };
  }, []);

  const handleKeyDown = (e) => {
    // Shift + Enter는 줄바꿈, Enter는 전송
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (message.trim() && !disabled && !isEmbedding) {
        handleSend();
      }
    }
    
    // 방향키 위/아래로 메시지 히스토리 탐색
    if (e.key === 'ArrowUp' && message === '' && messageHistory.length > 0) {
      // 현재 히스토리 인덱스가 -1이면 첫 번째 항목으로 이동
      if (historyIndex === -1) {
        setHistoryIndex(messageHistory.length - 1);
        setMessage(messageHistory[messageHistory.length - 1]);
      } 
      // 이미 히스토리 탐색 중이면 이전 항목으로 이동
      else if (historyIndex > 0) {
        setHistoryIndex(historyIndex - 1);
        setMessage(messageHistory[historyIndex - 1]);
      }
    } else if (e.key === 'ArrowDown' && historyIndex !== -1) {
      if (historyIndex < messageHistory.length - 1) {
        setHistoryIndex(historyIndex + 1);
        setMessage(messageHistory[historyIndex + 1]);
      } else {
        // 마지막 항목에서 더 아래로 가면 입력창을 비움
        setHistoryIndex(-1);
        setMessage('');
      }
    }
  };

  const handleSend = () => {
    if (!message.trim() || disabled || isEmbedding) return;
    
    // 히스토리에 현재 메시지 추가 (중복 제거)
    const trimmedMessage = message.trim();
    setMessageHistory(prev => {
      const newHistory = prev.filter(msg => msg !== trimmedMessage);
      newHistory.push(trimmedMessage);
      // 최대 50개까지만 저장
      if (newHistory.length > 50) {
        newHistory.shift();
      }
      localStorage.setItem('messageHistory', JSON.stringify(newHistory));
      return newHistory;
    });
    
    onSend(trimmedMessage, selectedCategory);
    setMessage('');
    setHistoryIndex(-1);
    
    // 높이 초기화
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleFileChange = (event) => {
    if (event.target.files && event.target.files.length > 0) {
      addFiles(Array.from(event.target.files));
    }
  };

  const addFiles = (newFiles) => {
    setFiles(newFiles);
    setShowFileUploadModal(true);
  };

  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const handleFileButtonClick = () => {
    setShowFileUploadModal(true);
  };

  const handleModalClose = () => {
    setShowFileUploadModal(false);
  };

  const handleFileUpload = async (uploadedFiles, category) => {
    setShowFileUploadModal(false);
    setSelectedCategory(category);
    
    // 업로드 성공 콜백 호출
    if (onUploadSuccess) {
      onUploadSuccess(uploadedFiles);
    }
    
    // 임베딩 완료 메시지 표시 (실제로는 서버에서 임베딩 완료 신호를 받아야 함)
    console.log(`${uploadedFiles.length}개 파일 업로드 및 임베딩 완료`);
  };

  // 히스토리 모달 토글
  const toggleHistoryModal = () => {
    setShowHistoryModal(!showHistoryModal);
  };

  // 히스토리 항목 선택
  const selectHistoryMessage = (msg) => {
    setMessage(msg);
    setShowHistoryModal(false);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  // 히스토리 모달 컴포넌트
  const HistoryModal = () => {
    if (!showHistoryModal) return null;
    
    return (
      <div className="absolute bottom-full left-0 w-full mb-2 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 max-h-64 overflow-y-auto z-10">
        <div className="p-2 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-750 sticky top-0">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center">
            <FiClock className="mr-2" size={14} />
            이전 메시지 기록
          </h3>
        </div>
        {messageHistory.length === 0 ? (
          <div className="p-3 text-sm text-gray-500 dark:text-gray-400 text-center">
            이전 메시지 기록이 없습니다.
          </div>
        ) : (
          <div className="divide-y divide-gray-100 dark:divide-gray-700">
            {[...messageHistory].reverse().map((msg, index) => (
              <div
                key={index}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 cursor-pointer text-sm text-gray-800 dark:text-gray-200 transition-colors truncate"
                onClick={() => selectHistoryMessage(msg)}
                title={msg}
              >
                {msg}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="relative" ref={dropZoneRef}>
      {/* 히스토리 모달 */}
      <HistoryModal />
      
      <form onSubmit={(e) => { e.preventDefault(); handleSend(); }} className="relative">
        <div className={`relative flex items-center border rounded-xl bg-white dark:bg-gray-800 shadow-sm transition-all ${
          isFocused ? 'border-indigo-500 ring-2 ring-indigo-500/20 dark:ring-indigo-500/10' : 'border-gray-300 dark:border-gray-600'
        } ${isDragging ? 'border-dashed border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30' : ''}`}>
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="메시지를 입력하세요..."
            className="w-full px-4 py-3 max-h-[150px] bg-transparent border-0 resize-none focus:ring-0 focus:outline-none text-gray-800 dark:text-gray-200 placeholder-gray-400 dark:placeholder-gray-500"
            disabled={disabled || isEmbedding}
            rows={1}
            aria-label="메시지 입력"
          />
          
          {/* 버튼 영역 */}
          <div className="absolute right-2 bottom-2 flex items-center space-x-1">
            {/* 히스토리 버튼 */}
            <button
              type="button"
              onClick={toggleHistoryModal}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors"
              aria-label="이전 대화 기록"
              disabled={disabled || messageHistory.length === 0}
            >
              <FiList size={18} />
            </button>
            
            {/* 파일 버튼 */}
            <button
              type="button"
              onClick={handleFileButtonClick}
              className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-full transition-colors"
              aria-label="파일 첨부"
              disabled={disabled || isEmbedding}
            >
              <FiPaperclip size={18} />
            </button>
            
            {/* 중지 버튼 - 스트리밍 중일 때만 표시 */}
            {isStreaming && (
              <button
                type="button"
                onClick={onStopGeneration}
                className="p-2 text-gray-500 hover:text-red-600 dark:text-gray-400 dark:hover:text-red-500 transition-colors"
                aria-label="응답 생성 중지"
                disabled={disabled || isEmbedding}
              >
                <FiStopCircle className="w-6 h-6" />
              </button>
            )}
            
            {/* 전송 버튼 */}
            <button
              type="button"
              onClick={handleSend}
              disabled={disabled || (!message.trim() && files.length === 0)}
              className={`p-2 rounded-full ${
                message.trim() || files.length > 0
                  ? 'bg-indigo-500 hover:bg-indigo-600 text-white'
                  : 'bg-gray-200 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
              } transition-colors`}
              aria-label="전송"
            >
              {disabled ? (
                <FiLoader className="animate-spin" size={18} />
              ) : (
                <FiSend size={18} />
              )}
            </button>
          </div>
        </div>
      </form>
      
      {/* 파일 업로드 모달 */}
      {showFileUploadModal && (
        <FileUpload
          onClose={handleModalClose}
          initialFiles={files}
          onUploadSuccess={handleFileUpload}
          initialCategory={selectedCategory}
        />
      )}
    </div>
  );
});

export default ChatInput;
