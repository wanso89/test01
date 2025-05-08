import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import { useRef, useEffect, useState, useMemo, useCallback } from "react";
// FiExternalLink 아이콘 추가
import { FiLoader, FiArrowUp, FiType, FiList, FiX, FiExternalLink, FiTrash2 } from "react-icons/fi"; 
import { FiAlertCircle } from "react-icons/fi";

// 전역 스타일 (CSS-in-JS 방식으로 변경)
const globalStyles = `
  .custom-scrollbar::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  .custom-scrollbar::-webkit-scrollbar-track {
    background: transparent;
    border-radius: 8px;
    margin: 2px;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb {
    background-color: rgba(156, 163, 175, 0.3);
    border-radius: 8px;
    border: 2px solid transparent;
    background-clip: padding-box;
  }
  .custom-scrollbar::-webkit-scrollbar-thumb:hover {
    background-color: rgba(156, 163, 175, 0.5);
  }
  
  /* Dark mode scrollbar */
  @media (prefers-color-scheme: dark) {
    .custom-scrollbar::-webkit-scrollbar-thumb {
      background-color: rgba(156, 163, 175, 0.2);
    }
    .custom-scrollbar::-webkit-scrollbar-thumb:hover {
      background-color: rgba(156, 163, 175, 0.4);
    }
  }
  
  @keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
  }
  
  @keyframes slide-up {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  
  @keyframes fade-in-up {
    from { transform: translateY(10px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  
  @keyframes shrink-fade-out {
    from { transform: scale(1); opacity: 1; }
    to { transform: scale(0.9); opacity: 0; }
  }
  
  .animate-fade-in {
    animation: fade-in 0.2s ease-out forwards;
  }
  
  .animate-slide-up {
    animation: slide-up 0.3s ease-out forwards;
  }
  
  .animate-shrink-fade-out {
    animation: shrink-fade-out 0.3s ease-out forwards;
  }
`;

// 스타일 요소를 DOM에 주입하는 함수
const injectGlobalStyles = () => {
  // 이미 존재하는지 확인
  if (!document.getElementById('custom-chat-styles')) {
    const styleElement = document.createElement('style');
    styleElement.id = 'custom-chat-styles';
    styleElement.innerHTML = globalStyles;
    document.head.appendChild(styleElement);
  }
};

// 삭제 확인 모달 컴포넌트 추가
const DeleteConfirmModal = ({ isOpen, onClose, onConfirm, fileName, isDeleting }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[60] p-4 animate-fade-in">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-5 sm:p-6 max-w-md w-full shadow-2xl animate-slide-up">
        <div className="flex items-start mb-4">
          <div className="flex-shrink-0 text-red-500">
            <FiAlertCircle size={24} />
          </div>
          <div className="ml-3">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              파일 삭제 확인
            </h3>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              <span className="font-medium text-gray-800 dark:text-gray-300">{fileName}</span> 파일을 삭제하시겠습니까?
            </p>
            <p className="mt-1 text-xs text-red-500">
              이 작업은 되돌릴 수 없으며, 파일과 관련된 모든 인덱스 데이터가 삭제됩니다.
            </p>
          </div>
        </div>
        <div className="flex justify-end space-x-3 mt-5">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-300 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-400"
            disabled={isDeleting}
          >
            취소
          </button>
          <button
            onClick={onConfirm}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-opacity-50 transition flex items-center"
            disabled={isDeleting}
          >
            {isDeleting ? (
              <>
                <FiLoader className="animate-spin mr-2" size={16} />
                삭제 중...
              </>
            ) : (
              <>삭제</>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// 파일 목록 모달 컴포넌트 개선
const FileListModal = ({ isOpen, onClose, files, isLoading, error, onDeleteFile }) => {
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [fileToDelete, setFileToDelete] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [deleteAnimation, setDeleteAnimation] = useState(null);
  
  const modalRef = useRef(null);

  // 모달 바깥 클릭 시 닫기
  useEffect(() => {
    function handleClickOutside(event) {
      if (modalRef.current && !modalRef.current.contains(event.target)) {
        if (!deleteConfirmOpen) { // 삭제 확인 모달이 열려있으면 닫지 않음
          onClose();
        }
      }
    }
    
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [isOpen, onClose, deleteConfirmOpen]);

  // 성공/에러 메시지 자동 제거
  useEffect(() => {
    if (successMessage || deleteError) {
      const timer = setTimeout(() => {
        setSuccessMessage(null);
        setDeleteError(null);
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [successMessage, deleteError]);

  const handleDeleteClick = (filename) => {
    setFileToDelete(filename);
    setDeleteConfirmOpen(true);
  };

  const confirmDelete = async () => {
    if (!fileToDelete) return;
    
    setIsDeleting(true);
    setDeleteError(null);
    
    try {
      // 삭제 애니메이션 시작
      setDeleteAnimation(fileToDelete);
      
      const response = await fetch(`http://172.10.2.70:8000/api/delete-file?filename=${encodeURIComponent(fileToDelete)}`, {
        method: 'DELETE',
      });
      
      const data = await response.json();
      
      if (response.ok) {
        if (data.status === "success") {
          setSuccessMessage(`"${cleanFilename(fileToDelete)}" 파일이 삭제되었습니다.`);
          // 애니메이션 완료 후 목록에서 제거
          setTimeout(() => {
            onDeleteFile(fileToDelete); // 부모 컴포넌트에 삭제 알림
            setDeleteAnimation(null);
          }, 500); // 500ms는 애니메이션 지속 시간
        } else if (data.status === "warning") {
          // 경고 메시지도 성공으로 처리하되, 다른 메시지 표시
          setSuccessMessage(`"${cleanFilename(fileToDelete)}" ${data.message}`);
          // 애니메이션 완료 후 목록에서 제거
          setTimeout(() => {
            onDeleteFile(fileToDelete); // 부모 컴포넌트에 삭제 알림
            setDeleteAnimation(null);
          }, 500);
        } else {
          setDeleteError(data.message || '파일 삭제 중 알 수 없는 오류가 발생했습니다.');
          setDeleteAnimation(null);
        }
      } else {
        setDeleteError(data.detail || '파일 삭제 실패');
        setDeleteAnimation(null);
      }
    } catch (err) {
      setDeleteError(`오류 발생: ${err.message}`);
      setDeleteAnimation(null);
    } finally {
      setIsDeleting(false);
      setDeleteConfirmOpen(false);
    }
  };

  if (!isOpen) return null;

  // 파일명에서 UUID 접두사 제거하는 함수
  const cleanFilename = (filename) => {
    if (typeof filename !== 'string') return filename;
    const underscoreIndex = filename.indexOf('_');
    if (underscoreIndex > -1 && underscoreIndex < filename.length - 1) {
      return filename.substring(underscoreIndex + 1);
    }
    return filename; 
  };

  // 파일 클릭 시 새 탭에서 여는 핸들러 함수
  const handleFileClick = (prefixedFilename) => {
    const backendBaseUrl = 'http://172.10.2.70:8000';
    const fileUrl = `${backendBaseUrl}/static/uploads/${encodeURIComponent(prefixedFilename)}`; 
    window.open(fileUrl, '_blank', 'noopener,noreferrer'); 
  };

  return (
    <>
      <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 animate-fade-in"></div>
      <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
        <div 
          ref={modalRef}
          className="bg-white dark:bg-gray-800 rounded-lg p-5 sm:p-6 max-w-lg w-full max-h-[80vh] shadow-2xl animate-slide-up flex flex-col"
        > 
        {/* 모달 헤더 */}
          <div className="flex justify-between items-center mb-4 pb-3 border-b border-gray-200 dark:border-gray-700 shrink-0">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            인덱싱된 파일 목록
          </h3>
            <button 
              onClick={onClose} 
              className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
            <FiX size={20} /> 
          </button>
        </div>
        
          {/* 알림 메시지 영역 */}
          {(successMessage || deleteError) && (
            <div 
              className={`mb-3 p-3 rounded-md text-sm animate-fade-in ${
                successMessage 
                  ? "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400" 
                  : "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400"
              }`}
            >
              {successMessage || deleteError}
            </div>
          )}
          
          {/* 스크롤 영역 */}
          <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
          {isLoading ? (
            <div className="flex justify-center items-center py-8 text-gray-500">
                <FiLoader className="animate-spin text-blue-500 mr-3" size={24} />
                <span>목록 로딩 중...</span>
            </div>
          ) : error ? (
            <p className="text-red-500 text-center py-4 px-2">{error}</p>
          ) : files && files.length > 0 ? (
              <ul className="space-y-2">
              {files.map((prefixedFilename, index) => {
                const displayName = cleanFilename(prefixedFilename);
                  const isDeleting = deleteAnimation === prefixedFilename;
                  
                return (
                    <li 
                      key={index} 
                      className={`bg-white dark:bg-gray-700/30 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700/60 transition-all duration-300
                        ${isDeleting ? 'animate-shrink-fade-out line-through' : ''}
                      `}
                    >
                      <div className="flex items-center justify-between p-3">
                    <button 
                      onClick={() => handleFileClick(prefixedFilename)}
                          className="flex-1 flex items-center text-left group focus:outline-none"
                      title={`새 탭에서 ${displayName} 보기`}
                          disabled={isDeleting}
                        >
                          <span className="truncate mr-2">{displayName}</span>
                          <FiExternalLink className="text-gray-400 dark:text-gray-500 group-hover:text-blue-500 dark:group-hover:text-blue-400 opacity-0 group-hover:opacity-100 transition-opacity" size={16} />
                        </button>
                        <button
                          onClick={() => handleDeleteClick(prefixedFilename)}
                          className={`ml-2 p-2 rounded-full focus:outline-none focus:ring-2 focus:ring-red-500/30 transition-colors
                            ${isDeleting 
                              ? 'bg-red-100 text-red-500 dark:bg-red-900/30 dark:text-red-400 cursor-not-allowed' 
                              : 'text-gray-400 hover:text-red-500 dark:text-gray-500 dark:hover:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-700/80'
                            }
                          `}
                          title="파일 삭제"
                          disabled={isDeleting}
                        >
                          {isDeleting ? (
                            <FiLoader className="animate-spin" size={16} />
                          ) : (
                            <FiTrash2 size={16} />
                          )}
                    </button>
                      </div>
                  </li>
                );
              })}
            </ul>
          ) : (
            <p className="text-gray-400 dark:text-gray-500 text-center py-8">
              인덱싱된 파일이 없습니다.
            </p>
          )}
        </div>

          <div className="pt-4 mt-auto border-t border-gray-200 dark:border-gray-700 mt-3">
          <button
            onClick={onClose}
            className="w-full sm:w-auto px-5 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition self-end float-right"
          >
            닫기
          </button>
        </div>
      </div>
    </div>

      {/* 삭제 확인 모달 */}
      <DeleteConfirmModal
        isOpen={deleteConfirmOpen}
        onClose={() => setDeleteConfirmOpen(false)}
        onConfirm={confirmDelete}
        fileName={fileToDelete ? cleanFilename(fileToDelete) : ''}
        isDeleting={isDeleting}
      />
    </>
  );
};


function ChatContainer({
  scrollLocked,
  activeConversationId,
  messages,
  searchTerm,
  filteredMessages,
  onUpdateMessages,
}) {
  // 글로벌 스타일 주입
  useEffect(() => {
    injectGlobalStyles();
  }, []);

  // --- 상태 변수 선언 ---
  const [localMessages, setLocalMessages] = useState(messages);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  const [isSending, setIsSending] = useState(false);
  const inputRef = useRef(null); 
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isFileListModalOpen, setIsFileListModalOpen] = useState(false);
  const [indexedFiles, setIndexedFiles] = useState([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [fileListError, setFileListError] = useState(null);

  // --- useEffect 및 핸들러 함수들 ---
  useEffect(() => { if (inputRef.current) { inputRef.current.focus(); } }, []);
  useEffect(() => { if (isTyping && messagesEndRef.current) { messagesEndRef.current.scrollIntoView({ behavior: "smooth" }); } }, [isTyping]);
  useEffect(() => { setLocalMessages(messages); }, [messages, activeConversationId]);
  useEffect(() => { if (!scrollLocked) { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); } }, [localMessages, scrollLocked]);
  useEffect(() => { if (!searchTerm) { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); } }, [searchTerm]);
  
  const handleScroll = () => { 
    const container = chatContainerRef.current; 
    if (container) { 
      const scrollTop = container.scrollTop; 
      const scrollHeight = container.scrollHeight; 
      const visibleHeight = container.clientHeight; 
      setShowScrollTop(scrollTop < scrollHeight - visibleHeight - 100); 
    } 
  };
  
  useEffect(() => { 
    const container = chatContainerRef.current; 
    if (container) { 
      container.addEventListener("scroll", handleScroll); 
      handleScroll(); 
      return () => container.removeEventListener("scroll", handleScroll); 
    } 
  }, []);
  
  const scrollToTop = () => { 
    chatContainerRef.current?.scrollTo({ top: 0, behavior: "smooth" }); 
  };
  
  const handleTyping = (typing) => { 
    setIsTyping(typing); 
  };
  
  // handleSend 함수
  const handleSend = async (msg) => { 
    if (!activeConversationId || isSending) return; 
    setIsSending(true); 
    const newMessage = { 
      role: "user", 
      content: msg, 
      reactions: {}, 
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9), 
    }; 
    const updatedMessagesWithUser = [...localMessages, newMessage]; 
    setLocalMessages(updatedMessagesWithUser); 
    onUpdateMessages(updatedMessagesWithUser); 
    setIsLoading(true); 
    setError(null); 
    try { 
      const response = await fetch("http://172.10.2.70:8000/api/chat", { 
        method: "POST", 
        headers: { 
          "Content-Type": "application/json" 
        }, 
        body: JSON.stringify({ 
          question: msg, 
          category: "메뉴얼", 
          history: updatedMessagesWithUser.slice(0, -1).map((m) => ({ 
            role: m.role, 
            content: m.content, 
          })), 
        }), 
      }); 
      if (!response.ok) { 
        const errData = await response.json().catch(() => ({message: `HTTP error ${response.status}`})); 
        throw new Error(errData.message || `FastAPI 호출 실패: ${response.status}`); 
      } 
      const data = await response.json(); 
      const aiResponse = { 
        role: "assistant", 
        content: data.bot_response || "응답을 받아왔습니다.", 
        sources: data.sources || [], 
        questionContext: data.questionContext, 
        reactions: {}, 
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9), 
      }; 
      const updatedMessagesWithAI = [...updatedMessagesWithUser, aiResponse]; 
      setLocalMessages(updatedMessagesWithAI); 
      onUpdateMessages(updatedMessagesWithAI); 
    } catch (err) { 
      setError(`응답을 가져오는 중 오류가 발생했습니다: ${err.message}.`); 
      console.error(err); 
    } finally { 
      setIsLoading(false); 
      setIsSending(false); 
    } 
  };

  // 파일 목록 가져오기
  const fetchIndexedFiles = useCallback(async () => { 
    setIsLoadingFiles(true); 
    setFileListError(null); 
    setIndexedFiles([]); 
    try { 
      const response = await fetch("http://172.10.2.70:8000/api/indexed-files"); 
      if (!response.ok) { 
        const errData = await response.json().catch(() => ({ message: `HTTP ${response.status}` })); 
        throw new Error(errData.detail || errData.message || `파일 목록 로드 실패: ${response.status}`); 
      } 
      const data = await response.json(); 
      if (data.status === "success") { 
        setIndexedFiles(data.files || []); 
        setIsFileListModalOpen(true); 
      } else { 
        throw new Error(data.message || "파일 목록 로드 실패"); 
      } 
    } catch (err) { 
      console.error("파일 목록 로드 오류:", err); 
      setFileListError(`오류: ${err.message}`); 
      setIsFileListModalOpen(true); 
    } finally { 
      setIsLoadingFiles(false); 
    } 
  }, []);

  // 파일 삭제 후 목록 업데이트
  const handleFileDeleted = useCallback((deletedFileName) => {
    setIndexedFiles(prev => prev.filter(filename => filename !== deletedFileName));
  }, []);

  // memoizedMessages
  const memoizedMessages = useMemo(() => { 
    return (searchTerm ? filteredMessages : localMessages).map((msg, i) => ( 
      <div 
        key={msg.id || i} 
        className="animate-fade-in-up opacity-0" 
        style={{ 
          animation: "fade-in-up 0.3s ease-out forwards", 
          animationDelay: `${i * 0.1}s`, 
        }}
      > 
        <ChatMessage message={msg} searchTerm={searchTerm || ""} /> 
      </div> 
    )); 
  }, [localMessages, filteredMessages, searchTerm]);

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* 파일 목록 보기 버튼 */}
      <div className="p-2 px-6 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 flex justify-end"> 
      <button
      onClick={fetchIndexedFiles}
      disabled={isLoadingFiles}
          className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-800 font-medium rounded-xl shadow hover:bg-gray-200 hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
    >
          {isLoadingFiles ? (
            <FiLoader className="animate-spin h-5 w-5 text-gray-700 dark:text-gray-300" />
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-700 dark:text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h3.586A1 1 0 018 3.414l1.293 1.293a1 1 0 01.293.707V6h7a1 1 0 011 1v11a1 1 0 01-1 1H4a1 1 0 01-1-1V4z" />
      </svg>
          )}
      인덱싱된 파일 보기
    </button>
      </div>

      {/* 채팅 메시지 목록 */}
      <div 
        ref={chatContainerRef} 
        className="flex-1 overflow-y-auto p-6 space-y-4 bg-gradient-to-br from-transparent to-white/50 dark:from-transparent dark:to-gray-900/50 relative custom-scrollbar" 
        onScroll={handleScroll}
      >
        {memoizedMessages}
        {isLoading && (
          <div className="flex justify-center items-center py-2">
            <FiLoader className="animate-spin text-blue-500 dark:text-blue-400" size={20} />
            <span className="ml-2 text-gray-600 dark:text-gray-400 text-sm">응답을 기다리는 중...</span>
          </div>
        )}
        {isTyping && !isLoading && !isSending && (
          <div className="flex justify-center items-center py-2">
            <FiType className="animate-pulse text-gray-500 dark:text-gray-400" size={20} />
            <span className="ml-2 text-gray-600 dark:text-gray-400 text-sm">입력 중...</span>
          </div>
        )}
        {error && (
          <div className="flex justify-center items-center py-2 text-red-500 dark:text-red-400 text-sm">{error}</div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 맨 위로 스크롤 버튼 */}
      {showScrollTop && (
        <button 
          onClick={scrollToTop} 
          className="fixed bottom-16 right-6 bg-blue-600 text-white rounded-full p-2 shadow-lg hover:bg-blue-700 transition z-20 animate-fade-in" 
          title="맨 위로 이동"
        >
          <FiArrowUp size={20} />
        </button>
      )}
      
      {/* 채팅 입력 컴포넌트 */}
      <div className="sticky bottom-0 z-10 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        <ChatInput 
          onSend={handleSend} 
          disabled={isSending || isLoading} 
          onTyping={handleTyping} 
          ref={inputRef}
        />
      </div>

      {/* 파일 목록 모달 */}
      <FileListModal
        isOpen={isFileListModalOpen}
        onClose={() => setIsFileListModalOpen(false)}
        files={indexedFiles}
        isLoading={isLoadingFiles}
        error={fileListError}
        onDeleteFile={handleFileDeleted}
      />
    </div>
  );
}

export default ChatContainer;