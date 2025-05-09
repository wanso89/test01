import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import { useRef, useEffect, useState, useMemo, useCallback } from "react";
// FiExternalLink 아이콘 추가
import { FiLoader, FiArrowUp, FiType, FiList, FiX, FiExternalLink, FiTrash2, FiHardDrive, FiFile, FiFolder, FiSearch } from "react-icons/fi"; 
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

// 인덱싱된 파일 목록 컴포넌트 추가
const IndexedFilesModal = ({ isOpen, onClose }) => {
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [deleteAnimation, setDeleteAnimation] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  
  const modalRef = useRef(null);

  // 파일 목록 가져오기
  useEffect(() => {
    if (isOpen) {
      loadFiles();
    }
  }, [isOpen]);

  // 모달 바깥 클릭 시 닫기
  useEffect(() => {
    function handleClickOutside(event) {
      if (modalRef.current && !modalRef.current.contains(event.target)) {
        onClose();
      }
    }
    
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => {
        document.removeEventListener("mousedown", handleClickOutside);
      };
    }
  }, [isOpen, onClose]);

  // 성공/에러 메시지 자동 제거
  useEffect(() => {
    let timer;
    if (successMessage || deleteError) {
      timer = setTimeout(() => {
        setSuccessMessage(null);
        setDeleteError(null);
      }, 3000);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [successMessage, deleteError]);

  const loadFiles = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch('http://172.10.2.70:8000/api/indexed-files');
      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        setFiles(data.files || []);
      } else {
        throw new Error(data.message || '파일 목록을 불러오는데 실패했습니다.');
      }
    } catch (err) {
      console.error('파일 목록 조회 오류:', err);
      setError('파일 목록을 불러오는데 실패했습니다. 다시 시도해주세요.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteClick = async (filename) => {
    try {
      setIsDeleting(true);
      setDeleteAnimation(filename);
      
      const response = await fetch(`http://172.10.2.70:8000/api/delete-file?filename=${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      });
      
      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        // 애니메이션 효과를 위해 약간의 지연 후 목록에서 제거
        setTimeout(() => {
          setFiles(prev => prev.filter(file => file !== filename));
          setSuccessMessage(`${cleanFilename(filename)} 파일이 삭제되었습니다.`);
          setDeleteAnimation(null);
        }, 300);
      } else {
        throw new Error(data.message || '파일 삭제 중 오류가 발생했습니다.');
      }
    } catch (err) {
      console.error('파일 삭제 오류:', err);
      setDeleteError('파일 삭제에 실패했습니다. 다시 시도해주세요.');
      setDeleteAnimation(null);
    } finally {
      setIsDeleting(false);
    }
  };

  const cleanFilename = (filename) => {
    // UUID와 같은 접두사가 있다면 제거
    const parts = filename.split('_');
    if (parts.length > 1 && parts[0].length >= 20) {
      return parts.slice(1).join('_');
    }
    return filename;
  };

  const getFileIcon = (filename) => {
    const ext = filename.split('.').pop().toLowerCase();
    
    // 파일 확장자에 따라 아이콘 결정
    switch (ext) {
      case 'pdf':
        return <div className="w-8 h-8 rounded-lg bg-red-100 dark:bg-red-900/30 flex items-center justify-center text-red-600 dark:text-red-400">PDF</div>;
      case 'doc':
      case 'docx':
        return <div className="w-8 h-8 rounded-lg bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-blue-600 dark:text-blue-400">DOC</div>;
      case 'xls':
      case 'xlsx':
        return <div className="w-8 h-8 rounded-lg bg-green-100 dark:bg-green-900/30 flex items-center justify-center text-green-600 dark:text-green-400">XLS</div>;
      case 'ppt':
      case 'pptx':
        return <div className="w-8 h-8 rounded-lg bg-orange-100 dark:bg-orange-900/30 flex items-center justify-center text-orange-600 dark:text-orange-400">PPT</div>;
      case 'jpg':
      case 'jpeg':
      case 'png':
      case 'gif':
        return <div className="w-8 h-8 rounded-lg bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center text-purple-600 dark:text-purple-400">IMG</div>;
      case 'txt':
        return <div className="w-8 h-8 rounded-lg bg-gray-100 dark:bg-gray-700 flex items-center justify-center text-gray-600 dark:text-gray-400">TXT</div>;
      default:
        return <div className="w-8 h-8 rounded-lg bg-gray-100 dark:bg-gray-700 flex items-center justify-center text-gray-600 dark:text-gray-400">FILE</div>;
    }
  };

  const handleFileClick = (prefixedFilename) => {
    // 파일 뷰어 URL 생성
    const viewerUrl = `http://172.10.2.70:8000/api/file-viewer/${encodeURIComponent(prefixedFilename)}`;
    // 새 탭에서 열기
    window.open(viewerUrl, '_blank');
  };

  // 검색어로 필터링된 파일 목록
  const filteredFiles = searchQuery 
    ? files.filter(file => cleanFilename(file).toLowerCase().includes(searchQuery.toLowerCase()))
    : files;

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div 
        ref={modalRef}
        className="bg-white dark:bg-gray-800 rounded-2xl max-w-2xl w-full max-h-[80vh] shadow-2xl overflow-hidden animate-slide-up"
      >
        {/* 헤더 */}
        <div className="p-5 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <h2 className="text-xl font-bold text-gray-800 dark:text-white flex items-center">
            <FiFolder className="mr-2 text-indigo-500" size={20} /> 
            <span>인덱싱된 파일 목록</span>
          </h2>
          <button 
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <FiX className="text-gray-500 dark:text-gray-400" size={20} />
          </button>
        </div>
        
        {/* 검색 바 */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="relative">
            <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
              <FiSearch className="text-gray-400" size={16} />
            </div>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="파일 검색..."
              className="w-full pl-10 pr-4 py-2.5 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:focus:ring-indigo-400 text-gray-800 dark:text-gray-200"
            />
          </div>
        </div>
        
        {/* 파일 목록 */}
        <div className="overflow-y-auto custom-scrollbar" style={{ maxHeight: 'calc(80vh - 150px)' }}>
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-12">
              <FiLoader size={36} className="animate-spin text-indigo-500 mb-4" />
              <p className="text-gray-500 dark:text-gray-400">파일 목록을 불러오는 중...</p>
            </div>
          ) : error ? (
            <div className="p-8 text-center">
              <FiAlertCircle size={36} className="mx-auto text-red-500 mb-4" />
              <p className="text-red-500 font-medium">{error}</p>
              <button 
                onClick={loadFiles}
                className="mt-4 px-4 py-2 bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg transition-colors"
              >
                다시 시도
              </button>
            </div>
          ) : filteredFiles.length === 0 ? (
            <div className="p-8 text-center">
              {searchQuery ? (
                <p className="text-gray-500 dark:text-gray-400">검색 결과가 없습니다.</p>
              ) : (
                <>
                  <p className="text-gray-500 dark:text-gray-400 mb-2">인덱싱된 파일이 없습니다.</p>
                  <p className="text-gray-400 dark:text-gray-500 text-sm">파일을 업로드하여 대화에 활용해보세요.</p>
                </>
              )}
            </div>
          ) : (
            <div className="grid grid-cols-1 divide-y divide-gray-200 dark:divide-gray-700">
              {filteredFiles.map((filename) => (
                <div 
                  key={filename}
                  className={`p-4 hover:bg-gray-50 dark:hover:bg-gray-750 transition-all ${
                    deleteAnimation === filename ? 'animate-shrink-fade-out' : ''
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div 
                      className="flex items-center flex-1 cursor-pointer"
                      onClick={() => handleFileClick(filename)}
                    >
                      {getFileIcon(filename)}
                      <div className="ml-3 flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-800 dark:text-white truncate">
                          {cleanFilename(filename)}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button 
                        onClick={() => handleFileClick(filename)}
                        className="p-2 text-indigo-500 hover:bg-indigo-50 dark:hover:bg-indigo-900/30 rounded-full transition-colors"
                        title="새 탭에서 보기"
                      >
                        <FiExternalLink size={16} />
                      </button>
                      <button 
                        onClick={() => handleDeleteClick(filename)}
                        className="p-2 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-full transition-colors"
                        title="파일 삭제"
                        disabled={isDeleting}
                      >
                        {isDeleting && deleteAnimation === filename ? (
                          <FiLoader size={16} className="animate-spin" />
                        ) : (
                          <FiTrash2 size={16} />
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
        
        {/* 상태 메시지 */}
        {(successMessage || deleteError) && (
          <div className={`p-3 m-4 rounded-lg text-sm font-medium ${
            successMessage 
              ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' 
              : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400'
          }`}>
            {successMessage || deleteError}
          </div>
        )}
      </div>
    </div>
  );
};

function ChatContainer({
  scrollLocked,
  activeConversationId,
  messages,
  searchTerm,
  filteredMessages,
  onUpdateMessages,
  isEmbedding,
  onUploadSuccess
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showScrollToTop, setShowScrollToTop] = useState(false);
  const [userInput, setUserInput] = useState("");
  const [category, setCategory] = useState("메뉴얼");
  const containerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const chatInputRef = useRef(null);
  const [fileManagerOpen, setFileManagerOpen] = useState(false);

  // 스타일 주입
  useEffect(() => {
    injectGlobalStyles();
  }, []);

  const scrollToBottom = useCallback(() => {
    if (!containerRef.current || scrollLocked) return;

    // RAF를 사용한 부드러운 스크롤
    requestAnimationFrame(() => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
      }
    });
  }, [scrollLocked]);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      if (container.scrollTop < -200) {
        setShowScrollToTop(true);
      } else {
        setShowScrollToTop(false);
      }
    };

    container.addEventListener("scroll", handleScroll);
    return () => container.removeEventListener("scroll", handleScroll);
  }, []);

  const handleSubmit = async (input, selectedCategory) => {
    if (!input.trim()) return;

    const newMessage = {
      role: "user",
      content: input,
      timestamp: new Date().getTime(),
    };
    
    // 메시지 목록에 사용자 메시지 추가
    const updatedMessages = [...messages, newMessage];
    onUpdateMessages(updatedMessages);
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch("http://172.10.2.70:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: input,
          category: selectedCategory || "메뉴얼",
          history: messages.slice(-10).map(m => ({
            role: m.role,
            content: m.content
          })),
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || "응답을 가져오지 못했습니다.");
      }
      
      // 응답 데이터 처리
      const assistantMessage = {
        role: "assistant",
        content: data.answer || "응답을 불러오지 못했습니다.",
        sources: data.sources || [],
        timestamp: new Date().getTime(),
      };
      
      // 메시지 목록에 어시스턴트 응답 추가
      const finalMessages = [...updatedMessages, assistantMessage];
      onUpdateMessages(finalMessages);
      
    } catch (err) {
      console.error("채팅 요청 오류:", err);
      setError("메시지를 보내는 중 오류가 발생했습니다. 다시 시도해주세요.");
      
      // 에러 메시지를 어시스턴트 응답으로 추가
      const errorMessage = {
        role: "assistant",
        content: "죄송합니다. 요청을 처리하는 중 오류가 발생했습니다. 다시 시도해주세요.",
        timestamp: new Date().getTime(),
      };
      onUpdateMessages([...updatedMessages, errorMessage]);
    } finally {
      setLoading(false);
      setUserInput("");
    }
  };

  const scrollToTop = () => { 
    containerRef.current?.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  };

  // 렌더링할 메시지 결정
  const displayMessages = searchTerm ? filteredMessages : messages;
  
  // 파일 관리 버튼 클릭 핸들러
  const handleFileManager = () => {
    setFileManagerOpen(true);
  };

  return (
    <div className="flex flex-col h-full relative">
      {/* 헤더 */}
      <div className="flex items-center justify-between py-3 px-6 border-b border-gray-800">
        <div className="flex items-center">
          <h2 className="text-base font-medium text-gray-200">
            지식검색봇
          </h2>
        </div>
        {/* 파일 관리 버튼 */}
        <div className="flex gap-2">
          <button
            onClick={handleFileManager}
            className="p-2 text-gray-400 hover:text-indigo-400 hover:bg-gray-800 rounded-full transition-colors"
            title="파일 관리"
            disabled={isEmbedding}
          >
            <FiHardDrive size={18} />
          </button>
        </div>
      </div>

      {/* 메시지 목록 */}
      <div 
        ref={containerRef}
        className="flex-1 overflow-y-auto custom-scrollbar py-4 bg-gradient-to-br from-gray-900 to-gray-850"
      >
        <div className="flex flex-col space-y-1 px-2">
          {displayMessages.map((msg, index) => (
            <ChatMessage
              key={`${activeConversationId}-${index}`}
              message={msg}
              searchTerm={searchTerm}
              isSearchMode={!!searchTerm}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>
        
        {/* 스크롤 위로 이동 버튼 */}
        {showScrollToTop && (
          <button
            className="absolute bottom-24 right-6 p-3 bg-indigo-600 text-white rounded-full shadow-lg opacity-80 hover:opacity-100 transition-opacity transform hover:scale-105"
            onClick={scrollToTop}
          >
            <FiArrowUp size={20} />
          </button>
        )}
      </div>

      {/* 입력 영역 */}
      <div className="p-4 border-t border-gray-800 bg-gray-900">
        <ChatInput
          ref={chatInputRef}
          onSend={handleSubmit}
          disabled={loading || isEmbedding}
          onTyping={(isTyping) => {}}
          onUploadSuccess={onUploadSuccess}
        />
        
        {/* 에러 표시 */}
        {error && (
          <div className="mt-2 text-red-500 text-sm px-2">
            {error}
          </div>
        )}
      </div>
      
      {/* 인덱싱된 파일 관리 모달 */}
      {fileManagerOpen && (
        <IndexedFilesModal 
          isOpen={fileManagerOpen}
          onClose={() => setFileManagerOpen(false)}
        />
      )}
    </div>
  );
}

export default ChatContainer;