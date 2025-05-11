import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import { useRef, useEffect, useState, useMemo, useCallback } from "react";
// FiExternalLink 아이콘 추가
import { FiLoader, FiArrowUp, FiList, FiX, FiExternalLink, FiTrash2, FiHardDrive, FiFile, FiFolder, FiSearch, FiMessageSquare, FiBookmark, FiUploadCloud, FiPlus } from "react-icons/fi"; 
import { FiAlertCircle } from "react-icons/fi";

// 로딩 인디케이터 컴포넌트 추가
const LoadingIndicator = ({ active }) => {
  // 로딩 시간이 길어질 경우 보여줄 다양한 메시지들
  const loadingMessages = [
    "문서를 검색하고 답변을 작성하고 있습니다",
    "관련 정보를 분석하고 있습니다",
    "최적의 답변을 생성하고 있습니다",
    "조금만 더 기다려주세요"
  ];
  
  // 메시지를 주기적으로 변경하기 위한 상태
  const [messageIndex, setMessageIndex] = useState(0);
  
  // 활성화되면 메시지 변경 타이머 시작
  useEffect(() => {
    if (!active) return;
    
    const interval = setInterval(() => {
      setMessageIndex(prev => (prev + 1) % loadingMessages.length);
    }, 3000);
    
    return () => clearInterval(interval);
  }, [active, loadingMessages.length]);
  
  if (!active) return null;
  
  return (
    <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm flex flex-col items-center justify-center z-[60] animate-fade-in">
      <div className="relative flex flex-col items-center p-10 rounded-2xl bg-gray-800/40 backdrop-blur-lg shadow-2xl">
        {/* 메인 로딩 원형 */}
        <div className="w-24 h-24 rounded-full border-4 border-indigo-500/20 relative flex items-center justify-center">
          <div className="absolute inset-0 rounded-full border-4 border-indigo-500 border-t-transparent animate-spin"></div>
          <div className="absolute inset-0 rounded-full border-4 border-indigo-500/10 border-b-transparent animate-ping opacity-30"></div>
          
          {/* 내부 원형 */}
          <div className="absolute inset-5 rounded-full bg-indigo-500/10 flex items-center justify-center">
            <FiMessageSquare className="text-indigo-400 animate-pulse" size={20} />
          </div>
          
          {/* 주변 빛나는 효과 */}
          <div className="absolute -inset-2 rounded-full bg-indigo-500/5 blur-xl"></div>
        </div>
        
        {/* 텍스트 */}
        <div className="mt-8 text-center space-y-2">
          <p className="text-gray-200 font-medium">응답 생성 중...</p>
          <p className="text-gray-400 text-sm min-h-[20px] transition-all duration-500">
            {loadingMessages[messageIndex]}
          </p>
        </div>
        
        {/* 로딩 점 애니메이션 */}
        <div className="flex items-center space-x-2 mt-4">
          <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '200ms' }}></div>
          <div className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce" style={{ animationDelay: '400ms' }}></div>
        </div>
      </div>
    </div>
  );
};

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
  
  // 외부에서 파일 목록 강제 새로고침 이벤트 처리
  useEffect(() => {
    const handleForceRefresh = () => {
      if (isOpen) {
        loadFiles();
      }
    };
    
    window.addEventListener('forceRefreshFiles', handleForceRefresh);
    
    return () => {
      window.removeEventListener('forceRefreshFiles', handleForceRefresh);
    };
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
      console.log('파일 목록 로드 시도...');
      const response = await fetch('http://172.10.2.70:8000/api/indexed-files', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
        // 타임아웃 설정
        signal: AbortSignal.timeout(10000)
      });
      
      const data = await response.json();
      
      if (response.ok && data.status === 'success') {
        console.log('파일 목록 로드 성공:', data.files?.length || 0, '개 항목');
        setFiles(data.files || []);
      } else {
        console.error('API 응답 오류:', data);
        throw new Error(data.detail || data.message || '파일 목록을 불러오는데 실패했습니다.');
      }
    } catch (err) {
      console.error('파일 목록 조회 오류:', err);
      
      // 오류 타입에 따른 메시지 설정
      let errorMsg = '파일 목록을 불러오는데 실패했습니다.';
      
      if (err.name === 'AbortError') {
        errorMsg = '서버 응답 시간이 너무 깁니다. 서버 상태를 확인해주세요.';
      } else if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        errorMsg = '서버에 연결할 수 없습니다. 네트워크 연결 또는 서버 상태를 확인해주세요.';
      } else if (err.message) {
        errorMsg = `${errorMsg} ${err.message}`;
      }
      
      setError(errorMsg);
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

// 북마크 모달 컴포넌트 추가
const BookmarkModal = ({ isOpen, onClose }) => {
  const [bookmarks, setBookmarks] = useState([]);
  const modalRef = useRef(null);
  
  // 북마크 로드
  useEffect(() => {
    if (isOpen) {
      const savedBookmarks = JSON.parse(localStorage.getItem('chat_bookmarks') || '[]');
      setBookmarks(savedBookmarks.sort((a, b) => b.timestamp - a.timestamp)); // 최신순 정렬
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
  
  // 북마크 삭제
  const handleRemoveBookmark = (id) => {
    const updatedBookmarks = bookmarks.filter(bookmark => bookmark.id !== id);
    localStorage.setItem('chat_bookmarks', JSON.stringify(updatedBookmarks));
    setBookmarks(updatedBookmarks);
  };
  
  // 북마크 포맷
  const formatTimestamp = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div
        ref={modalRef}
        className="bg-gray-800 rounded-xl w-full max-w-2xl max-h-[80vh] overflow-hidden shadow-xl border border-gray-700/50 animate-slide-up"
      >
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <h2 className="text-lg font-medium text-gray-100 flex items-center">
            <FiBookmark className="mr-2 text-yellow-500" size={18} />
            북마크된 메시지
          </h2>
          <button
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-700 text-gray-400 hover:text-gray-300 transition-colors"
          >
            <FiX size={20} />
          </button>
        </div>
        
        <div className="overflow-y-auto custom-scrollbar" style={{ maxHeight: 'calc(80vh - 70px)' }}>
          {bookmarks.length === 0 ? (
            <div className="p-8 text-center text-gray-400">
              <FiBookmark size={36} className="mx-auto mb-4 opacity-50" />
              <p>북마크된 메시지가 없습니다.</p>
              <p className="text-sm mt-2 text-gray-500">대화 중 중요한 내용을 북마크해보세요.</p>
            </div>
          ) : (
            <div className="p-4 space-y-3">
              {bookmarks.map((bookmark) => (
                <div 
                  key={bookmark.id}
                  className="p-4 rounded-lg border border-gray-700/50 bg-gray-750 hover:bg-gray-700 transition-colors group"
                >
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-2 ${
                        bookmark.role === 'assistant' 
                          ? 'bg-indigo-600/30 text-indigo-400' 
                          : 'bg-blue-600/30 text-blue-400'
                      }`}>
                        {bookmark.role === 'assistant' ? 
                          <FiMessageSquare size={12} /> : 
                          <FiType size={12} />}
                      </div>
                      <span className="text-sm text-gray-400">
                        {bookmark.role === 'assistant' ? '어시스턴트' : '사용자'} • {formatTimestamp(bookmark.timestamp)}
                      </span>
                    </div>
                    <button
                      onClick={() => handleRemoveBookmark(bookmark.id)}
                      className="p-1 rounded-full text-gray-500 hover:text-red-400 hover:bg-red-900/20 opacity-0 group-hover:opacity-100 transition-opacity"
                      title="북마크 제거"
                    >
                      <FiX size={16} />
                    </button>
                  </div>
                  <div className="pl-8 pr-2 py-1 text-gray-200 text-sm border-l-2 border-gray-600">
                    {/* 내용 길이가 길 경우 잘라서 표시 */}
                    {bookmark.content.length > 300
                      ? `${bookmark.content.substring(0, 300)}...`
                      : bookmark.content}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// 새 대화 시작 모달 컴포넌트 추가
const NewChatModal = ({ isOpen, onClose, onStart }) => {
  const [topic, setTopic] = useState('');
  const [category, setCategory] = useState('메뉴얼');
  const modalRef = useRef(null);
  
  const categories = ['메뉴얼', '장애보고서', '기술문서', '기타'];
  
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
  
  const handleSubmit = (e) => {
    e.preventDefault();
    onStart(topic, category);
    setTopic('');
    onClose();
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div
        ref={modalRef}
        className="bg-gray-800 rounded-xl w-full max-w-md overflow-hidden shadow-xl border border-gray-700/50 animate-slide-up"
      >
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <h2 className="text-lg font-medium text-gray-100 flex items-center">
            <FiMessageSquare className="mr-2 text-indigo-500" size={18} />
            새 대화 시작
          </h2>
          <button
            onClick={onClose}
            className="p-2 rounded-full hover:bg-gray-700 text-gray-400 hover:text-gray-300 transition-colors"
          >
            <FiX size={20} />
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="p-5 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">주제 (선택사항)</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="대화 주제를 입력하세요"
              className="w-full px-4 py-2.5 bg-gray-700 border border-gray-600 rounded-lg text-gray-200 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <p className="mt-1 text-xs text-gray-500">주제를 설정하면 더 명확한 컨텍스트로 대화를 시작할 수 있습니다.</p>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1.5">카테고리</label>
            <div className="flex flex-wrap gap-2">
              {categories.map((cat) => (
                <button
                  key={cat}
                  type="button"
                  onClick={() => setCategory(cat)}
                  className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                    category === cat
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>
          
          <div className="pt-2 flex justify-end">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 mr-2"
            >
              취소
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
            >
              대화 시작
            </button>
          </div>
        </form>
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
  onUploadSuccess,
  onNewConversation,
  fileManagerOpen,
  setFileManagerOpen,
  sidebarOpen
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showScrollToTop, setShowScrollToTop] = useState(false);
  const [userInput, setUserInput] = useState("");
  const [category, setCategory] = useState("메뉴얼");
  const containerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const chatInputRef = useRef(null);
  const [bookmarkModalOpen, setBookmarkModalOpen] = useState(false);
  const [newChatModalOpen, setNewChatModalOpen] = useState(false);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const dropZoneRef = useRef(null);

  // 스타일 주입
  useEffect(() => {
    injectGlobalStyles();
  }, []);
  
  // 메시지 배열 변경 시 스크롤 강제 이동
  useEffect(() => {
    // 메시지가 변경되면 (새 메시지 추가, 메시지 로드 등) 스크롤 강제 이동
    if (!messages.length) return;
    
    // 스크롤 함수
    const scrollToBottomForced = () => {
      // 컨테이너 직접 스크롤
      if (containerRef.current) {
        const scrollHeight = containerRef.current.scrollHeight;
        containerRef.current.scrollTop = scrollHeight;
      }
      
      // messagesEndRef 스크롤
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "auto" });
      }
    };
    
    // 즉시 실행
    scrollToBottomForced();
    
    // 약간의 지연 후 다시 실행 (렌더링 완료 보장)
    setTimeout(scrollToBottomForced, 100);
    setTimeout(scrollToBottomForced, 500);
    setTimeout(scrollToBottomForced, 1000);
    
  }, [messages]); // messages 배열이 변경될 때마다 실행

  // 채팅 스크롤 이벤트 리스너 추가 - 대화창 전환 시 스크롤 자동 이동
  useEffect(() => {
    const handleChatScroll = () => {
      // 스크롤을 최신 메시지로 강제 이동 - 여러 방법 조합
      
      // 1. 컨테이너의 scrollTop 직접 조작 (가장 강력한 방법)
      if (containerRef.current) {
        const scrollHeight = containerRef.current.scrollHeight;
        containerRef.current.scrollTop = scrollHeight;
      }
      
      // 2. messagesEndRef를 이용한 scrollIntoView
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "auto" });
      }
      
      // 3. 시간차를 두고 여러 번 스크롤 시도 (지연시간 다르게)
      const scrollAttempts = [10, 50, 100, 300, 500, 800];
      
      scrollAttempts.forEach(delay => {
        setTimeout(() => {
          // 컨테이너 직접 스크롤
          if (containerRef.current) {
            const scrollHeight = containerRef.current.scrollHeight;
            containerRef.current.scrollTop = scrollHeight;
          }
          
          // messagesEndRef 스크롤
          if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: "auto" });
          }
        }, delay);
      });
    };
    
    // chatScrollToBottom 이벤트 리스너 등록
    window.addEventListener('chatScrollToBottom', handleChatScroll);
    
    // 컴포넌트 마운트 시 자동 스크롤
    handleChatScroll();
    
    return () => {
      window.removeEventListener('chatScrollToBottom', handleChatScroll);
    };
  }, [messages.length]); // messages.length가 변경될 때마다 실행되도록 의존성 배열 설정

  // activeConversationId가 변경될 때마다 자동 스크롤
  useEffect(() => {
    if (!activeConversationId) return;
    
    // 대화 ID가 변경되면 강제 스크롤 실행
    const forceScrollToBottom = () => {
      // 컨테이너 직접 스크롤
      if (containerRef.current) {
        const scrollHeight = containerRef.current.scrollHeight;
        containerRef.current.scrollTop = scrollHeight;
      }
      
      // messagesEndRef 스크롤
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "auto" });
      }
    };
    
    // 즉시 실행
    forceScrollToBottom();
    
    // 타이핑 애니메이션이 끝날 때까지 지속적으로 스크롤 이벤트 트리거 (약 10초간)
    // 초기 1초 동안은 짧은 간격으로 스크롤
    for (let i = 1; i <= 10; i++) {
      setTimeout(forceScrollToBottom, i * 100); // 100ms 간격으로 1초간 실행
    }
    
    // 그 후 10초까지 긴 간격으로 스크롤 지속
    const longDelays = [1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000];
    longDelays.forEach(delay => {
      setTimeout(forceScrollToBottom, delay);
    });
    
  }, [activeConversationId]);

  // 임베딩 완료 후 파일 목록 자동 새로고침
  useEffect(() => {
    if (!isEmbedding) {
      // 임베딩이 끝나면 파일 목록 갱신 (isEmbedding이 true에서 false로 변경될 때)
      const refreshTimeout = setTimeout(() => {
        if (fileManagerOpen) {
          // 이미 열려있으면 목록만 새로고침 (모달 내부의 loadFiles 함수 호출)
          const refreshEvent = new CustomEvent('forceRefreshFiles');
          window.dispatchEvent(refreshEvent);
        }
      }, 1000); // 서버에서 인덱싱 완료 후 약간의 지연시간을 두고 새로고침
      
      return () => clearTimeout(refreshTimeout);
    }
  }, [isEmbedding, fileManagerOpen]);

  // 드래그 앤 드롭 이벤트 핸들러 설정
  useEffect(() => {
    const handleDragOver = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDraggingFile(true);
    };
    
    const handleDragEnter = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDraggingFile(true);
    };
    
    const handleDragLeave = (e) => {
      e.preventDefault();
      e.stopPropagation();
      
      // 부모 요소 밖으로 나갔을 때만 드래그 상태 해제
      const rect = dropZoneRef.current.getBoundingClientRect();
      if (
        e.clientX <= rect.left ||
        e.clientX >= rect.right ||
        e.clientY <= rect.top ||
        e.clientY >= rect.bottom
      ) {
        setIsDraggingFile(false);
      }
    };
    
    const handleDrop = (e) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDraggingFile(false);
      
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        // 파일 업로드 모달 열기
        const droppedFiles = Array.from(e.dataTransfer.files);
        
        // ChatInput 컴포넌트 참조를 통해 메서드 호출
        if (chatInputRef.current && typeof chatInputRef.current.handleDroppedFiles === 'function') {
          console.log('Calling handleDroppedFiles with:', droppedFiles.length, 'files');
          chatInputRef.current.handleDroppedFiles(droppedFiles);
        } else {
          console.error('chatInputRef.current.handleDroppedFiles is not a function or chatInputRef is null');
        }
      }
    };
    
    const dropZone = dropZoneRef.current;
    if (dropZone) {
      dropZone.addEventListener('dragover', handleDragOver);
      dropZone.addEventListener('dragenter', handleDragEnter);
      dropZone.addEventListener('dragleave', handleDragLeave);
      dropZone.addEventListener('drop', handleDrop);
      
      return () => {
        dropZone.removeEventListener('dragover', handleDragOver);
        dropZone.removeEventListener('dragenter', handleDragEnter);
        dropZone.removeEventListener('dragleave', handleDragLeave);
        dropZone.removeEventListener('drop', handleDrop);
      };
    }
  }, []);

  const scrollToBottom = useCallback(() => {
    if (!containerRef.current || scrollLocked) return;

    // 스크롤을 최하단으로 이동하기 위한 다양한 방법 조합
    
    // 1. 컨테이너의 scrollTop 직접 조작
    if (containerRef.current) {
      const scrollHeight = containerRef.current.scrollHeight;
      containerRef.current.scrollTop = scrollHeight;
    }
    
    // 2. messagesEndRef를 이용한 scrollIntoView
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "auto" });
    }
    
    // 3. 약간의 지연 후 한번 더 스크롤 시도
    setTimeout(() => {
      if (containerRef.current) {
        const scrollHeight = containerRef.current.scrollHeight;
        containerRef.current.scrollTop = scrollHeight;
      }
      
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "auto" });
      }
    }, 100);
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

    // 사용자 메시지 추가
    const userMessage = {
      role: "user",
      content: input,
      timestamp: new Date().getTime(),
    };
    
    const updatedMessages = [...messages, userMessage];
    onUpdateMessages(updatedMessages);
    
    // 로딩 상태 활성화 - 약간의 지연 추가 (UI가 부드럽게 전환되도록)
    setTimeout(() => {
      setLoading(true);
    }, 100);
    setError(null);
    
    try {
      // history 배열 처리 - role과 content만 포함
      const history = messages.slice(-10).map(m => ({
        role: m.role,
        content: m.content
      }));
      
      // category가 문자열인지 확인하고 기본값 설정
      const categoryValue = typeof selectedCategory === 'string' && selectedCategory.trim() !== '' 
        ? selectedCategory 
        : "메뉴얼";
      
      console.log('요청 데이터:', {
        question: input,
        category: categoryValue,
        history
      });
      
      const response = await fetch("http://172.10.2.70:8000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: input,
          category: categoryValue,
          history
        }),
      });
      
      // 응답 데이터 로깅 추가
      const responseText = await response.text();
      console.log('원본 응답 텍스트:', responseText);
      
      let data;
      try {
        data = JSON.parse(responseText);
        console.log('파싱된 응답 데이터:', data);
      } catch (parseError) {
        console.error('JSON 파싱 오류:', parseError);
        throw new Error(`서버 응답을 처리할 수 없습니다: ${responseText.substring(0, 100)}...`);
      }
      
      if (!response.ok) {
        // 오류 메시지를 올바르게 문자열로 변환
        const errorMessage = typeof data === 'object' ? 
          (data.detail || data.message || JSON.stringify(data)) : 
          String(data);
        throw new Error(errorMessage);
      }
      
      // 응답 데이터 처리 확인 로깅
      console.log('사용할 응답 키:', {
        bot_response: data.bot_response,
        answer: data.answer,
        sourceType: Array.isArray(data.sources) ? 'array' : typeof data.sources
      });
      
      // 응답 데이터 처리
      const assistantMessage = {
        role: "assistant",
        content: data.bot_response || data.answer || "응답을 불러오지 못했습니다.",
        sources: Array.isArray(data.sources) ? data.sources : [],
        timestamp: new Date().getTime(),
      };
      
      console.log('생성된 어시스턴트 메시지:', assistantMessage);
      
      // 메시지 목록에 어시스턴트 응답 추가
      const finalMessages = [...updatedMessages, assistantMessage];
      onUpdateMessages(finalMessages);
      
    } catch (err) {
      console.error("채팅 요청 오류:", err);
      // 오류 메시지를 명확하게 표시
      const errorMsg = err.message && err.message !== '[object Object]' ? 
        err.message : 
        "서버에서 응답을 처리하는 중 오류가 발생했습니다. 개발자 콘솔을 확인해주세요.";
      
      setError(errorMsg);
      
      // 에러 메시지를 어시스턴트 응답으로 추가
      const errorMessage = {
        role: "assistant",
        content: `죄송합니다. 요청을 처리하는 중 오류가 발생했습니다: ${errorMsg}`,
        timestamp: new Date().getTime(),
      };
      onUpdateMessages([...updatedMessages, errorMessage]);
    } finally {
      // 로딩 상태 비활성화 - 약간의 지연 추가 (UI가 부드럽게 전환되도록)
      setTimeout(() => {
        setLoading(false);
      }, 300);
      setUserInput("");
    }
  };

  const scrollToTop = () => { 
    containerRef.current?.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  };

  // 메시지 렌더링 부분 수정
  const renderMessages = () => {
    const displayMessages = searchTerm ? filteredMessages : messages;
    
    return displayMessages.map((message, index) => {
      const prevMessage = index > 0 ? displayMessages[index - 1] : null;
      const nextMessage = index < displayMessages.length - 1 ? displayMessages[index + 1] : null;
      
      // 고유한 key 생성 (timestamp + index 조합)
      const messageKey = `${message.timestamp || Date.now()}-${index}`;
      
      return (
        <ChatMessage
          key={messageKey}
          message={message}
          searchTerm={searchTerm}
          isSearchMode={!!searchTerm}
          prevMessage={prevMessage}
          nextMessage={nextMessage}
          onAskFollowUp={handleAskFollowUp}
        />
      );
    });
  };

  // 파일 관리 버튼 클릭 핸들러
  const handleFileManager = () => {
    setFileManagerOpen(true);
  };

  // 후속 질문 핸들러
  const handleAskFollowUp = (question) => {
    if (!question || !chatInputRef.current) return;
    
    // 채팅 입력창에 질문 설정 후 자동 포커스
    chatInputRef.current.clear();
    setTimeout(() => {
      if (chatInputRef.current) {
        chatInputRef.current.setMessage(question);
        chatInputRef.current.focus();
      }
    }, 50);
  };
  
  // 북마크 버튼 핸들러
  const handleBookmarkClick = () => {
    setBookmarkModalOpen(true);
  };
  
  // 새 대화 시작 핸들러 - 모달 없이 바로 새 대화 생성으로 수정
  const handleStartNewChat = () => {
    if (onNewConversation) {
      // 모달 대신 바로 기본 제목("새 대화" 또는 "대화 N")으로 대화 생성
      onNewConversation(null, "메뉴얼");
    }
  };

  return (
    <div className="flex flex-col h-full relative bg-gray-900" ref={dropZoneRef}>
      {/* 로딩 인디케이터 추가 */}
      <LoadingIndicator active={loading} />
      
      {/* 파일 드래그 오버레이 */}
      {isDraggingFile && (
        <div className="absolute inset-0 bg-indigo-900/50 backdrop-blur-sm z-50 flex items-center justify-center border-2 border-dashed border-indigo-400 animate-pulse">
          <div className="bg-gray-800/80 backdrop-blur-md p-8 rounded-2xl text-center shadow-2xl">
            <div className="w-20 h-20 bg-indigo-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
              <FiUploadCloud size={40} className="text-indigo-400" />
            </div>
            <h3 className="text-xl font-bold text-white mb-1">파일을 이곳에 놓으세요</h3>
            <p className="text-gray-300">파일을 업로드하여 대화에 활용하세요</p>
          </div>
        </div>
      )}
      
      {/* 헤더 */}
      <div className="h-16 flex items-center justify-between px-6 bg-gray-900 border-b border-gray-800 shadow-sm z-10">
        <div className={`flex items-center ${!sidebarOpen ? 'ml-12' : ''} transition-all duration-300`}>
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 mr-4 flex items-center justify-center shadow-md">
            <FiMessageSquare size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-blue-500 bg-clip-text text-transparent">
              지식 QnA 어시스턴트
            </h1>
            <p className="text-xs text-gray-400 mt-0.5">AI 기반 업무 지능화 솔루션</p>
          </div>
        </div>
        {/* 헤더 버튼 그룹 */}
        <div className="flex gap-2">
          <button
            onClick={handleStartNewChat}
            className="p-2 rounded-xl bg-blue-600/80 text-white hover:bg-blue-600 transition-colors flex items-center gap-2 group shadow-sm"
            title="새 대화 시작"
            disabled={isEmbedding}
          >
            <FiPlus size={18} className="group-hover:scale-110 transition-transform" />
            <span className="text-sm font-medium hidden sm:inline-block">새 대화</span>
          </button>
          <button
            onClick={handleBookmarkClick}
            className="p-2 rounded-xl text-gray-400 hover:text-blue-400 hover:bg-gray-800/70 transition-colors flex items-center gap-2 group"
            title="북마크 보기"
          >
            <FiBookmark size={18} className="group-hover:scale-110 transition-transform" />
            <span className="text-sm hidden sm:inline-block">북마크</span>
          </button>
          <button
            onClick={handleFileManager}
            className="p-2 rounded-xl text-gray-400 hover:text-blue-400 hover:bg-gray-800/70 transition-colors flex items-center gap-2 group"
            title="인덱싱된 파일 관리"
            disabled={isEmbedding}
          >
            <FiHardDrive size={18} className="group-hover:scale-110 transition-transform" />
            <span className="text-sm hidden sm:inline-block">파일 관리</span>
          </button>
        </div>
      </div>

      {/* 메시지 목록 */}
      <div 
        ref={containerRef}
        className="flex-1 overflow-y-auto overflow-x-hidden bg-gray-900 px-4 pt-2 pb-4 custom-scrollbar"
      >
        <div className="flex flex-col space-y-2 max-w-4xl mx-auto">
          {/* 메시지가 없을 때 안내 메시지 */}
          {renderMessages().length === 0 && (
            <div className="flex flex-col items-center justify-center h-full py-10 text-center">
              <div className="w-16 h-16 rounded-full bg-indigo-900/30 flex items-center justify-center mb-4">
                <FiMessageSquare size={28} className="text-indigo-400" />
              </div>
              <h3 className="text-xl font-medium text-gray-300 mb-2">대화를 시작해보세요</h3>
              <p className="text-gray-500 max-w-md">
                질문을 입력하시면 인덱싱된 문서 기반으로 답변해 드립니다.
              </p>
            </div>
          )}
          
          {/* 메시지 목록 */}
          {renderMessages()}
        <div ref={messagesEndRef} />
      </div>

        {/* 스크롤 위로 이동 버튼 */}
        {showScrollToTop && (
          <button
            className="absolute bottom-24 right-6 p-3 bg-indigo-600 text-white rounded-full shadow-lg hover:bg-indigo-700 hover:shadow-glow-sm transition-all transform hover:scale-105 active:scale-95"
            onClick={scrollToTop}
          >
            <FiArrowUp size={20} />
          </button>
        )}
      </div>

      {/* 입력 영역 */}
      <div className="bg-gray-900 relative z-10">
        <div className="max-w-4xl mx-auto w-full">
          <ChatInput
            ref={chatInputRef}
            onSend={handleSubmit}
            disabled={loading || isEmbedding}
            onTyping={(isTyping) => {}}
            onUploadSuccess={onUploadSuccess}
            isEmbedding={isEmbedding}
          />
          
          {/* 로딩 상태 표시 */}
          {loading && (
            <div className="absolute top-0 left-0 w-full h-1 overflow-hidden bg-gray-800">
              <div className="h-full bg-gradient-to-r from-indigo-500 via-blue-500 to-indigo-500 animate-shine bg-[length:200%_100%]" />
            </div>
          )}
          
          {/* 에러 표시 */}
          {error && (
            <div className="mx-auto max-w-4xl mt-1 mb-2 text-red-500 text-sm px-4 animate-fade-in">
              <div className="flex items-center bg-red-950/30 p-2 rounded-lg">
                <FiAlertCircle className="mr-2 flex-shrink-0" size={14} />
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 인덱싱된 파일 관리 모달 */}
      {fileManagerOpen && (
        <IndexedFilesModal 
          isOpen={fileManagerOpen}
          onClose={() => setFileManagerOpen(false)}
        />
      )}
      
      {/* 북마크 모달 */}
      {bookmarkModalOpen && (
        <BookmarkModal
          isOpen={bookmarkModalOpen}
          onClose={() => setBookmarkModalOpen(false)}
        />
      )}
    </div>
  );
}

export default ChatContainer;