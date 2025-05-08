import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import { useRef, useEffect, useState, useMemo, useCallback } from "react";
// FiExternalLink 아이콘 추가
import { FiLoader, FiArrowUp, FiType, FiList, FiX, FiExternalLink } from "react-icons/fi"; 

// 파일 목록 모달 컴포넌트 (내부 로직 수정)
const FileListModal = ({ isOpen, onClose, files, isLoading, error }) => {
  if (!isOpen) return null;

  // 파일명에서 UUID 접두사 제거하는 함수 (제공해주신 코드와 동일)
  const cleanFilename = (filename) => {
    if (typeof filename !== 'string') return filename;
    const underscoreIndex = filename.indexOf('_');
    if (underscoreIndex > -1 && underscoreIndex < filename.length - 1) {
      return filename.substring(underscoreIndex + 1);
    }
    return filename; 
  };

  // !!!!! 파일 클릭 시 새 탭에서 여는 핸들러 함수 추가 !!!!!
  const handleFileClick = (prefixedFilename) => {
    // '/static/uploads/' 경로에 파일이 실제로 존재하고 웹 서버가 해당 경로를 서빙한다고 가정합니다.
    // 만약 백엔드에서 파일을 삭제하거나 다른 경로에 있다면 이 URL은 작동하지 않습니다.
    
    const backendBaseUrl = 'http://172.10.2.70:8000'; // 
    const fileUrl = `${backendBaseUrl}/static/uploads/${encodeURIComponent(prefixedFilename)}`; 
    
    // window.open()을 사용하여 새 탭에서 직접 URL 열기 (SPA 라우터 문제 우회)
    // 세번째 인자 'noopener,noreferrer'는 보안 권장 사항입니다.
    window.open(fileUrl, '_blank', 'noopener,noreferrer'); 
    console.log(`Attempting to open file in new tab: ${fileUrl}`); // 디버깅 로그 추가
  };
  // !!!!! 핸들러 함수 추가 끝 !!!!!

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in">
      {/* 모달 전체 컨테이너: 최대 높이 유지, 내부 요소 세로 정렬 */}
      <div className="bg-white dark:bg-gray-800 rounded-lg p-5 sm:p-6 max-w-lg w-full max-h-[80vh] shadow-2xl animate-slide-up flex flex-col"> 
        {/* 모달 헤더 */}
        <div className="flex justify-between items-center mb-4 pb-3 border-b border-gray-200 dark:border-gray-700 shrink-0"> {/* 헤더는 높이 고정 */}
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            인덱싱된 파일 목록
          </h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 p-1 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
            <FiX size={20} /> 
          </button>
        </div>
        
        {/* !!!!! 스크롤 및 높이 제한 적용 !!!!! */}
        <div className="flex-1 overflow-y-auto py-3 pr-1 space-y-2"> {/* flex-1, overflow-y-auto */}
          {isLoading ? (
            <div className="flex justify-center items-center py-8 text-gray-500">
              <FiLoader className="animate-spin text-blue-500" size={24} />
              <span className="ml-3">목록 로딩 중...</span>
            </div>
          ) : error ? (
            <p className="text-red-500 text-center py-4 px-2">{error}</p>
          ) : files && files.length > 0 ? (
            <ul className=""> {/* space-y 제거 (li 내부 버튼에서 마진 관리) */}
              {files.map((prefixedFilename, index) => {
                const displayName = cleanFilename(prefixedFilename);
                return (
                  <li key={index}> 
                    <button 
                      onClick={() => handleFileClick(prefixedFilename)}
                      className="w-full flex items-center justify-between bg-white dark:bg-gray-700/30 px-3 py-2.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700/60 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition group text-gray-700 dark:text-gray-300 text-sm text-left"
                      title={`새 탭에서 ${displayName} 보기`}
                    >
                      <span className="truncate pr-2">{displayName}</span>
                      <FiExternalLink className="text-gray-400 dark:text-gray-500 group-hover:text-blue-500 dark:group-hover:text-blue-400 shrink-0 opacity-70 group-hover:opacity-100 transition-opacity" size={16} />
                    </button>
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
        {/* !!!!! 수정 끝 !!!!! */}

        <div className="pt-4 mt-auto"> {/* 닫기 버튼 위치 조정 */}
          <button
            onClick={onClose}
            className="w-full sm:w-auto px-5 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition self-end float-right"
          >
            닫기
          </button>
        </div>
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
}) {
  // --- 상태 변수 선언 (제공해주신 코드와 동일하게 유지) ---
  const [localMessages, setLocalMessages] = useState(messages);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);
  // const [input, setInput] = useState(""); // ChatInput에서 관리하므로 제거해도 무방
  const [isSending, setIsSending] = useState(false);
  const inputRef = useRef(null); 
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isFileListModalOpen, setIsFileListModalOpen] = useState(false);
  const [indexedFiles, setIndexedFiles] = useState([]);
  const [isLoadingFiles, setIsLoadingFiles] = useState(false);
  const [fileListError, setFileListError] = useState(null);

  // --- useEffect 및 핸들러 함수들 (제공해주신 코드와 동일하게 유지) ---
  useEffect(() => { if (inputRef.current) { inputRef.current.focus(); } }, []);
  useEffect(() => { if (isTyping && messagesEndRef.current) { messagesEndRef.current.scrollIntoView({ behavior: "smooth" }); } }, [isTyping]);
  useEffect(() => { setLocalMessages(messages); }, [messages, activeConversationId]);
  useEffect(() => { if (!scrollLocked) { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); } }, [localMessages, scrollLocked]);
  useEffect(() => { if (!searchTerm) { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); } }, [searchTerm]);
  const handleScroll = () => { const container = chatContainerRef.current; if (container) { const scrollTop = container.scrollTop; const scrollHeight = container.scrollHeight; const visibleHeight = container.clientHeight; setShowScrollTop(scrollTop < scrollHeight - visibleHeight - 100); } };
  useEffect(() => { const container = chatContainerRef.current; if (container) { container.addEventListener("scroll", handleScroll); handleScroll(); return () => container.removeEventListener("scroll", handleScroll); } }, []);
  const scrollToTop = () => { chatContainerRef.current?.scrollTo({ top: 0, behavior: "smooth" }); };
  const handleTyping = (typing) => { setIsTyping(typing); };
  
  // handleSend 함수: questionContext 포함 버전 (제공해주신 코드와 동일하게 유지)
  const handleSend = async (msg) => { if (!activeConversationId || isSending) return; setIsSending(true); const newMessage = { role: "user", content: msg, reactions: {}, id: Date.now().toString() + Math.random().toString(36).substr(2, 9), }; const updatedMessagesWithUser = [...localMessages, newMessage]; setLocalMessages(updatedMessagesWithUser); onUpdateMessages(updatedMessagesWithUser); setIsLoading(true); setError(null); try { const response = await fetch("http://172.10.2.70:8000/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: msg, category: "메뉴얼", history: updatedMessagesWithUser.slice(0, -1).map((m) => ({ role: m.role, content: m.content, })), }), }); if (!response.ok) { const errData = await response.json().catch(() => ({message: `HTTP error ${response.status}`})); throw new Error(errData.message || `FastAPI 호출 실패: ${response.status}`); } const data = await response.json(); console.log("DEBUG: API 응답 data:", data); const aiResponse = { role: "assistant", content: data.bot_response || "응답을 받아왔습니다.", sources: data.sources || [], questionContext: data.questionContext, reactions: {}, id: Date.now().toString() + Math.random().toString(36).substr(2, 9), }; console.log("DEBUG: 생성된 aiResponse 객체:", aiResponse); const updatedMessagesWithAI = [...updatedMessagesWithUser, aiResponse]; setLocalMessages(updatedMessagesWithAI); onUpdateMessages(updatedMessagesWithAI); } catch (err) { setError(`응답을 가져오는 중 오류가 발생했습니다: ${err.message}.`); console.error(err); } finally { setIsLoading(false); setIsSending(false); } };

  // fetchIndexedFiles 함수 (제공해주신 코드와 동일하게 유지)
  const fetchIndexedFiles = useCallback(async () => { setIsLoadingFiles(true); setFileListError(null); setIndexedFiles([]); try { const response = await fetch("http://172.10.2.70:8000/api/indexed-files"); if (!response.ok) { const errData = await response.json().catch(() => ({ message: `HTTP ${response.status}` })); throw new Error(errData.detail || errData.message || `파일 목록 로드 실패: ${response.status}`); } const data = await response.json(); if (data.status === "success") { setIndexedFiles(data.files || []); setIsFileListModalOpen(true); } else { throw new Error(data.message || "파일 목록 로드 실패"); } } catch (err) { console.error("파일 목록 로드 오류:", err); setFileListError(`오류: ${err.message}`); setIsFileListModalOpen(true); } finally { setIsLoadingFiles(false); } }, []);

  // memoizedMessages (제공해주신 코드와 동일하게 유지)
  const memoizedMessages = useMemo(() => { return (searchTerm ? filteredMessages : localMessages).map((msg, i) => ( <div key={msg.id || i} className="animate-fade-in-up opacity-0" style={{ animation: "fade-in-up 0.3s ease-out forwards", animationDelay: `${i * 0.1}s`, }}> <ChatMessage message={msg} searchTerm={searchTerm || ""} /> </div> )); }, [localMessages, filteredMessages, searchTerm]);

  // --- return JSX (제공해주신 구조에 버튼 및 모달 렌더링 추가) ---
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {/* 파일 목록 보기 버튼 */}
      <div className="p-2 px-6 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 flex justify-end"> 
      <button
      onClick={fetchIndexedFiles}
      disabled={isLoadingFiles}
      className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-800 font-medium rounded-xl shadow hover:bg-gray-200 hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h3.586A1 1 0 018 3.414l1.293 1.293a1 1 0 01.293.707V6h7a1 1 0 011 1v11a1 1 0 01-1 1H4a1 1 0 01-1-1V4z" />
      </svg>
      인덱싱된 파일 보기
    </button>

      </div>

      {/* 채팅 메시지 목록 */}
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-6 space-y-4 bg-gradient-to-br from-transparent to-white/50 dark:from-transparent dark:to-gray-900/50 relative" onScroll={handleScroll}>
        {memoizedMessages}
        {isLoading && (<div className="flex justify-center items-center py-2"><FiLoader className="animate-spin text-blue-500 dark:text-blue-400" size={20} /><span className="ml-2 text-gray-600 dark:text-gray-400 text-sm">응답을 기다리는 중...</span></div>)}
        {isTyping && !isLoading && !isSending && (<div className="flex justify-center items-center py-2"><FiType className="animate-pulse text-gray-500 dark:text-gray-400" size={20} /><span className="ml-2 text-gray-600 dark:text-gray-400 text-sm">입력 중...</span></div>)}
        {error && (<div className="flex justify-center items-center py-2 text-red-500 dark:text-red-400 text-sm">{error}</div>)}
        <div ref={messagesEndRef} />
      </div>

      {/* 맨 위로 스크롤 버튼 */}
      {showScrollTop && (<button onClick={scrollToTop} className="fixed bottom-16 right-6 bg-blue-600 text-white rounded-full p-2 shadow-lg hover:bg-blue-700 transition z-20 animate-fade-in" title="맨 위로 이동"><FiArrowUp size={20} /></button>)}
      
      {/* 채팅 입력 컴포넌트 */}
      <div className="sticky bottom-0 z-10 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        <ChatInput onSend={handleSend} disabled={isSending || isLoading} onTyping={handleTyping} ref={inputRef}/>
      </div>

      {/* 파일 목록 모달 렌더링 */}
      <FileListModal
        isOpen={isFileListModalOpen}
        onClose={() => setIsFileListModalOpen(false)}
        files={indexedFiles}
        isLoading={isLoadingFiles}
        error={fileListError}
      />
    </div>
  );
}
export default ChatContainer; // memo 제거 유지