import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import rehypeRaw from "rehype-raw";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { useState, memo, useCallback, useMemo, useEffect, useRef } from "react";
import {
  FiEye,
  FiZoomIn,
  FiCopy,
  FiCheck,
  FiThumbsUp,
  FiThumbsDown,
  FiStar,
  FiLoader,
  FiExternalLink,
  FiUser,
  FiServer,
  FiX,
  FiClock,
  FiInfo,
  FiLink
} from "react-icons/fi";

const KOREAN_STOPWORDS = new Set([
  "이",
  "가",
  "을",
  "를",
  "은",
  "는",
  "에",
  "에서",
  "로",
  "으로",
  "과",
  "와",
  "도",
  "의",
  "들",
  "좀",
  "등",
  "및",
  "그",
  "저",
  "것",
  "수",
  "알려줘",
  "궁금해",
  "대한",
  "대해",
  "내용",
  "무엇인가요",
  "뭔가요",
  "뭐야",
  "설명해줘",
  "알고싶어",
]);

function ChatMessage({ message, searchTerm = "", isSearchMode }) {
  const isUser = message.role === "user";
  const [previewSource, setPreviewSource] = useState(null);
  const [previewContent, setPreviewContent] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null); // 'up', 'down', null
  const [star, setStar] = useState(0); // 별점(1~5)
  const [feedbackSent, setFeedbackSent] = useState(false); // 피드백 전송 여부
  const [loadingContent, setLoadingContent] = useState(false);
  const [highlightKeywords, setHighlightKeywords] = useState(""); // 모달 하이라이트용 키워드
  const [sourcesVisible, setSourcesVisible] = useState(false);
  
  // 메시지 시간을 참조로 저장
  const messageTime = useMemo(() => {
    // 메시지가 타임스탬프를 가지고 있으면 그것을 사용, 아니면 현재 시간 생성
    return message.timestamp || new Date().getTime();
  }, [message.timestamp]);

  const formatMessageTime = (timestamp) => {
    if (!timestamp) return '';
    
    // timestamp가 숫자 또는 문자열인 경우 처리
    let dateObj;
    if (typeof timestamp === 'number' || !isNaN(parseInt(timestamp))) {
      dateObj = new Date(timestamp);
    } else if (typeof timestamp === 'string') {
      dateObj = new Date(timestamp);
    } else {
      return '';
    }
    
    // 올바른 날짜가 아닌 경우 빈 문자열 반환
    if (isNaN(dateObj.getTime())) return '';
    
    // 현재 시간과의 차이 계산
    const now = new Date();
    const diffMs = now - dateObj;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    // 상대적 시간 표시
    if (diffMins < 1) {
      return '방금 전';
    } else if (diffMins < 60) {
      return `${diffMins}분 전`;
    } else if (diffHours < 24) {
      return `${diffHours}시간 전`;
    } else if (diffDays < 7) {
      return `${diffDays}일 전`;
    } else {
      // 7일 이상 지난 경우 날짜 표시
      return dateObj.toLocaleDateString('ko-KR', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      });
    }
  };

  const handleClosePreview = useCallback(() => {
    setPreviewSource(null);
    setPreviewContent("");
    setPreviewImage(null);
    setHighlightKeywords([]);
  }, []);

  const handleCopy = useCallback(() => {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(message.content).then(
          () => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
          },
          () => {
            console.error("Clipboard API failed");
            fallbackCopyTextToClipboard(message.content);
          }
        );
      } else {
        console.error("Clipboard API not supported");
        fallbackCopyTextToClipboard(message.content);
      }
    } catch (err) {
      console.error("Clipboard API not supported or failed:", err);
      fallbackCopyTextToClipboard(message.content);
    }
  }, [message.content]);

  // Clipboard API가 지원되지 않을 때 사용할 대체 복사 방법
  const fallbackCopyTextToClipboard = useCallback((text) => {
    try {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.style.top = "0";
      textArea.style.left = "0";
      textArea.style.position = "fixed";
      document.body.appendChild(textArea);
      textArea.focus();
      textArea.select();
      const successful = document.execCommand("copy");
      document.body.removeChild(textArea);
      if (successful) {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } else {
        alert("복사에 실패했습니다. 텍스트를 직접 선택해 복사해주세요.");
      }
    } catch (err) {
      console.error("Fallback copy failed:", err);
      alert("복사에 실패했습니다. 텍스트를 직접 선택해 복사해주세요.");
    }
  }, []);

  const handleFeedback = useCallback(
    async (type) => {
      setFeedback((current) => (current === type ? null : type));
      if (!feedbackSent) {
        try {
          const response = await fetch("http://172.10.2.70:8000/api/feedback", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              messageId: Date.now().toString(),
              feedbackType: type,
              rating: star,
              content: message.content,
            }),
          });
          if (!response.ok) {
            throw new Error(
              `피드백 전송 실패: ${response.status} ${response.statusText}`
            );
          }
          setFeedbackSent(true);
          alert("피드백이 전송되었습니다. 감사합니다!");
        } catch (err) {
          console.error("피드백 전송 중 오류 발생:", err);
          alert("피드백 전송에 실패했습니다. 다시 시도해주세요.");
        }
      }
    },
    [feedbackSent, message.content, star]
  );

  const handleStar = useCallback((n) => {
    setStar((current) => (current === n ? 0 : n));
  }, []);

  // 키워드 추출 함수 (간단한 버전)
  const extractKeywords = useCallback((text) => {
    if (!text || typeof text !== 'string' || text.trim() === "") return [];
    const words = text.toLowerCase().split(/[\s\.,\?!;\(\)\[\]\{\}"""'']+/)
                       .filter(word => word.length > 0 && !KOREAN_STOPWORDS.has(word));
    const uniqueKeywords = [...new Set(words)].filter(kw => kw.length > 0).slice(0, 5);
    return uniqueKeywords;
  }, []);

  const handlePreviewSource = useCallback(async (source) => {
    // 1. 상태 초기화 및 클릭된 source 객체 저장
    setPreviewSource(source); 
    setLoadingContent(true); // 미리보기 내용 로딩 시작
    setHighlightKeywords([]); // 이전 하이라이트 키워드 초기화
    setPreviewContent(null); // 이전 미리보기 내용 초기화

    console.log("[handlePreviewSource] 함수 시작. 클릭된 source 객체:", source);

    // 2. 하이라이트 키워드 추출을 위한 원본 텍스트 결정
    let textForKeywordExtraction = ""; 
    console.log("[handlePreviewSource] 키워드 추출 대상 결정 로직 시작");
    console.log("  - message.role:", message.role);
    console.log("  - message.questionContext:", message.questionContext);
    console.log("  - searchTerm:", searchTerm);

    if (message.role === 'assistant' && message.questionContext) {
      // 우선순위 1: AI 응답 메시지이고, 원본 사용자 질문(questionContext)이 있는 경우
      textForKeywordExtraction = message.questionContext;
      console.log("  - 결정: message.questionContext 사용");
    } else if (searchTerm && searchTerm.trim() !== "") {
      // 우선순위 2: questionContext가 없지만, 채팅방 상단 검색어(searchTerm)가 있는 경우
      textForKeywordExtraction = searchTerm;
      console.log("  - 결정: searchTerm 사용");
    } else if (message.role === 'user') {
      // 우선순위 3: 사용자 메시지 자체 (출처 클릭 시나리오에서는 거의 해당 안 됨)
      textForKeywordExtraction = message.content;
       console.log("  - 결정: message.content 사용 (사용자 메시지)");
    }

    // 3. 키워드 추출 및 저장
    const keywords = extractKeywords(textForKeywordExtraction);
    console.log("[handlePreviewSource] 추출된 키워드:", keywords);
    setHighlightKeywords(keywords);

    // 4. API 호출하여 출처 내용 가져오기
    try {
      console.log("[handlePreviewSource] API 호출 시작");
      console.log("  - 요청 URL: /api/source-preview");
      console.log(`  - 요청 내용: path=${source.path}, page=${source.page}, chunk_id=${source.chunk_id}`);

      const response = await fetch('http://172.10.2.70:8000/api/source-preview', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          path: source.path,
          page: source.page,
          chunk_id: source.chunk_id,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`미리보기 API 오류: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      console.log("[handlePreviewSource] API 응답:", data);

      if (data.status === 'success' && data.content) {
        setPreviewContent(data.content);
      } else {
        setPreviewContent(data.message || "내용을 불러올 수 없습니다.");
      }
    } catch (error) {
      console.error("[handlePreviewSource] 오류 발생:", error);
      setPreviewContent("출처 내용을 불러오는 중 오류가 발생했습니다. 다시 시도해주세요.");
    } finally {
      setLoadingContent(false); // 로딩 종료
    }
  }, [message.role, message.questionContext, message.content, searchTerm, extractKeywords]);

  const toggleSourcesVisible = useCallback(() => {
    setSourcesVisible((prevVisible) => !prevVisible);
  }, []);

  const renderPreviewModal = () => {
    if (!previewSource) return null;
    
    return (
      <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center z-50 p-4 animate-fade-in">
        <div className="bg-white dark:bg-gray-800 rounded-xl w-full max-w-3xl max-h-[90vh] flex flex-col overflow-hidden shadow-soft-2xl">
          <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/80">
            <h3 className="font-medium flex items-center text-gray-700 dark:text-gray-200">
              <FiEye className="mr-2 text-indigo-500" />
              <span className="truncate max-w-md">{previewSource.title || previewSource.path}</span>
              {previewSource.page && <span className="ml-2 text-gray-500 dark:text-gray-400 text-sm">(페이지 {previewSource.page})</span>}
            </h3>
            <button 
              onClick={handleClosePreview}
              className="p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors text-gray-500 dark:text-gray-400"
            >
              <FiX size={20} />
            </button>
          </div>
          
          <div className="flex-1 overflow-auto p-6 custom-scrollbar">
            {loadingContent ? (
              <div className="flex flex-col justify-center items-center h-32 space-y-3">
                <FiLoader className="animate-spin text-indigo-500" size={24} />
                <span className="text-gray-500 dark:text-gray-400 text-sm">내용을 불러오는 중...</span>
              </div>
            ) : previewContent ? (
              <div className="prose prose-sm dark:prose-invert max-w-none">
                {previewContent}
              </div>
            ) : (
              <div className="text-center text-gray-500 dark:text-gray-400 py-8">
                미리보기를 불러올 수 없습니다.
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  const CustomCodeBlock = ({ language, value }) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
      navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    };

    return (
      <div className="relative group">
        <div className="absolute right-2 top-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={handleCopy}
            className="bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-md p-1.5 text-xs font-medium flex items-center shadow-md transition-colors"
          >
            {copied ? <FiCheck size={14} /> : <FiCopy size={14} />}
            <span className="ml-1">{copied ? "복사됨" : "복사"}</span>
          </button>
        </div>
        <SyntaxHighlighter
          language={language}
          style={oneDark}
          wrapLines={true}
          customStyle={{
            margin: '0.5rem 0',
            borderRadius: '0.5rem',
            padding: '1rem',
            fontSize: '0.9rem',
            lineHeight: '1.5',
          }}
        >
          {value}
        </SyntaxHighlighter>
      </div>
    );
  };

  return (
    <>
      {/* 소스 프리뷰 모달 */}
      {previewSource && renderPreviewModal()}

      {/* 메시지 컨테이너 */}
      <div 
        className={`w-full my-4 animate-float-in ${
          isUser ? "flex justify-end" : "flex justify-start"
        }`}
      >
        {/* 프로필 아이콘 (어시스턴트만) */}
        {!isUser && (
          <div className="flex-shrink-0 mr-3 mt-1">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-avatar">
              <FiServer size={15} className="text-white" />
            </div>
          </div>
        )}
        
        {/* 메시지 내용 */}
        <div
          className={`flex flex-col max-w-[85%] md:max-w-[75%] relative ${
            isUser 
              ? "items-end" 
              : "items-start"
          }`}
        >
          {/* 날짜 표시 */}
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1 px-1">
            {formatMessageTime(messageTime)}
          </div>
          
          {/* 메시지 카드 */}
          <div
            className={`message-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'} transform transition-all`}
          >
            {/* 텍스트 내용 */}
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeKatex, rehypeRaw, rehypeHighlight]}
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || "");
                    return !inline && match ? (
                      <div className="relative group">
                        <div className="absolute right-2 top-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(String(children).replace(/\n$/, ""));
                              setCopied(true);
                              setTimeout(() => setCopied(false), 2000);
                            }}
                            className="p-1 rounded-md bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                            title="코드 복사"
                          >
                            {copied ? <FiCheck size={14} /> : <FiCopy size={14} />}
                          </button>
                        </div>
                        <SyntaxHighlighter
                          style={oneDark}
                          language={match[1]}
                          PreTag="div"
                          className="rounded-md overflow-hidden !my-3"
                          showLineNumbers
                          wrapLines
                          {...props}
                        >
                          {String(children).replace(/\n$/, "")}
                        </SyntaxHighlighter>
                      </div>
                    ) : (
                      <code
                        className={`${className} rounded-md bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 text-gray-800 dark:text-gray-200`}
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                  img({ src, alt, ...props }) {
                    return (
                      <img
                        src={src}
                        alt={alt}
                        className="max-w-full h-auto rounded-md"
                        {...props}
                      />
                    );
                  },
                  a({ node, ...props }) {
                    return (
                      <a
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
                        {...props}
                      />
                    );
                  },
                  ul({ node, ...props }) {
                    return <ul className="list-disc pl-5 my-2.5" {...props} />;
                  },
                  ol({ node, ...props }) {
                    return <ol className="list-decimal pl-5 my-2.5" {...props} />;
                  },
                  li({ node, ...props }) {
                    return <li className="my-0.5" {...props} />;
                  },
                  blockquote({ node, ...props }) {
                    return (
                      <blockquote
                        className="border-l-2 border-indigo-300 dark:border-indigo-700 pl-4 my-3 italic text-gray-700 dark:text-gray-300"
                        {...props}
                      />
                    );
                  },
                  table({ node, ...props }) {
                    return (
                      <div className="overflow-x-auto my-3">
                        <table className="border-collapse border border-slate-300 dark:border-slate-700 w-full text-sm" {...props} />
                      </div>
                    );
                  },
                  thead({ node, ...props }) {
                    return <thead className="bg-slate-100 dark:bg-slate-800" {...props} />;
                  },
                  tbody({ node, ...props }) {
                    return <tbody {...props} />;
                  },
                  tr({ node, ...props }) {
                    return <tr className="border-b border-slate-300 dark:border-slate-700" {...props} />;
                  },
                  th({ node, ...props }) {
                    return <th className="border border-slate-300 dark:border-slate-700 p-2 text-left font-medium" {...props} />;
                  },
                  td({ node, ...props }) {
                    return <td className="border border-slate-300 dark:border-slate-700 p-2" {...props} />;
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          </div>

          {/* 소스 목록 */}
          {message.sources && message.sources.length > 0 && (
            <div className="w-full mt-1">
              <button
                onClick={() => setSourcesVisible(!sourcesVisible)}
                className="text-xs flex items-center gap-1 text-gray-500 hover:text-indigo-500 dark:text-gray-400 dark:hover:text-indigo-400 transition-colors mt-1 mb-1 px-1 py-0.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800/50"
              >
                <FiLink size={12} />
                <span>
                  {sourcesVisible ? "출처 숨기기" : `${message.sources.length}개 출처 보기`}
                </span>
              </button>
              
              {sourcesVisible && (
                <div className="animate-fade-in mt-1 mb-2 space-y-1.5">
                  {message.sources.map((source, idx) => (
                    <div
                      key={idx}
                      onClick={() => handlePreviewSource(source)}
                      className="flex items-center cursor-pointer p-2.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-750 text-sm text-gray-700 dark:text-gray-300 transition-all border border-gray-200 dark:border-gray-700 group bg-white/70 dark:bg-gray-800/50 backdrop-blur-sm shadow-sm hover:shadow-md"
                    >
                      <div className="flex-shrink-0 mr-2">
                        <div className="w-6 h-6 bg-indigo-100 dark:bg-indigo-900/50 rounded-full flex items-center justify-center">
                          <span className="text-xs font-medium text-indigo-700 dark:text-indigo-400">
                            {idx + 1}
                          </span>
                        </div>
                      </div>
                      <div className="flex-1 truncate">
                        <div className="truncate font-medium">{source.path || "Unknown"}</div>
                        {source.page && (
                          <div className="text-xs text-gray-500 dark:text-gray-400">
                            페이지: {source.page}
                          </div>
                        )}
                      </div>
                      <FiEye
                        className="text-gray-400 group-hover:text-indigo-500 dark:group-hover:text-indigo-400 ml-2 transform group-hover:scale-110 transition-all"
                        size={16}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* 액션 버튼 */}
          <div
            className={`flex items-center gap-1 mt-1 opacity-0 group-hover:opacity-100 transition-opacity ${
              isUser ? "justify-start" : "justify-end"
            }`}
          >
            <button
              onClick={handleCopy}
              className="p-1.5 rounded-full text-gray-500 hover:text-gray-800 hover:bg-gray-200 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-700 transition-colors"
              title="복사하기"
            >
              {copied ? <FiCheck size={15} /> : <FiCopy size={15} />}
            </button>
            
            {!isUser && (
              <>
                <button
                  onClick={() => handleFeedback("up")}
                  className={`p-1.5 rounded-full transition-colors ${
                    feedback === "up"
                      ? "text-green-500 bg-green-50 dark:bg-green-900/30"
                      : "text-gray-500 hover:text-gray-800 hover:bg-gray-200 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-700"
                  }`}
                  title="좋아요"
                >
                  <FiThumbsUp size={15} />
                </button>
                <button
                  onClick={() => handleFeedback("down")}
                  className={`p-1.5 rounded-full transition-colors ${
                    feedback === "down"
                      ? "text-red-500 bg-red-50 dark:bg-red-900/30"
                      : "text-gray-500 hover:text-gray-800 hover:bg-gray-200 dark:text-gray-400 dark:hover:text-gray-200 dark:hover:bg-gray-700"
                  }`}
                  title="싫어요"
                >
                  <FiThumbsDown size={15} />
                </button>
              </>
            )}
          </div>
        </div>
        
        {/* 프로필 아이콘 (사용자만) */}
        {isUser && (
          <div className="flex-shrink-0 ml-3 mt-1">
            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center shadow-avatar">
              <FiUser size={15} className="text-white" />
            </div>
          </div>
        )}
      </div>
    </>
  );
}

export default memo(ChatMessage);
