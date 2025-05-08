import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import rehypeRaw from "rehype-raw";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { useState, memo, useCallback, useMemo } from "react";
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
  FiX
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

function ChatMessage({ message, searchTerm = "" }) {
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

  const renderSources = useCallback(() => {
    if (!message.sources || message.sources.length === 0 || message.role === 'user') {
      return null;
    }

    return (
      <div className="sources-container mt-3 pt-2.5 border-t border-slate-200 dark:border-slate-700">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-2">참고 문서</p>
        <div className="flex flex-wrap gap-2">
          {message.sources.map((source, index) => (
            <button
              key={index}
              className="text-xs py-1.5 px-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 
                text-blue-700 dark:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-800/30 
                transition-colors flex items-center gap-1.5 shadow-sm border border-blue-100 dark:border-blue-800/50"
              onClick={() => handlePreviewSource(source)}
            >
              <span className="whitespace-nowrap overflow-hidden text-ellipsis max-w-[150px]">{source.path}</span>
              <FiExternalLink size={12} className="flex-shrink-0" />
            </button>
          ))}
        </div>
      </div>
    );
  }, [message.sources, message.role, handlePreviewSource]);

  return (
    <div className={`flex items-start gap-4 py-6 group ${isUser ? "justify-end" : ""}`}>
      {/* 사용자/봇 아바타 */}
      {!isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center 
          bg-gradient-to-br from-blue-500 to-blue-600 dark:from-blue-600 dark:to-blue-700 text-white shadow-md">
          <FiServer size={16} />
        </div>
      )}

      {/* 메시지 내용 */}
      <div className={`flex-1 flex flex-col overflow-hidden ${isUser ? "items-end" : "items-start"} max-w-3xl`}>
        <div className={`chat-message relative ${isUser ? "user" : "assistant"} 
          ${isUser ? "bg-blue-50 dark:bg-blue-900/30 border-blue-100 dark:border-blue-800/40" : 
            "bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700"} 
          border shadow-sm rounded-2xl`}>
          <div className="chat-content prose dark:prose-invert max-w-none p-4">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeHighlight, rehypeKatex, rehypeRaw]}
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || "");
                  return !inline && match ? (
                    <div className="code-block relative group/code">
                      <div className="absolute right-2 top-2 flex gap-1 opacity-0 group-hover/code:opacity-100 transition-opacity">
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(String(children).replace(/\n$/, ""));
                            setCopied(true);
                            setTimeout(() => setCopied(false), 2000);
                          }}
                          className="p-1.5 rounded-md bg-slate-700/70 text-white hover:bg-slate-600 transition-colors"
                          title="코드 복사"
                        >
                          {copied ? <FiCheck size={14} /> : <FiCopy size={14} />}
                        </button>
                      </div>
                      <SyntaxHighlighter
                        style={oneDark}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                        className="rounded-md"
                      >
                        {String(children).replace(/\n$/, "")}
                      </SyntaxHighlighter>
                    </div>
                  ) : (
                    <code className={className} {...props}>
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
                  return <ul className="list-disc pl-6 my-3" {...props} />;
                },
                ol({ node, ...props }) {
                  return <ol className="list-decimal pl-6 my-3" {...props} />;
                },
                li({ node, ...props }) {
                  return <li className="my-1" {...props} />;
                },
                blockquote({ node, ...props }) {
                  return (
                    <blockquote
                      className="border-l-2 border-slate-300 dark:border-slate-700 pl-4 my-4 italic"
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
          
          {/* 출처 표시 */}
          {renderSources()}
        </div>
        
        {/* 피드백 버튼 (AI 응답만) */}
        {!isUser && (
          <div className="message-actions mt-2 flex justify-end gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={() => handleFeedback("up")}
              className={`p-1.5 rounded-md text-sm transition-colors ${
                feedback === "up"
                  ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                  : "text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
              }`}
              title="답변이 도움이 됨"
            >
              <FiThumbsUp size={14} />
            </button>
            <button
              onClick={() => handleFeedback("down")}
              className={`p-1.5 rounded-md text-sm transition-colors ${
                feedback === "down"
                  ? "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
                  : "text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
              }`}
              title="답변이 도움이 되지 않음"
            >
              <FiThumbsDown size={14} />
            </button>
            <button
              onClick={handleCopy}
              className={`p-1.5 rounded-md text-sm transition-colors ${
                copied
                  ? "bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"
                  : "text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
              }`}
              title="텍스트 복사"
            >
              {copied ? <FiCheck size={14} /> : <FiCopy size={14} />}
            </button>
          </div>
        )}
        
        {/* 타임스탬프 (옵션) */}
        <div className="text-xs text-slate-400 dark:text-slate-500 mt-1 px-2">
          {new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
        </div>
      </div>
      
      {/* 사용자 아바타 - 오른쪽 정렬 */}
      {isUser && (
        <div className="flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center 
          bg-gradient-to-br from-blue-600 to-indigo-600 dark:from-blue-500 dark:to-indigo-500 text-white shadow-md">
          <FiUser size={16} />
        </div>
      )}

      {/* 미리보기 모달 */}
      {previewSource && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white dark:bg-slate-900 rounded-xl w-full max-w-3xl h-[80vh] max-h-[800px] flex flex-col shadow-xl animate-fade-in">
            <div className="p-4 border-b border-slate-200 dark:border-slate-700 flex justify-between items-center bg-slate-50 dark:bg-slate-800 rounded-t-xl">
              <div className="flex-1">
                <h3 className="text-lg font-medium text-slate-800 dark:text-slate-200">{previewSource.path}</h3>
                <p className="text-sm text-slate-500 dark:text-slate-400">페이지: {previewSource.page} · 출처 ID: {previewSource.chunk_id}</p>
              </div>
              <button
                onClick={handleClosePreview}
                className="p-1.5 rounded-full hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-500 dark:text-slate-400 transition-colors"
              >
                <FiX size={20} />
              </button>
            </div>
            <div className="flex-1 overflow-auto p-6 prose dark:prose-invert max-w-none custom-scrollbar">
              {loadingContent ? (
                <div className="flex justify-center items-center h-full">
                  <div className="flex flex-col items-center gap-3">
                    <FiLoader size={30} className="animate-spin text-blue-500 dark:text-blue-400" />
                    <p className="text-slate-500 dark:text-slate-400">문서 내용을 불러오는 중...</p>
                  </div>
                </div>
              ) : (
                <div className="relative">
                  {highlightKeywords && highlightKeywords.length > 0 && (
                    <div className="absolute top-0 right-0 p-2 bg-slate-100 dark:bg-slate-800 rounded-md shadow-md text-xs">
                      <div className="font-medium text-slate-500 dark:text-slate-400 mb-1">검색 키워드:</div>
                      <div className="flex flex-wrap gap-1">
                        {highlightKeywords.map((kw, idx) => (
                          <span key={idx} className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/30 
                            text-blue-700 dark:text-blue-300 rounded-md">
                            {kw}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {previewContent}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default memo(ChatMessage);
