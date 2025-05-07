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
    setHighlightKeywords([]); // 올바른 세터 함수명 사용 (setHighlightKeywords)
  }, [setHighlightKeywords]);

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
    // console.log("[extractKeywords] 입력 텍스트:", text);
    const words = text.toLowerCase().split(/[\s\.,\?!;\(\)\[\]\{\}"“”‘’]+/)
                       .filter(word => word.length > 0 && !KOREAN_STOPWORDS.has(word));
    const uniqueKeywords = [...new Set(words)].filter(kw => kw.length > 0).slice(0, 5);
    // console.log("[extractKeywords] 최종 추출 키워드:", uniqueKeywords);
    return uniqueKeywords;
  }, []);

    // ChatMessage.jsx 내부

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
      console.log("  - message.questionContext:", message.questionContext); // 백엔드에서 전달된 값 확인!
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
      } else {
         console.log("  - 결정: 키워드 추출 대상 텍스트를 찾을 수 없음.");
         textForKeywordExtraction = ""; // 모든 조건 불만족 시 빈 문자열 유지
      }
      console.log("[handlePreviewSource] 최종 키워드 추출 대상 텍스트:", textForKeywordExtraction);
  
      // 3. 키워드 추출 및 상태 업데이트
      const keywords = extractKeywords(textForKeywordExtraction); // extractKeywords 함수 호출
      setHighlightKeywords(keywords); // 추출된 키워드 배열로 상태 업데이트
  
      // 4. 미리보기 내용 설정 (content_full 사용 또는 API 폴백)
      if (source && source.content_full) {
        console.log("[handlePreviewSource] source.content_full 사용 가능. 미리보기 내용 설정.");
        setPreviewContent(source.content_full);
        setLoadingContent(false);
      } else {
        // content_full이 없을 경우 API 호출 (폴백 로직)
        console.warn("source.content_full 누락. API 호출 시도.");
        try {
          const reqBody = { path: source.path, page: source.page, chunk_id: source.chunk_id };
          const response = await fetch("http://172.10.2.70:8000/api/source-preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(reqBody),
          });
          if (!response.ok) { 
              const errData = await response.json().catch(() => ({message: `HTTP error ${response.status}`}));
              throw new Error(errData.message || `문서 내용 불러오기 실패: ${response.status}`);
          }
          const data = await response.json();
          if (data.status === "success") {
            console.log("[handlePreviewSource] API 통해 미리보기 내용 로드 성공.");
            setPreviewContent(data.content);
          } else {
            console.error("[handlePreviewSource] API 응답 실패:", data.message);
            setPreviewContent(data.message || "API에서 문서 내용을 불러올 수 없습니다.");
          }
        } catch (err) {
          console.error("[handlePreviewSource] API 호출 중 오류 발생:", err);
          setPreviewContent(`미리보기 로드 오류: ${err.message}`);
        } finally {
          setLoadingContent(false);
        }
      }
    }, 
    // 의존성 배열: message 객체 또는 필요한 속성, searchTerm, extractKeywords, setHighlightKeywords
    [message, searchTerm, extractKeywords, setHighlightKeywords] 
    // message 객체 전체를 넣으면 questionContext 변경 시에도 함수가 재생성됨
    );

  // 모달 내용에서 여러 키워드 하이라이팅 (정규식 사용)
  const getHighlightedText = useCallback((text, keywordsToHighlightArray) => {
    if (!keywordsToHighlightArray || keywordsToHighlightArray.length === 0 || typeof text !== "string") {
      return <p className="whitespace-pre-wrap leading-relaxed">{text}</p>;
    }
    try {
      const validKeywords = keywordsToHighlightArray.filter(kw => kw && kw.trim() !== "");
      if (validKeywords.length === 0) {
        return <p className="whitespace-pre-wrap leading-relaxed">{text}</p>;
      }
      // 키워드를 |로 연결하여 하나의 정규식 생성 (띄어쓰기 등 고려하지 않고 정확히 일치하는 단어만 찾음)
      const escapedKeywords = validKeywords.map(kw => kw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
      const regex = new RegExp(`(${escapedKeywords.join("|")})`, "gi"); 

      // split 사용 시, 구분자(키워드)와 구분자 사이의 텍스트가 번갈아 나옴
      const parts = text.split(regex);
      
      return (
        <p className="whitespace-pre-wrap leading-relaxed">
          {parts.map((part, i) => {
            if (!part) return null; // 빈 문자열 건너뛰기
            // 현재 part가 validKeywords 중 하나와 대소문자 구분 없이 일치하는지 확인
            const isHighlight = validKeywords.some(kw => part.toLowerCase() === kw.toLowerCase()); 
            return isHighlight ? (
              <mark key={i} className="bg-yellow-300 dark:bg-yellow-600 text-black dark:text-white px-0.5 rounded">
                {part}
              </mark>
            ) : (
              // key prop 추가를 위해 span으로 감쌀 수 있음 (선택적)
              <span key={i}>{part}</span>
            );
          })}
        </p>
      );
    } catch (error) {
      console.error("Error in getHighlightedText:", keywordsToHighlightArray, error);
      return <p className="whitespace-pre-wrap leading-relaxed">{text}</p>;
    }
  }, []);

  const contentAsString = useMemo(() => (
    typeof message.content === "string" ? message.content : 
    Array.isArray(message.content) ? message.content.join("\n") : 
    String(message.content)
  ), [message.content]);

  const HighlightMainContent = useCallback(({ children }) => {
    if (!searchTerm || typeof children !== "string") {
      return <span>{children}</span>;
    }
    try {
        const escapedSearchTerm = searchTerm.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        if (!escapedSearchTerm) return <span>{children}</span>;

        const regex = new RegExp(`(${escapedSearchTerm})`, "gi");
        const parts = children.split(regex);
        return (
          <span>
            {parts.map((part, index) =>
              part.toLowerCase() === searchTerm.toLowerCase() ? (
                <span key={index} className="bg-yellow-200 dark:bg-yellow-700 text-black dark:text-white px-0.5 rounded">
                  {part}
                </span>
              ) : ( part )
            )}
          </span>
        );
    } catch (error) {
        console.error("Error in HighlightMainContent (regex issue likely):", error);
        return <span>{children}</span>;
    }
  }, [searchTerm]);

  return (
    <div
      className={`flex w-full ${isUser ? "justify-end" : "justify-start"} my-3`}
    >
      <div
        className={`
        relative max-w-xl px-5 py-3 rounded-2xl shadow-lg transition-all duration-200
          ${
            isUser
              ? "bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-br-none"
              : message.error
              ? "bg-red-50 border border-red-200 text-red-700 rounded-bl-none"
              : "bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-100 rounded-bl-none"
          }
          group hover:shadow-xl hover:scale-[1.02]
        `}
      >
        {/* 말풍선 꼬리 */}
        {!isUser && (
          <span className="absolute left-0 top-3 w-3 h-3 bg-white dark:bg-gray-800 border-b border-l border-gray-200 dark:border-gray-700 rounded-bl-2xl transform -translate-x-1/2 rotate-45"></span>
        )}
        {isUser && (
          <span className="absolute right-0 top-3 w-3 h-3 bg-indigo-500 rounded-br-2xl transform translate-x-1/2 rotate-45"></span>
        )}

        {/* 마크다운/코드 하이라이팅/수식 */}
        <div className="markdown-content prose max-w-none dark:prose-invert">
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[rehypeHighlight, rehypeKatex, rehypeRaw]}
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || "");
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={oneDark}
                    language={match[1]}
                    PreTag="div"
                    {...props}
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                ) : (
                  // 인라인 코드에도 className 전달
                  <code className={className} {...props}>
                    {children}
                  </code>
                );
              },
              img({ src, alt, ...props }) {
                return (
                  <div
                    className="relative inline-block cursor-pointer"
                    onClick={() => setPreviewImage(src)}
                  >
                    <img
                      src={src}
                      alt={alt || "이미지"}
                      className="max-h-40 rounded border border-gray-200 dark:border-gray-700"
                      {...props}
                    />
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition bg-black/30 rounded">
                      <FiZoomIn className="text-white" size={24} />
                    </div>
                  </div>
                );
              },
              table: ({ children, ...props }) => (
                <div className="overflow-x-auto">
                  <table
                    className="min-w-full border border-gray-200 dark:border-gray-700 rounded"
                    {...props}
                  >
                    {children}
                  </table>
                </div>
              ),
              th: ({ children, ...props }) => (
                <th
                  className="px-3 py-2 border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 font-semibold"
                  {...props}
                >
                  {children}
                </th>
              ),
              td: ({ children, ...props }) => (
                <td
                  className="px-3 py-2 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-200"
                  {...props}
                >
                  {children}
                </td>
              ),
              // p 태그의 내용을 HighlightMainContent 컴포넌트로 래핑하여 searchTerm 하이라이트
              p: HighlightMainContent,
              // 필요하다면 span 등 다른 텍스트 컨테이너에도 적용 가능
              // span: HighlightMainContent,
            }}
          >
            {contentAsString}
          </ReactMarkdown>
        </div>

        {/* 메시지 복사 버튼 */}
        <button
          onClick={handleCopy}
          className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition text-gray-500 dark:text-gray-400 hover:text-blue-500 dark:hover:text-blue-400"
          title={copied ? "복사됨!" : "메시지 복사"}
        >
          {copied ? <FiCheck size={16} /> : <FiCopy size={16} />}
        </button>

        {/* 피드백/별점 버튼 (AI 응답에만 표시, 에러 메시지 아닐 때) */}
        {message.role === "assistant" && !message.error && (
          <div className="flex gap-2 mt-2 items-center opacity-0 group-hover:opacity-100 transition">
            <button
              className={`ml-2 ${
                feedback === "up" ? "text-green-500" : "text-gray-400"
              } hover:text-green-600`}
              onClick={() => handleFeedback("up")}
              title="좋아요"
            >
              <FiThumbsUp size={16} />
            </button>
            <button
              className={`ml-1 ${
                feedback === "down" ? "text-red-500" : "text-gray-400"
              } hover:text-red-600`}
              onClick={() => handleFeedback("down")}
              title="별로예요"
            >
              <FiThumbsDown size={16} />
            </button>
            <span className="ml-2 flex">
              {[1, 2, 3, 4, 5].map((n) => (
                <button
                  key={n}
                  className={`p-0.5 ${
                    star >= n ? "text-yellow-400" : "text-gray-300"
                  } hover:text-yellow-500`}
                  onClick={() => handleStar(n)}
                  title={`${n}점`}
                  style={{
                    background: "none",
                    border: "none",
                    cursor: "pointer",
                  }}
                >
                  <FiStar
                    fill={star >= n ? "rgb(250 204 21)" : "none"}
                    size={16}
                  />{" "}
                  {/* fill 색상 직접 지정 */}
                </button>
              ))}
            </span>
          </div>
        )}

        {/* 참고문서 카드 */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
              참고 문서:
            </div>
            <div className="flex flex-wrap gap-2">
              {message.sources.map((source, idx) => (
                <div
                  key={idx}
                  className="text-xs bg-gray-100 dark:bg-gray-700 hover:bg-blue-100 dark:hover:bg-blue-800 border border-gray-200 dark:border-gray-600 rounded px-2 py-1 cursor-pointer transition shadow-sm flex items-center"
                  onClick={() => handlePreviewSource(source)} // source 객체 전체 전달
                  title="클릭하여 원문 미리보기"
                >
                  {/* source.path가 너무 길면 잘라서 표시하는 로직 추가 가능 */}
                  <span
                    className="truncate max-w-[150px] sm:max-w-[200px]"
                    title={source.path}
                  >
                    {source.path}
                  </span>
                  {source.page &&
                    source.page !== "N/A" &&
                    Number(source.page) > 0 &&
                    ` (p.${source.page})`}
                  <FiEye
                    className="ml-1.5 text-blue-500 dark:text-blue-400 shrink-0"
                    size={12}
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>{" "}
      {/* End of message bubble content */}
      {/* 원문 미리보기 모달 */}
      {previewSource && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-5 sm:p-6 max-w-2xl w-full max-h-[85vh] overflow-y-auto shadow-2xl animate-slide-up">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1.5">
              원문 미리보기:{" "}
              <span className="font-normal text-gray-700 dark:text-gray-300">
                {previewSource.path}
              </span>
            </h3>
            <div className="text-xs sm:text-sm text-gray-500 dark:text-gray-400 mb-3">
              {previewSource.page &&
                previewSource.page !== "N/A" &&
                Number(previewSource.page) > 0 &&
                `페이지: ${previewSource.page}`}
              {previewSource.chunk_id &&
                previewSource.chunk_id !== "N/A" &&
                ` (청크 ID: ${previewSource.chunk_id})`}
            </div>
            <div
              className="bg-gray-50 dark:bg-gray-700 p-3 sm:p-4 rounded border border-gray-200 dark:border-gray-600 text-gray-800 dark:text-gray-200 text-sm"
              style={{ minHeight: "100px" }} // 최소 높이 설정
            >
              {loadingContent ? (
                <div className="flex justify-center items-center py-8">
                  <FiLoader
                    className="animate-spin text-blue-500 dark:text-blue-400"
                    size={24}
                  />
                  <span className="ml-2.5 text-gray-600 dark:text-gray-400">
                    문서 내용을 불러오는 중...
                  </span>
                </div>
              ) : previewContent ? (
                getHighlightedText(previewContent, highlightKeywords) // 하이라이팅 적용
              ) : (
                <p className="text-gray-500 dark:text-gray-400">
                  문서 내용을 불러올 수 없습니다.
                </p>
              )}
            </div>
            <button
              onClick={handleClosePreview}
              className="mt-4 w-full sm:w-auto px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 transition"
            >
              닫기
            </button>
          </div>
        </div>
      )}
      {/* 이미지 미리보기 모달 */}
      {previewImage && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in"
          onClick={() => setPreviewImage(null)} // 배경 클릭 시 닫기
        >
          <div
            className="relative max-w-3xl max-h-[90vh] flex animate-slide-up"
            onClick={(e) => e.stopPropagation()} // 모달 내부 클릭 시 닫힘 방지
          >
            <img
              src={previewImage}
              alt="이미지 확대 보기"
              className="max-w-full max-h-full object-contain rounded shadow-2xl"
            />
            <button
              onClick={() => setPreviewImage(null)}
              className="absolute top-2 right-2 bg-black/40 text-white rounded-full p-1.5 hover:bg-black/60 transition focus:outline-none"
              title="닫기"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                  clipRule="evenodd"
                />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
export default memo(ChatMessage);
