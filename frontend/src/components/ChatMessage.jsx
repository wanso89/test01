import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState, memo } from 'react';
import { FiEye, FiZoomIn, FiCopy, FiCheck, FiThumbsUp, FiThumbsDown, FiStar, FiLoader } from 'react-icons/fi';

function ChatMessage({ message, searchTerm = '' }) {
  const isUser = message.role === 'user';
  const [previewSource, setPreviewSource] = useState(null);
  const [previewContent, setPreviewContent] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null); // 'up', 'down', null
  const [star, setStar] = useState(0); // 별점(1~5)
  const [feedbackSent, setFeedbackSent] = useState(false); // 피드백 전송 여부
  const [loadingContent, setLoadingContent] = useState(false);

  const handleClosePreview = () => {
    setPreviewSource(null);
    setPreviewContent("");
    setPreviewImage(null);
  };

  const handleCopy = () => {
    try {
      navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Clipboard API not supported or failed:", err);
      alert("복사에 실패했습니다. 텍스트를 직접 선택해 복사해주세요.");
    }
  };

  const handleFeedback = async (type) => {
    setFeedback(current => current === type ? null : type);
    if (!feedbackSent) {
      try {
        const response = await fetch('http://172.10.2.70:8000/api/feedback', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            messageId: Date.now().toString(), // 임시 ID, 실제로는 고유 ID 필요
            feedbackType: type, // 'up' or 'down'
            rating: star,
            content: message.content
          })
        });
        if (!response.ok) {
          throw new Error(`피드백 전송 실패: ${response.status} ${response.statusText}`);
        }
        setFeedbackSent(true);
        alert("피드백이 전송되었습니다. 감사합니다!");
      } catch (err) {
        console.error("피드백 전송 중 오류 발생:", err);
        alert("피드백 전송에 실패했습니다. 다시 시도해주세요.");
      }
    }
  };

  const handleStar = (n) => {
    setStar(current => current === n ? 0 : n);
  };

  const handlePreviewSource = async (source) => {
    setPreviewSource(source);
    setLoadingContent(true);
    try {
      const response = await fetch('http://172.10.2.70:8000/api/source-preview', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          path: source.path,
          page: source.page
        })
      });
      if (!response.ok) {
        throw new Error(`문서 내용 불러오기 실패: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      if (data.status === "success") {
        setPreviewContent(data.content); // 변수 이름 일치
      } else {
        setPreviewContent(data.message || "문서 내용을 불러올 수 없습니다."); // 변수 이름 일치
      }
    } catch (err) {
      console.error("문서 내용 불러오기 중 오류 발생:", err);
      setPreviewContent("문서 내용을 불러오는 중 오류가 발생했습니다."); // 변수 이름 일치
    } finally {
      setLoadingContent(false);
    }
  };

  const contentAsString = typeof message.content === 'string' 
    ? message.content 
    : Array.isArray(message.content) 
      ? message.content.join('\n') 
      : String(message.content);

  // 검색 키워드 하이라이트를 위한 커스터마이징 컴포넌트
  const highlightSearchTerm = ({ children }) => {
    if (!searchTerm || typeof children !== 'string') {
      return <span>{children}</span>;
    }
    const regex = new RegExp(`(${searchTerm})`, 'gi');
    const parts = children.split(regex);
    return (
      <span>
        {parts.map((part, index) =>
          part.toLowerCase() === searchTerm.toLowerCase() ? (
            <span key={index} className="bg-yellow-200 dark:bg-yellow-700 text-black dark:text-white">
              {part}
            </span>
          ) : (
            part
          )
        )}
      </span>
    );
  };



  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} my-3`}>
      <div className={`
        relative max-w-xl px-5 py-3 rounded-2xl shadow-lg transition-all duration-200
          ${isUser
            ? 'bg-gradient-to-r from-blue-500 to-indigo-500 text-white rounded-br-none'
            : message.error
              ? 'bg-red-50 border border-red-200 text-red-700 rounded-bl-none'
              : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-100 rounded-bl-none'}
          group hover:shadow-xl hover:scale-[1.02]
        `}>
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
                  <code {...props}>{children}</code>
                );
              },
              img({ src, alt, ...props }) {
                return (
                  <div className="relative inline-block cursor-pointer" onClick={() => setPreviewImage(src)}>
                    <img
                      src={src}
                      alt={alt || '이미지'}
                      className="max-h-40 rounded border border-gray-200 dark:border-gray-700"
                      {...props}
                    />
                    <div className="absolute inset-0 flex items-center justify-center opacity-0 hover:opacity-100 transition bg-black/30 rounded">
                      <FiZoomIn className="text-white" size={24} />
                    </div>
                  </div>
                );
              },
              table({ children, ...props }) {
                return (
                  <div className="overflow-x-auto">
                    <table className="min-w-full border border-gray-200 dark:border-gray-700 rounded" {...props}>
                      {children}
                    </table>
                  </div>
                );
              },
              th({ children, ...props }) {
                return (
                  <th className="px-3 py-2 border border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 font-semibold" {...props}>
                    {children}
                  </th>
                );
              },
              td({ children, ...props }) {
                return (
                  <td className="px-3 py-2 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-200" {...props}>
                    {children}
                  </td>
                );
              },
              p: highlightSearchTerm,
              span: highlightSearchTerm
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
        {/* 피드백/별점 버튼 (AI 응답에만 표시) */}
        {message.role === 'assistant' && (
          <div className="flex gap-2 mt-2 items-center opacity-0 group-hover:opacity-100 transition">
            <button
              className={`ml-2 ${feedback === 'up' ? 'text-green-500' : 'text-gray-400'} hover:text-green-600`}
              onClick={() => handleFeedback('up')}
              title="좋아요"
            >
              <FiThumbsUp size={16} />
            </button>
            <button
              className={`ml-1 ${feedback === 'down' ? 'text-red-500' : 'text-gray-400'} hover:text-red-600`}
              onClick={() => handleFeedback('down')}
              title="별로예요"
            >
              <FiThumbsDown size={16} />
            </button>
            {/* 별점 */}
            <span className="ml-2 flex">
              {[1, 2, 3, 4, 5].map(n => (
                <button
                  key={n}
                  className={`p-0.5 ${star >= n ? 'text-yellow-400' : 'text-gray-300'} hover:text-yellow-500`}
                  onClick={() => handleStar(n)}
                  title={`${n}점`}
                  style={{ background: 'none', border: 'none', cursor: 'pointer' }}
                >
                  <FiStar fill={star >= n ? '#facc15' : 'none'} size={16} />
                </button>
              ))}
            </span>
          </div>
        )}
        {/* 참고문서 카드 */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">참고 문서:</div>
            <div className="flex flex-wrap gap-2">
              {message.sources.map((source, idx) => (
                <div
                  key={idx}
                  className="text-xs bg-gray-100 dark:bg-gray-700 hover:bg-blue-100 dark:hover:bg-blue-800 border border-gray-200 dark:border-gray-600 rounded px-2 py-1 cursor-pointer transition shadow-sm flex items-center"
                  onClick={() => handlePreviewSource(source)}
                  title="클릭하여 원문 미리보기"
                >
                  {source.path} {source.page && `(p.${source.page})`}
                  <FiEye className="ml-1 text-blue-500 dark:text-blue-400" size={12} />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 원문 미리보기 모달 */}
      {previewSource && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4 animate-fade-in">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto shadow-2xl animate-slide-up">
            <h3 className="text-lg font-bold text-gray-800 dark:text-white mb-2">원문 미리보기: {previewSource.path}</h3>
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              {previewSource.page && `페이지: ${previewSource.page}`}
            </div>
            <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded border border-gray-200 dark:border-gray-600 text-gray-800 dark:text-gray-200">
              {loadingContent ? (
                <div className="flex justify-center items-center py-2">
                  <FiLoader className="animate-spin text-blue-500 dark:text-blue-400" size={20} />
                  <span className="ml-2 text-gray-600 dark:text-gray-400 text-sm">문서 내용을 불러오는 중...</span>
                </div>
              ) : (
                <p>{previewContent || "문서 내용을 불러올 수 없습니다."}</p>
              )}
            </div>
            <button
              onClick={handleClosePreview}
              className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            >
              닫기
            </button>
          </div>
        </div>
      )}

      {/* 이미지 미리보기 모달 */}
      {previewImage && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4 animate-fade-in" onClick={handleClosePreview}>
          <div className="relative max-w-4xl w-full h-full flex items-center justify-center animate-slide-up">
            <img
              src={previewImage}
              alt="이미지 확대 보기"
              className="max-w-full max-h-full rounded shadow-2xl"
            />
            <button
              onClick={handleClosePreview}
              className="absolute top-2 right-2 bg-black/50 text-white rounded-full p-2 hover:bg-black/70 transition"
            >
              닫기
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
export default memo(ChatMessage);
