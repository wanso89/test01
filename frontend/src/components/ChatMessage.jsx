import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useState } from 'react';
import { FiEye, FiZoomIn, FiCopy, FiCheck } from 'react-icons/fi';

function ChatMessage({ message }) {
  const isUser = message.role === 'user';
  const [previewSource, setPreviewSource] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [copied, setCopied] = useState(false);

  const handleClosePreview = () => {
    setPreviewSource(null);
    setPreviewImage(null);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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
            rehypePlugins={[rehypeHighlight, rehypeKatex]}
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
              }
            }}
          >
            {message.content}
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
        {/* 참고문서 카드 */}
        {message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-2 border-t border-gray-200 dark:border-gray-700">
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">참고 문서:</div>
            <div className="flex flex-wrap gap-2">
              {message.sources.map((source, idx) => (
                <div
                  key={idx}
                  className="text-xs bg-gray-100 dark:bg-gray-700 hover:bg-blue-100 dark:hover:bg-blue-800 border border-gray-200 dark:border-gray-600 rounded px-2 py-1 cursor-pointer transition shadow-sm flex items-center"
                  onClick={() => setPreviewSource(source)}
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
              {/* 실제 원문 내용은 백엔드에서 가져오거나 source 객체에 포함시켜야 함 */}
              <p>여기에 {previewSource.path}의 원문 내용이 표시됩니다. (추후 백엔드 연동 필요)</p>
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
export default ChatMessage;
