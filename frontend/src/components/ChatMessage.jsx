import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
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
  FiLink,
  FiCornerDownRight,
  FiMessageSquare,
  FiCircle,
  FiList,
  FiChevronDown,
  FiChevronRight,
  FiHash,
  FiBookmark,
  FiMessageCircle,
  FiMoreHorizontal,
  FiImage,
  FiMaximize2,
  FiMinimize2,
  FiSend,
  FiSmile,
  FiAlertCircle,
  FiFrown,
  FiHelpCircle,
  FiFile,
  FiSearch
} from "react-icons/fi";

// 키워드 하이라이트 애니메이션을 위한 키프레임 스타일 정의
const keyframesStyle = `
@keyframes highlightPulse {
  0% { background-color: rgba(253, 224, 71, 0.75); box-shadow: 0 0 3px rgba(250, 204, 21, 0.4); }
  50% { background-color: rgba(250, 204, 21, 0.9); box-shadow: 0 0 5px rgba(250, 204, 21, 0.6); }
  100% { background-color: rgba(253, 224, 71, 0.75); box-shadow: 0 0 3px rgba(250, 204, 21, 0.4); }
}

.animate-highlight-pulse {
  animation: highlightPulse 2.5s ease-in-out infinite;
}

.highlight-keyword {
  background-color: rgba(253, 224, 71, 0.8);
  color: rgba(0, 0, 0, 0.95);
  text-shadow: 0 0 0 rgba(0, 0, 0, 0.1);
  border-radius: 3px;
  padding: 2px 4px;
  margin: 0 -1px;
  font-weight: 600;
  position: relative;
  box-shadow: 0 0 3px rgba(250, 204, 21, 0.7);
  display: inline-block;
  border-bottom: 1px solid rgba(234, 179, 8, 0.3);
}

/* 다크 모드 스타일 */
@media (prefers-color-scheme: dark) {
  .highlight-keyword {
    background-color: rgba(253, 224, 71, 0.7);
    color: rgba(0, 0, 0, 0.9);
    box-shadow: 0 0 3px rgba(250, 204, 21, 0.7);
  }
}
`;

// 스타일을 문서에 주입하는 함수
const injectStyleOnce = (id, css) => {
  if (!document.getElementById(id)) {
    const head = document.head || document.getElementsByTagName('head')[0];
    const style = document.createElement('style');
    style.id = id;
    style.appendChild(document.createTextNode(css));
    head.appendChild(style);
  }
};

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

// 타이핑 효과 애니메이션 컴포넌트
const TypeWriter = ({ text, speed = 2, onComplete }) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  
  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
        
        // 주기적으로 스크롤 이벤트 발생 (타이핑 중에도 스크롤 유지)
        if (currentIndex % 50 === 0) {
          window.dispatchEvent(new CustomEvent('chatScrollToBottom'));
        }
      }, speed);
      
      return () => clearTimeout(timeout);
    } else if (!isComplete) {
      setIsComplete(true);
      
      // 타이핑 완료 시 스크롤 이벤트 발생
      window.dispatchEvent(new CustomEvent('chatScrollToBottom'));
      
      if (onComplete) onComplete();
    }
  }, [text, currentIndex, speed, isComplete, onComplete]);
  
  return (
    <div>
      {displayText}
      {currentIndex < text.length && (
        <span className="inline-block w-1.5 h-4 ml-0.5 bg-indigo-400 animate-pulse"></span>
      )}
    </div>
  );
};

// 내용에서 제목 추출 함수
const extractHeadings = (markdownText) => {
  const headings = [];
  const lines = markdownText.split('\n');
  
  lines.forEach((line) => {
    // # 스타일 헤딩 매칭
    const headingMatch = line.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const text = headingMatch[2].trim();
      
      headings.push({
        level,
        text,
        id: text.toLowerCase().replace(/[^\w\s가-힣]/g, '').replace(/\s+/g, '-')
      });
    }
  });
  
  return headings;
};

// 목차 컴포넌트
const TableOfContents = ({ headings, onClickHeading }) => {
  const [isOpen, setIsOpen] = useState(true);
  
  if (!headings || headings.length === 0) return null;
  
  return (
    <div className="my-3 border border-gray-700/40 rounded-lg overflow-hidden">
      <div 
        className="bg-gray-800/80 px-3 py-2 flex items-center justify-between cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="flex items-center text-sm font-medium text-gray-200">
          <FiList className="mr-2 text-indigo-400" size={16} />
          목차
        </div>
        <button className="text-gray-400 hover:text-white transition-colors">
          {isOpen ? <FiChevronDown size={16} /> : <FiChevronRight size={16} />}
        </button>
      </div>
      
      {isOpen && (
        <div className="bg-gray-850/50 p-3">
          <ul className="space-y-1 text-sm">
            {headings.map((heading, idx) => (
              <li key={idx} className="leading-snug">
                <button
                  onClick={() => onClickHeading(heading.id)}
                  className={`hover:text-indigo-400 transition-colors flex items-start ${
                    heading.level === 1 ? "font-medium text-gray-200" :
                    heading.level === 2 ? "pl-4 text-gray-300" : 
                    "pl-8 text-gray-400"
                  }`}
                >
                  <FiHash size={12} className="mr-1 mt-1 flex-shrink-0" />
                  <span className="truncate">{heading.text}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

// 프로필 아바타 컴포넌트
const ProfileAvatar = ({ role, isGrouped }) => {
  const isAssistant = role === 'assistant';
  
  return (
    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
      isGrouped ? 'opacity-0' : ''
    } transition-opacity duration-200 ${
      isAssistant 
        ? 'bg-gradient-to-br from-indigo-500 to-purple-600' 
        : 'bg-gradient-to-br from-blue-500 to-cyan-500'
    }`}>
      {isAssistant ? (
        <FiMessageSquare className="text-white" size={16} />
      ) : (
        <FiUser className="text-white" size={16} />
      )}
    </div>
  );
};

// 이미지 미리보기 컴포넌트
const ImagePreview = ({ src, alt }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    <div className="relative group">
      <img 
        src={src} 
        alt={alt || '이미지'} 
        className={`rounded-lg shadow-md transition-all duration-300 ${
          isExpanded ? 'max-w-none' : 'max-w-md'
        }`}
      />
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="absolute top-2 right-2 p-2 bg-gray-800/80 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
      >
        {isExpanded ? (
          <FiMinimize2 className="text-white" size={16} />
        ) : (
          <FiMaximize2 className="text-white" size={16} />
        )}
      </button>
    </div>
  );
};

// 피드백 모달 컴포넌트
const FeedbackModal = ({ isOpen, onClose, messageContent, onSubmit, feedbackType }) => {
  const [reasons, setReasons] = useState([]);
  const [comment, setComment] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [rating, setRating] = useState(0);
  
  const modalRef = useRef(null);
  
  // 좋아요/싫어요에 따른 피드백 이유 옵션
  const reasonOptions = feedbackType === 'up'
    ? [
        '정확한 정보를 제공함',
        '이해하기 쉽게 설명됨',
        '필요한 정보를 모두 담고 있음',
        '잘 정리되어 있음',
        '유용한 예시나 코드가 포함됨'
      ]
    : [
        '부정확한 정보가 포함됨',
        '이해하기 어려움',
        '너무 장황함',
        '질문에 제대로 답변하지 않음',
        '필요한 정보가 누락됨'
      ];
  
  // 모달 외부 클릭 시 닫기
  useEffect(() => {
    if (!isOpen) return;
    
    const handleClickOutside = (e) => {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        onClose();
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);
  
  // 모달 열릴 때마다 상태 초기화
  useEffect(() => {
    if (isOpen) {
      setReasons([]);
      setComment('');
      setRating(0);
      setIsSubmitting(false);
    }
  }, [isOpen, feedbackType]);
  
  // 이유 토글 핸들러
  const toggleReason = (reason) => {
    setReasons(prev => 
      prev.includes(reason)
        ? prev.filter(r => r !== reason)
        : [...prev, reason]
    );
  };
  
  // 제출 핸들러
  const handleSubmit = async () => {
    if (isSubmitting) return;
    
    setIsSubmitting(true);
    try {
      // 피드백 데이터 준비
      const feedbackData = {
        feedbackType,
        reasons,
        comment: comment.trim(),
        rating,
        content: messageContent
      };
      
      await onSubmit(feedbackData);
      onClose();
    } catch (error) {
      console.error('피드백 제출 중 오류 발생:', error);
    } finally {
      setIsSubmitting(false);
    }
  };
  
  if (!isOpen) return null;
  
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in">
      <div 
        ref={modalRef}
        className="bg-gray-800 rounded-xl max-w-md w-full shadow-2xl border border-gray-700/50 overflow-hidden animate-slide-up"
      >
        <div className="p-4 border-b border-gray-700/50 flex items-center">
          <div className={`mr-2 p-2 rounded-full ${
            feedbackType === 'up' 
              ? 'bg-green-900/30 text-green-400' 
              : 'bg-red-900/30 text-red-400'
          }`}>
            {feedbackType === 'up' 
              ? <FiThumbsUp size={18} /> 
              : <FiThumbsDown size={18} />
            }
          </div>
          <h3 className="text-lg font-medium text-gray-100">
            {feedbackType === 'up' ? '긍정적인 피드백' : '부정적인 피드백'}
          </h3>
          <button
            onClick={onClose}
            className="ml-auto p-2 text-gray-400 hover:text-gray-200 rounded-full hover:bg-gray-700/50"
          >
            <FiX size={20} />
          </button>
        </div>
        
        <div className="p-5 max-h-[70vh] overflow-y-auto">
          {/* 피드백 이유 선택 */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              {feedbackType === 'up' ? '좋았던 점' : '개선이 필요한 점'}
            </label>
            <div className="space-y-2">
              {reasonOptions.map((reason, idx) => (
                <button
                  key={idx}
                  onClick={() => toggleReason(reason)}
                  className={`w-full text-left px-3 py-2 rounded-lg border text-sm transition-colors flex items-center ${
                    reasons.includes(reason)
                      ? feedbackType === 'up'
                        ? 'bg-green-900/20 border-green-500/30 text-green-300'
                        : 'bg-red-900/20 border-red-500/30 text-red-300'
                      : 'bg-gray-700/50 border-gray-600 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  <div className={`w-4 h-4 mr-2 rounded-full border flex-shrink-0 ${
                    reasons.includes(reason)
                      ? feedbackType === 'up'
                        ? 'bg-green-500 border-green-500'
                        : 'bg-red-500 border-red-500'
                      : 'border-gray-500'
                  }`}>
                    {reasons.includes(reason) && (
                      <FiCheck 
                        size={12} 
                        className="text-gray-900 m-auto" 
                      />
                    )}
                  </div>
                  <span>{reason}</span>
                </button>
              ))}
            </div>
          </div>
          
          {/* 별점 */}
          {feedbackType === 'up' && (
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                평가
              </label>
              <div className="flex items-center space-x-1">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    onClick={() => setRating(star)}
                    className={`p-1 transition-all ${
                      star <= rating
                        ? 'text-yellow-400 scale-110'
                        : 'text-gray-500 hover:text-gray-400'
                    }`}
                  >
                    <FiStar 
                      size={24} 
                      className={star <= rating ? 'fill-current' : ''}
                    />
                  </button>
                ))}
              </div>
            </div>
          )}
          
          {/* 추가 코멘트 */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              추가 코멘트 (선택사항)
            </label>
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="의견이 있으시면 자유롭게 작성해주세요..."
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg resize-none h-24 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-gray-200 placeholder-gray-500"
            />
          </div>
        </div>
        
        <div className="p-4 border-t border-gray-700/50 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-gray-700 text-gray-300 hover:bg-gray-600 transition-colors"
          >
            취소
          </button>
          <button
            onClick={handleSubmit}
            disabled={isSubmitting}
            className={`px-4 py-2 rounded-lg flex items-center ${
              feedbackType === 'up'
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-red-600 hover:bg-red-700 text-white'
            } transition-colors`}
          >
            {isSubmitting ? (
              <>
                <FiLoader className="animate-spin mr-2" size={16} />
                전송 중...
              </>
            ) : (
              <>
                <FiSend className="mr-2" size={16} />
                피드백 전송
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

// 피드백 결과 토스트 컴포넌트
const FeedbackToast = ({ isVisible, message, type, onClose }) => {
  useEffect(() => {
    if (isVisible) {
      const timer = setTimeout(() => {
        onClose();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [isVisible, onClose]);
  
  if (!isVisible) return null;
  
  return (
    <div className="fixed bottom-4 right-4 z-50 animate-fade-in">
      <div className={`p-3 rounded-lg shadow-lg flex items-center ${
        type === 'success'
          ? 'bg-green-600 text-white'
          : 'bg-red-600 text-white'
      }`}>
        {type === 'success' ? (
          <FiCheck className="mr-2" size={18} />
        ) : (
          <FiAlertCircle className="mr-2" size={18} />
        )}
        <span>{message}</span>
        <button
          onClick={onClose}
          className="ml-2 p-1 rounded-full hover:bg-black/10"
        >
          <FiX size={16} />
        </button>
      </div>
    </div>
  );
};

// 출처 미리보기 모달 컴포넌트
const SourcePreviewModal = ({ isOpen, onClose, source, content, image, isLoading, keywords }) => {
  if (!isOpen) return null;
  
  const [copySuccess, setCopySuccess] = useState(false);
  
  const handleCopyContent = () => {
    if (!content) return;
    
    navigator.clipboard.writeText(content);
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2000);
  };
  
  return (
    <div className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 rounded-xl max-w-3xl w-full max-h-[85vh] flex flex-col">
        {/* 헤더 */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <div className="flex items-center space-x-2">
            <FiFile className="text-indigo-400" />
            <h3 className="font-medium text-gray-200">
              {source?.title || source?.display_name || (source?.path && source.path.split('/').pop().replace(/^[^_]*_/, ''))}
            </h3>
            {source?.page && (
              <span className="text-xs text-gray-400 bg-gray-800 px-2 py-0.5 rounded">
                페이지 {source.page}
              </span>
            )}
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={handleCopyContent}
              className="p-2 rounded-full text-gray-400 hover:text-gray-200 hover:bg-gray-700"
              title="내용 복사"
              disabled={!content || isLoading}
            >
              {copySuccess ? <FiCheck /> : <FiCopy />}
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-full text-gray-400 hover:text-gray-200 hover:bg-gray-700"
            >
              <FiX />
            </button>
          </div>
        </div>
        
        {/* 본문 */}
        <div className="flex-1 overflow-auto p-4">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-400"></div>
            </div>
          ) : image ? (
            <div className="flex justify-center">
              <img 
                src={image} 
                alt="문서 이미지" 
                className="max-w-full max-h-[70vh] object-contain"
              />
            </div>
          ) : content ? (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown 
                remarkPlugins={[remarkGfm]} 
                rehypePlugins={[rehypeHighlight]}
              >
                {content}
              </ReactMarkdown>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-6">
              내용을 불러올 수 없습니다.
            </div>
          )}
        </div>
        
        {/* 키워드 표시 영역 */}
        {keywords && keywords.length > 0 && (
          <div className="px-4 py-3 border-t border-gray-700">
            <p className="text-xs text-gray-400 mb-2">관련 키워드:</p>
            <div className="flex flex-wrap gap-2">
              {keywords.map((keyword, idx) => (
                <span 
                  key={idx}
                  className="text-xs rounded-full px-2 py-1 bg-indigo-500/20 text-indigo-300"
                >
                  {keyword}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

function ChatMessage({ message, searchTerm = "", isSearchMode, prevMessage, nextMessage, onAskFollowUp }) {
  const isUser = message.role === "user";
  const [previewSource, setPreviewSource] = useState(null);
  const [previewContent, setPreviewContent] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null); // 'up', 'down', null
  const [star, setStar] = useState(0); // 별점(1~5)
  const [feedbackSent, setFeedbackSent] = useState(false); // 피드백 전송 여부
  const [loadingContent, setLoadingContent] = useState(false);
  const [highlightKeywords, setHighlightKeywords] = useState([]); // 모달 하이라이트용 키워드
  const [sourcesVisible, setSourcesVisible] = useState(true); // 출처 목록 기본적으로 표시
  const [isTypingComplete, setIsTypingComplete] = useState(true); // 기본값 true로 설정
  const [showTypeWriter, setShowTypeWriter] = useState(false); // 타이핑 효과 비활성화 (기본값 false)
  const [showSourcePreview, setShowSourcePreview] = useState(false); // 출처 미리보기 모달 표시 여부
  const contentRef = useRef(null);
  const messageContainerRef = useRef(null); // 메시지 컨테이너 ref 추가
  const [showImagePreview, setShowImagePreview] = useState(false);
  // 피드백 모달 상태 추가
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [currentFeedbackType, setCurrentFeedbackType] = useState(null);
  // 토스트 알림 상태 추가
  const [toast, setToast] = useState({ visible: false, message: '', type: 'success' });
  const [isGrouped, setIsGrouped] = useState(false);
  // 스레드 인디케이터 관련 변수들 추가
  const [threadStartLine, setThreadStartLine] = useState(false);
  const [showThreadLine, setShowThreadLine] = useState(false);
  const [threadEndLine, setThreadEndLine] = useState(false);
  const [showTableOfContents, setShowTableOfContents] = useState(false);
  
  // 피드백 버튼 상호작용 상태
  const [hasInteractedWithButtons, setHasInteractedWithButtons] = useState(false);
  
  // 출처 미리보기 관련 상태 추가
  const [sourceKeywords, setSourceKeywords] = useState([]);
  const [sourceFilterText, setSourceFilterText] = useState("");
  
  // 추천 질문 관련 상태 추가
  const suggestedQuestions = useMemo(() => {
    // 메시지 내용에서 추천 질문을 추출하거나 기본 배열 반환
    return message.suggestedQuestions || [];
  }, [message.suggestedQuestions]);
  
  // 메시지 ID 생성
  const messageId = useMemo(() => `message-${message.timestamp || Date.now()}-${Math.random().toString(36).substr(2, 9)}`, [message.timestamp]);
  
  // 키워드 하이라이트 스타일 주입
  useEffect(() => {
    injectStyleOnce('highlight-animation-style', keyframesStyle);
  }, []);
  
  // 같은 화자의 연속 메시지인지 확인 - 위치 이동
  const isPrevSameSender = useMemo(() => {
    return prevMessage && prevMessage.role === message.role;
  }, [prevMessage, message.role]);
  
  const isNextSameSender = useMemo(() => {
    return nextMessage && nextMessage.role === message.role;
  }, [nextMessage, message.role]);
  
  // 스레드 라인 관련 상태 업데이트
  useEffect(() => {
    // 현재 메시지가 어시스턴트인 경우만 스레드 라인 표시
    if (!isUser) {
      // 앞 메시지가 다른 화자(사용자)면 스레드 시작
      setThreadStartLine(!isPrevSameSender);
      
      // 뒤 메시지가 같은 화자(어시스턴트)면 스레드 라인 표시
      setShowThreadLine(isNextSameSender);
      
      // 뒤 메시지가 다른 화자(사용자)면 스레드 종료
      setThreadEndLine(!isNextSameSender);
    } else {
      // 사용자 메시지는 스레드 라인 표시 안함
      setThreadStartLine(false);
      setShowThreadLine(false);
      setThreadEndLine(false);
    }
  }, [isUser, isPrevSameSender, isNextSameSender]);

  // isPrevSameSender에 따라 isGrouped 상태 업데이트
  useEffect(() => {
    setIsGrouped(isPrevSameSender);
  }, [isPrevSameSender]);
  
  // 제목 추출 및 목차 생성
  const headings = useMemo(() => {
    if (isUser || !message.content) return [];
    return extractHeadings(message.content);
  }, [isUser, message.content]);
  
  // 목차 표시 여부 업데이트
  useEffect(() => {
    setShowTableOfContents(!isUser && headings.length >= 2);
  }, [isUser, headings.length]);
  
  // 목차 클릭 시 해당 제목으로 스크롤
  const scrollToHeading = (headingId) => {
    if (!contentRef.current) return;
    
    const element = contentRef.current.querySelector(`#${headingId}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      // 강조 효과 추가
      element.classList.add('bg-indigo-500/20');
      setTimeout(() => {
        element.classList.remove('bg-indigo-500/20');
      }, 2000);
    }
  };
  
  // 메시지가 변경될 때마다 스크롤 이벤트 발생
  useEffect(() => {
    // 메시지가 렌더링된 후 스크롤 이벤트 발생
    const timer = setTimeout(() => {
      window.dispatchEvent(new CustomEvent('chatScrollToBottom'));
    }, 100);
    
    return () => clearTimeout(timer);
  }, [message.content]);
  
  // 메시지 시간을 참조로 저장
  const messageTime = useMemo(() => {
    // 메시지가 타임스탬프를 가지고 있으면 그것을 사용, 아니면 현재 시간 생성
    return message.timestamp || new Date().getTime();
  }, [message.timestamp]);

  const formatMessageTime = (timestamp) => {
    if (!timestamp) return '방금 전';
    
    // timestamp가 숫자 또는 문자열인 경우 처리
    let dateObj;
    try {
      if (typeof timestamp === 'number' || !isNaN(parseInt(timestamp))) {
        dateObj = new Date(timestamp);
      } else if (typeof timestamp === 'string') {
        dateObj = new Date(timestamp);
      } else {
        return '방금 전';
      }
      
      // 올바른 날짜가 아닌 경우 (Invalid Date) 현재 시간으로 대체
      if (isNaN(dateObj.getTime())) {
        console.warn('Invalid date detected, using current time');
        dateObj = new Date();
      }
      
      // 현재 시간과의 차이 계산
      const now = new Date();
      const diffMs = now - dateObj;
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);
      
      // 미래 시간인 경우 처리
      if (diffMs < 0) {
        return '방금 전';
      }
      
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
    } catch (e) {
      console.error('Error formatting timestamp:', e);
      return '방금 전';
    }
  };

  const handleClosePreview = useCallback(() => {
    setPreviewSource(null);
    setPreviewContent("");
    setPreviewImage(null);
    setHighlightKeywords([]);
    setSourceKeywords([]);
    setSourceFilterText("");
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
      // 이미 피드백을 보냈다면 토스트 메시지 표시
      if (feedbackSent) {
        setToast({
          visible: true,
          message: '이미 피드백을 제출하셨습니다',
          type: 'info'
        });
        return;
      }
      
      // 피드백 버튼 상태 변경
      setFeedback((current) => (current === type ? null : type));
      
      // 피드백 모달 열기
      setCurrentFeedbackType(type);
      setShowFeedbackModal(true);
      
      // 버튼 상호작용 상태 업데이트
      setHasInteractedWithButtons(true);
    },
    [feedbackSent]
  );
  
  // 피드백 제출 처리
  const handleSubmitFeedback = async (feedbackData) => {
    try {
      const response = await fetch("http://172.10.2.70:8000/api/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messageId: Date.now().toString(),
          feedbackType: feedbackData.feedbackType,
          rating: feedbackData.rating,
          content: message.content,
          reasons: feedbackData.reasons,
          comment: feedbackData.comment
        }),
      });
      
      if (!response.ok) {
        throw new Error(`피드백 전송 실패: ${response.status} ${response.statusText}`);
      }
      
      // 피드백 전송 성공
      setFeedbackSent(true);
      setToast({
        visible: true,
        message: '피드백이 제출되었습니다. 감사합니다!',
        type: 'success'
      });
      return true;
    } catch (err) {
      console.error("피드백 전송 중 오류 발생:", err);
      setToast({
        visible: true,
        message: '피드백 전송에 실패했습니다. 다시 시도해주세요.',
        type: 'error'
      });
      throw err;
    }
  };
  
  // 토스트 닫기 핸들러
  const handleCloseToast = () => {
    setToast(prev => ({ ...prev, visible: false }));
  };
  
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

  // 소스 미리보기 핸들러
  const handlePreviewSource = async (source) => {
    if (loadingContent) return;

    setLoadingContent(true);
    setPreviewSource(source); 
    setSourceFilterText(""); // 필터 초기화
    setShowSourcePreview(true); // 미리보기 모달 표시

    try {
      const sourcePath = source.path;
      const chunkId = source.chunk_id || "";
      const page = source.page || 1;

      const response = await fetch("http://172.10.2.70:8000/api/source-preview", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          path: sourcePath,
          page: page,
          chunk_id: chunkId,
          answer_text: message.content, // 챗봇 응답 텍스트 전달
        }),
      });
      
      if (!response.ok) {
        throw new Error(`소스 미리보기 실패: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.content_type && data.content_type.startsWith('image/')) {
        // 이미지 소스인 경우
        setPreviewImage(data.image_url);
        setPreviewContent(null);
        setSourceKeywords([]);
        setHighlightKeywords([]);
      } else {
        // 텍스트 소스인 경우
        setPreviewContent(data.content || "내용을 불러올 수 없습니다.");
        setPreviewImage(null);
        
        // 백엔드에서 제공한 키워드 설정
        if (data.keywords && Array.isArray(data.keywords)) {
          setSourceKeywords(data.keywords);
          setHighlightKeywords(data.keywords);
        } else {
          setSourceKeywords([]);
          setHighlightKeywords([]);
        }
      }
    } catch (error) {
      console.error("소스 미리보기 오류:", error);
      setPreviewContent("소스를 불러오는 중 오류가 발생했습니다.");
      setPreviewImage(null);
      setSourceKeywords([]);
    } finally {
      setLoadingContent(false);
    }
  };

  // 출처 목록 토글 함수
  const toggleSourcesVisible = useCallback(() => {
    setSourcesVisible((prevVisible) => !prevVisible);
  }, []);

  // 소스 콘텐츠 필터링 함수
  const filterSourceContent = useCallback((content, filterText) => {
    if (!filterText || !content) return content;
    
    const lines = content.split('\n');
    const filteredLines = lines.filter(line => 
      line.toLowerCase().includes(filterText.toLowerCase())
    );
    
    // 검색 결과가 없으면 원본 반환
    if (filteredLines.length === 0) return content;
    
    // 필터링된 라인에 하이라이트 적용
    return filteredLines.join('\n\n');
  }, []);

  // 소스 콘텐츠에 키워드 하이라이트 적용 함수
  const applyKeywordHighlighting = useCallback((content, keywords) => {
    if (!content || !keywords || keywords.length === 0) return content;
    
    // 문자열로 확실하게 변환
    let processedContent = String(content);
    
    // 정규식 특수문자 이스케이프
    const escapedKeywords = keywords.map(kw => String(kw).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    
    // 정규식 패턴 생성
    const pattern = new RegExp(`(${escapedKeywords.join('|')})`, 'gi');
    
    // 마크다운 볼드체로 변환
    return processedContent.replace(pattern, '**$1**');
  }, []);

  // 마크다운으로 콘텐츠 렌더링 함수 - 출처 미리보기용
  const renderMarkdownContent = useCallback((content) => {
    if (!content) return null;
    
    // 스타일 주입 (한 번만)
    injectStyleOnce('highlight-style', keyframesStyle);
    
    // 필터 적용
    let displayContent = sourceFilterText ? filterSourceContent(content, sourceFilterText) : content;
    
    return (
      <div className="prose prose-sm dark:prose-invert max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeHighlight]}
          components={{
            // 코드 블록 커스텀 렌더링
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || "");
              return !inline && match ? (
                <div className="relative">
                  <div className="absolute top-2 right-2 flex space-x-2">
                    <div className="text-xs text-gray-500 mr-2">
                      {match[1]}
                    </div>
                    <button
                      onClick={() => {
                        const code = String(children).replace(/\n$/, "");
                        navigator.clipboard.writeText(code);
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
                  className={`${className || ''} rounded-md bg-gray-800/80 dark:bg-gray-900/80 px-1.5 py-0.5 text-gray-200 dark:text-gray-200`}
                  {...props}
                >
                  {children}
                </code>
              );
            },
            // 키워드 하이라이트를 위한 텍스트 처리
            p({ node, children, ...props }) {
              // 소스 키워드가 있으면 하이라이트 적용
              if (highlightKeywords.length > 0 && typeof children === 'string') {
                // 정규식으로 키워드 하이라이트
                const parts = [];
                let lastIndex = 0;
                
                // 정규식 특수문자 이스케이프
                const escapedKeywords = highlightKeywords.map(kw => String(kw).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
                const pattern = new RegExp(`(${escapedKeywords.join('|')})`, 'gi');
                
                // 키워드 매칭 및 하이라이트 처리
                let match;
                const text = String(children);
                
                // 정규식 매칭 결과 처리
                while ((match = pattern.exec(text)) !== null) {
                  // 매치 이전 텍스트 추가
                  if (match.index > lastIndex) {
                    parts.push(text.substring(lastIndex, match.index));
                  }
                  
                  // 매치된 키워드를 하이라이트 처리
                  parts.push(
                    <span 
                      key={`kw-${match.index}`} 
                      className="highlight-keyword animate-highlight-pulse"
                    >
                      {match[0]}
                    </span>
                  );
                  
                  lastIndex = pattern.lastIndex;
                }
                
                // 마지막 텍스트 추가
                if (lastIndex < text.length) {
                  parts.push(text.substring(lastIndex));
                }
                
                return <p {...props}>{parts.length > 0 ? parts : children}</p>;
              }
              
              return <p {...props}>{children}</p>;
            },
            // 볼드체 텍스트 처리 (키워드 하이라이트용)
            strong({ node, children, ...props }) {
              if (highlightKeywords.some(kw => {
                // 대소문자 구분 없이 비교
                const kwRegex = new RegExp(`^${String(kw).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}$`, 'i');
                return typeof children === 'string' && kwRegex.test(children);
              })) {
                return (
                  <span 
                    className="highlight-keyword animate-highlight-pulse"
                    {...props}
                  >
                    {children}
                  </span>
                );
              }
              return <strong {...props}>{children}</strong>;
            }
          }}
        >
          {displayContent}
        </ReactMarkdown>
      </div>
    );
  }, [sourceFilterText, filterSourceContent, highlightKeywords, copied, injectStyleOnce, keyframesStyle]);

  // 원시 HTML 태그를 이스케이프 처리하는 함수
  const escapeHtmlTags = (content) => {
    if (!content) return '';
    
    // 원본 콘텐츠 보존
    let processedContent = content;
    
    // 마크다운 테이블 형식이 있는지 먼저 확인
    const hasMarkdownTable = /\|[\s-]+\|/.test(processedContent);
    
    // 테이블 변환 - 마크다운 테이블 형식으로 변환
    if (!hasMarkdownTable && /<table>|<tr>|<td>|<th>/.test(processedContent)) {
      try {
        // 테이블 태그 균형 확인 로그
        const tableOpenCount = (processedContent.match(/<table>/g) || []).length;
        const tableCloseCount = (processedContent.match(/<\/table>/g) || []).length;
        
        console.log(`테이블 태그 균형: 열림(${tableOpenCount}) 닫힘(${tableCloseCount})`);
        
        // 테이블 추출 및 변환 (멀티라인 지원)
        const tablePattern = /<table>([\s\S]*?)<\/table>/g;
        let matches;
        let lastIndex = 0;
        let result = '';
        
        while ((matches = tablePattern.exec(processedContent)) !== null) {
          // 테이블 앞 부분 추가
          result += processedContent.substring(lastIndex, matches.index);
          
          // 테이블 콘텐츠 추출
          const tableContent = matches[1];
          lastIndex = matches.index + matches[0].length;
          
          // 테이블 헤더와 바디 추출
          let tableRows = [];
          let headerProcessed = false;
          
          // thead와 tbody 태그가 있는 경우 분리 처리
          const theadMatch = /<thead>([\s\S]*?)<\/thead>/.exec(tableContent);
          const tbodyMatch = /<tbody>([\s\S]*?)<\/tbody>/.exec(tableContent);
          
          if (theadMatch) {
            // thead에서 행 추출
            const headerRowMatch = /<tr>([\s\S]*?)<\/tr>/.exec(theadMatch[1]);
            if (headerRowMatch) {
              const headerCells = headerRowMatch[1].match(/<th>([\s\S]*?)<\/th>/g) || [];
              if (headerCells.length > 0) {
                const headerValues = headerCells.map(cell => 
                  cell.replace(/<th>([\s\S]*?)<\/th>/, '$1').trim()
                );
                tableRows.push(`| ${headerValues.join(' | ')} |`);
                tableRows.push(`| ${headerValues.map(() => '---').join(' | ')} |`);
                headerProcessed = true;
              }
            }
          }
          
          // tbody 처리 또는 직접 tr 추출
          const rowsContent = tbodyMatch ? tbodyMatch[1] : tableContent;
          const rowPattern = /<tr>([\s\S]*?)<\/tr>/g;
          let rowMatch;
          
          while ((rowMatch = rowPattern.exec(rowsContent)) !== null) {
            const rowContent = rowMatch[1];
            const thCells = rowContent.match(/<th>([\s\S]*?)<\/th>/g) || [];
            const tdCells = rowContent.match(/<td>([\s\S]*?)<\/td>/g) || [];
            
            // 첫 번째 행이 th 셀을 가지고 있고, 헤더가 아직 처리되지 않은 경우
            if (thCells.length > 0 && !headerProcessed) {
              const headerValues = thCells.map(cell => 
                cell.replace(/<th>([\s\S]*?)<\/th>/, '$1').trim()
              );
              tableRows.push(`| ${headerValues.join(' | ')} |`);
              tableRows.push(`| ${headerValues.map(() => '---').join(' | ')} |`);
              headerProcessed = true;
            } 
            // td 셀 처리
            else if (tdCells.length > 0) {
              const cellValues = tdCells.map(cell => 
                cell.replace(/<td>([\s\S]*?)<\/td>/, '$1').trim()
              );
              
              // 첫 번째 행이고 아직 헤더가 없는 경우, 헤더로 처리
              if (tableRows.length === 0) {
                tableRows.push(`| ${cellValues.join(' | ')} |`);
                tableRows.push(`| ${cellValues.map(() => '---').join(' | ')} |`);
                headerProcessed = true;
              } else {
                tableRows.push(`| ${cellValues.join(' | ')} |`);
              }
            }
          }
          
          // 마크다운 테이블 생성
          if (tableRows.length > 0) {
            result += '\n' + tableRows.join('\n') + '\n';
          } else {
            // 변환에 실패한 경우 원본 텍스트 유지
            result += matches[0];
          }
        }
        
        // 테이블 다음 부분 추가
        result += processedContent.substring(lastIndex);
        processedContent = result;
      } catch (error) {
        console.error("테이블 변환 중 오류 발생:", error);
      }
    }
    
    // 닫히지 않은 br 태그 처리
    processedContent = processedContent.replace(/<br\s*\/?>/g, '\n');
    
    // p 태그 변환
    processedContent = processedContent.replace(/<p.*?>([\s\S]*?)<\/p>/g, '\n$1\n');
    
    // 기타 일반적인 태그 처리
    processedContent = processedContent
      .replace(/<b>([\s\S]*?)<\/b>/g, '**$1**')
      .replace(/<i>([\s\S]*?)<\/i>/g, '*$1*')
      .replace(/<strong>([\s\S]*?)<\/strong>/g, '**$1**')
      .replace(/<em>([\s\S]*?)<\/em>/g, '*$1*')
      .replace(/<ul>([\s\S]*?)<\/ul>/g, (match, content) => {
        return '\n' + content.replace(/<li>([\s\S]*?)<\/li>/g, '- $1\n');
      })
      .replace(/<ol>([\s\S]*?)<\/ol>/g, (match, content) => {
        return '\n' + content.replace(/<li>([\s\S]*?)<\/li>/g, '1. $1\n');
      });
    
    // 색상 태그 처리 (font color)
    processedContent = processedContent.replace(/<font color="([\s\S]*?)">([\s\S]*?)<\/font>/g, '**$2**');
    
    // 남은 모든 HTML 태그 제거
    const remainingTags = processedContent.match(/<[^>]+>/g);
    if (remainingTags) {
      console.log('남은 HTML 태그:', remainingTags);
      processedContent = processedContent.replace(/<[^>]+>/g, '');
    }
    
    return processedContent;
  };

  // 마크다운 컴포넌트 생성 - 채팅 메시지용
  const MarkdownContent = useMemo(() => {
    // 헤딩에 ID 추가하는 함수
    const addHeadingIds = (content) => {
      return content.replace(/^(#{1,3})\s+(.+)$/gm, (match, hashes, text) => {
        const id = text.toLowerCase().replace(/[^\w\s가-힣]/g, '').replace(/\s+/g, '-');
        return `${hashes} <span id="${id}">${text}</span>`;
      });
    };
    
    // 콘텐츠 전처리
    const processedContent = escapeHtmlTags(message.content);
    
    // 헤딩이 있는 경우 ID 추가
    const contentWithIds = headings.length > 0 ? addHeadingIds(processedContent) : processedContent;
    
    return (
      <div className="prose prose-sm dark:prose-invert max-w-none">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeHighlight, rehypeKatex]}
          components={{
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || "");
              return !inline && match ? (
                <div className="relative">
                  <div className="absolute top-2 right-2 flex space-x-2">
                    <div className="text-xs text-gray-500 mr-2">
                      {match[1]}
                    </div>
                    <button
                      onClick={() => {
                        const code = String(children).replace(/\n$/, "");
                        navigator.clipboard.writeText(code);
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
                  className={`${className || ''} rounded-md bg-gray-800/80 dark:bg-gray-900/80 px-1.5 py-0.5 text-gray-200 dark:text-gray-200`}
                  {...props}
                >
                  {children}
                </code>
              );
            }
          }}
        >
          {contentWithIds}
        </ReactMarkdown>
      </div>
    );
  }, [message.content, headings, copied]);

  // 메시지 그룹핑 로직
  const isLastInGroup = !nextMessage || nextMessage.role !== message.role;
  
  // 이미지 URL 추출 및 처리
  const extractImageUrl = (content) => {
    const imageRegex = /!\[.*?\]\((.*?)\)/;
    const match = content.match(imageRegex);
    return match ? match[1] : null;
  };
  
  const imageUrl = extractImageUrl(message.content);
  
  // 메시지 내용에서 이미지 마크다운 제거
  const cleanContent = message.content.replace(/!\[.*?\]\(.*?\)/g, '').trim();
  
  // 검색어 하이라이트 처리
  const highlightSearchTerm = (text) => {
    if (!searchTerm || !text) return text;
    const regex = new RegExp(`(${searchTerm})`, 'gi');
    return text.split(regex).map((part, i) => 
      regex.test(part) ? <mark key={i} className="bg-yellow-500/30">{part}</mark> : part
    );
  };
  
  // 출처 정보 디버깅 로그 추가
  useEffect(() => {
    if (message.role === 'assistant') {
      console.log('출처 정보 상태:', {
        hasSources: Boolean(message.sources),
        sourcesLength: message.sources?.length || 0,
        hasCitedSources: Boolean(message.cited_sources),
        citedSourcesLength: message.cited_sources?.length || 0,
        filteredSourcesLength: message.sources?.filter(s => s.is_cited)?.length || 0,
        sourcesVisible,
        isLastInGroup
      });
    }
  }, [message, sourcesVisible, isLastInGroup]);
  
  return (
    <div ref={messageContainerRef} id={messageId}>
      <div 
        className={`py-4 px-4 ${
          isGrouped ? 'pt-1' : ''
        } relative group transition-colors hover:bg-gray-800/40`}
      >
        {/* 메시지 시간 및 컨트롤 영역 */}
        <div className={`flex items-center justify-${isUser ? 'end' : 'start'} text-xs text-gray-500 mb-1.5 ${isGrouped ? 'opacity-0 group-hover:opacity-100' : ''}`}>
          <time dateTime={(() => {
            try {
              // timestamp가 유효한지 확인
              const timestamp = message.timestamp ? new Date(message.timestamp) : new Date();
              // Invalid Date 확인
              return isNaN(timestamp.getTime()) ? new Date().toISOString() : timestamp.toISOString();
            } catch (e) {
              console.error('Invalid timestamp:', message.timestamp);
              return new Date().toISOString();
            }
          })()}>{formatMessageTime(message.timestamp)}</time>
        </div>
        
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
          {/* 사용자 메시지는 오른쪽 정렬, AI 응답은 왼쪽 정렬 */}
          <div className={`flex ${isUser ? 'flex-row-reverse' : 'flex-row'} items-start`}>
            {/* 프로필 아바타 */}
            <ProfileAvatar role={message.role} isGrouped={isGrouped} />
            
            {/* 메시지 컨텐츠 */}
            <div 
              className={`relative mx-3 ${isUser ? 'mr-3 ml-12' : 'ml-3 mr-12'} flex-1`}
              style={{ maxWidth: 'calc(100% - 80px)' }}
            >
              {/* 메시지 말풍선 */}
              <div 
                className={`rounded-2xl py-3 px-4 ${
                  isUser 
                    ? 'bg-blue-600 text-white rounded-tr-md' 
                    : 'bg-gray-800 text-gray-100 rounded-tl-md'
                }`}
              >
                {/* 메시지 내용 */}
                <div className="prose dark:prose-invert max-w-none marker:text-indigo-400" ref={contentRef}>
                  {!isUser && showTypeWriter ? (
                    <TypeWriter 
                      text={message.content} 
                      speed={1} 
                      onComplete={() => {
                        setIsTypingComplete(true);
                        setShowTypeWriter(false);
                      }}
                    />
                  ) : (
                    MarkdownContent
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* 출처 및 피드백 영역 */}
        {message.role === 'assistant' && (
          <div className="ml-11 mt-2">
            {/* 출처 카드 (토글형) */}
            {message.sources && message.sources.length > 0 && (
              <div className="mt-1">
                <button
                  onClick={toggleSourcesVisible}
                  className="flex items-center text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
                >
                  {sourcesVisible ? (
                    <FiChevronDown size={14} className="mr-1" />
                  ) : (
                    <FiChevronRight size={14} className="mr-1" />
                  )}
                  {/* 인용된 출처만 표시 */}
                  <span>
                    출처 {(message.cited_sources?.length || message.sources.filter(s => s.is_cited)?.length || 0)}개
                    {sourcesVisible ? " (접기)" : " (펼치기)"}
                  </span>
                </button>
                
                {sourcesVisible && (
                  <div className="mt-2 mb-3 space-y-2 text-xs border border-gray-700/50 rounded-lg overflow-hidden">
                    <div className="bg-gray-800/60 px-3 py-2 text-gray-300 font-medium flex items-center border-b border-gray-700/50">
                      <FiLink size={12} className="mr-2" />
                      <span>답변과 관련된 출처</span>
                    </div>
                    <div className="px-2 py-1 space-y-2 max-h-48 overflow-y-auto">
                      {(() => {
                        // 표시할 출처 목록 결정
                        const sourcesToShow = message.cited_sources?.length > 0 
                          ? message.cited_sources 
                          : (message.sources?.filter(s => s.is_cited) || []);

                        console.log('표시할 출처 목록:', sourcesToShow.length, '개');
                          
                        return sourcesToShow.length > 0 ? (
                          // 출처가 있는 경우
                          sourcesToShow.map((source, idx) => (
                            <div
                              key={idx}
                              className="flex items-start hover:bg-gray-700/30 p-2 rounded-md cursor-pointer transition-colors"
                              onClick={() => handlePreviewSource(source)}
                            >
                              <FiFile size={14} className="mt-0.5 mr-2 text-indigo-300 flex-shrink-0" />
                              <div className="overflow-hidden flex-1">
                                <div className="font-medium text-gray-200 truncate">
                                  {source.title || source.display_name || source.path?.split('/').pop().replace(/^[^_]*_/, '') || "출처 문서"}
                                </div>
                                <div className="flex items-center mt-1 text-gray-400 text-[10px]">
                                  {source.page && (
                                    <span className="flex items-center">
                                      <FiHash size={10} className="mr-1" /> 
                                      페이지 {source.page}
                                    </span>
                                  )}
                                  {source.score && (
                                    <span className="ml-3 flex items-center">
                                      <FiStar size={10} className="mr-1" /> 
                                      관련도: {Math.round(source.score * 100) / 100}
                                    </span>
                                  )}
                                  <span className="ml-auto text-indigo-300 flex items-center">
                                    <FiSearch size={10} className="mr-1" /> 
                                    내용 보기
                                  </span>
                                </div>
                              </div>
                            </div>
                          ))
                        ) : (
                          // 출처가 없는 경우
                          <div className="text-center py-4 text-gray-400">
                            <FiInfo size={18} className="mx-auto mb-2" />
                            <p>답변과 일치하는 출처 정보가 없습니다.</p>
                          </div>
                        );
                      })()}
                    </div>
                  </div>
                )}
              </div>
            )}
            
            {/* 후속 질문 영역 */}
            {suggestedQuestions.length > 0 && (
              <div className="mt-4 space-y-2">
                <p className="text-xs text-gray-400 ml-1">후속 질문 추천:</p>
                <div className="flex flex-wrap gap-2">
                  {suggestedQuestions.map((question, idx) => (
                    <button
                      key={idx}
                      onClick={() => onAskFollowUp(question)}
                      className="px-2 py-1 text-xs rounded-full transition-colors bg-gray-700 text-gray-300 hover:bg-gray-600"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              </div>
            )}
            
            {/* 피드백 버튼 영역 */}
            <div className={`mt-2 flex items-center text-xs gap-3 transition-opacity ${
              hasInteractedWithButtons ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
            }`}>
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
        )}
        
        {/* 콘텐츠 목차 */}
        {showTableOfContents && (
          <div className="ml-11 mt-3 max-w-2xl">
            <TableOfContents 
              headings={headings} 
              onClickHeading={scrollToHeading}
            />
          </div>
        )}
      </div>
      
      {/* 피드백 모달 */}
      <FeedbackModal
        isOpen={showFeedbackModal}
        onClose={() => setShowFeedbackModal(false)}
        messageContent={message.content}
        onSubmit={handleSubmitFeedback}
        feedbackType={currentFeedbackType}
      />
      
      {/* 출처 미리보기 모달 */}
      <SourcePreviewModal
        isOpen={showSourcePreview}
        onClose={() => setShowSourcePreview(false)}
        source={previewSource}
        content={previewContent}
        image={previewImage}
        isLoading={loadingContent}
        keywords={highlightKeywords}
      />
      
      {/* 피드백 토스트 */}
      <FeedbackToast 
        isVisible={toast.visible}
        message={toast.message}
        type={toast.type}
        onClose={handleCloseToast}
      />
    </div>
  );
}

export default memo(ChatMessage);
