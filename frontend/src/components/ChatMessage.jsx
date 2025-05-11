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
  FiHelpCircle
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
  const [highlightKeywords, setHighlightKeywords] = useState(""); // 모달 하이라이트용 키워드
  const [sourcesVisible, setSourcesVisible] = useState(false);
  const [isTypingComplete, setIsTypingComplete] = useState(true); // 기본값 true로 설정
  const [showTypeWriter, setShowTypeWriter] = useState(false); // 타이핑 효과 비활성화 (기본값 false)
  const contentRef = useRef(null);
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
      } else {
        // 텍스트 소스인 경우
        setPreviewContent(data.content || "내용을 불러올 수 없습니다.");
        setPreviewImage(null);
        
        // 검색어가 있는 경우, 하이라이트 키워드 설정
        if (data.keywords && Array.isArray(data.keywords)) {
          setHighlightKeywords(data.keywords);
        }
      }
    } catch (error) {
      console.error("소스 미리보기 오류:", error);
      setPreviewContent("소스를 불러오는 중 오류가 발생했습니다.");
      setPreviewImage(null);
    } finally {
      setLoadingContent(false);
    }
  };

  const toggleSourcesVisible = useCallback(() => {
    setSourcesVisible((prevVisible) => !prevVisible);
  }, []);

  // 마크다운 컴포넌트 생성
  const MarkdownContent = useMemo(() => {
    // 헤딩에 ID 추가하는 함수
    const addHeadingIds = (content) => {
      return content.replace(/^(#{1,3})\s+(.+)$/gm, (match, hashes, text) => {
        const id = text.toLowerCase().replace(/[^\w\s가-힣]/g, '').replace(/\s+/g, '-');
        return `${hashes} <span id="${id}">${text}</span>`;
      });
    };
    
    // 헤딩이 있는 경우 ID 추가
    const contentWithIds = headings.length > 0 ? addHeadingIds(message.content) : message.content;
    
    return (
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeHighlight, rehypeKatex, rehypeRaw]}
                components={{
          h1: ({ node, ...props }) => (
            <h1 className="text-xl font-bold mt-6 mb-3 pb-1 border-b border-gray-700/30 text-gray-100" {...props} />
          ),
          h2: ({ node, ...props }) => (
            <h2 className="text-lg font-semibold mt-5 mb-2 text-gray-100" {...props} />
          ),
          h3: ({ node, ...props }) => (
            <h3 className="text-base font-medium mt-4 mb-1 text-gray-200" {...props} />
          ),
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
                        className={`${className} rounded-md bg-gray-800/80 dark:bg-gray-900/80 px-1.5 py-0.5 text-gray-200 dark:text-gray-200`}
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
        {contentWithIds}
              </ReactMarkdown>
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
  
  return (
    <>
      {/* 소스 프리뷰 모달 */}
      {previewSource && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center z-50 p-4 animate-fade-in">
          <div className="bg-gradient-to-br from-gray-800 to-gray-850 rounded-xl w-full max-w-3xl max-h-[90vh] flex flex-col overflow-hidden shadow-soft-2xl border border-gray-700/50">
            <div className="flex items-center justify-between p-4 border-b border-gray-700/50 bg-gradient-to-r from-gray-800/90 to-gray-850/90">
              <h3 className="font-medium flex items-center text-gray-200">
                <FiEye className="mr-2 text-indigo-500" />
                <span className="truncate max-w-md">{previewSource.title || previewSource.path}</span>
                {previewSource.page && <span className="ml-2 text-gray-400 text-sm">(페이지 {previewSource.page})</span>}
              </h3>
              <button 
                onClick={handleClosePreview}
                className="p-2 rounded-full hover:bg-gray-700 transition-colors text-gray-400"
              >
                <FiX size={20} />
              </button>
            </div>
            
            <div className="flex-1 overflow-auto p-6 custom-scrollbar bg-gradient-to-b from-gray-800/90 to-gray-850/95">
              {loadingContent ? (
                <div className="flex flex-col justify-center items-center h-32 space-y-3">
                  <FiLoader className="animate-spin text-indigo-500" size={24} />
                  <span className="text-gray-400 text-sm">내용을 불러오는 중...</span>
          </div>
              ) : previewContent ? (
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {previewContent}
                </div>
              ) : (
                <div className="text-center text-gray-400 py-8">
                  미리보기를 불러올 수 없습니다.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 메시지 컨테이너 */}
      <div 
        className={`flex w-full mb-4 relative animate-fade-in group ${
          isUser ? "justify-end" : "justify-start"
        }`}
      >
        {/* 스레드 인디케이터 - 어시스턴트 메시지만 표시 */}
        {!isUser && (
          <div className="absolute left-4 top-0 bottom-0 w-0.5">
            {/* 상단 스레드 라인 */}
            {threadStartLine && (
              <div className="absolute top-0 left-0 w-0.5 h-4 bg-gradient-to-b from-transparent to-indigo-500/30"></div>
            )}
            
            {/* 중간 스레드 라인 */}
            {showThreadLine && (
              <div className="absolute top-0 left-0 w-0.5 h-full bg-indigo-500/30"></div>
            )}
            
            {/* 하단 스레드 라인 */}
            {threadEndLine && (
              <div className="absolute bottom-0 left-0 w-0.5 h-4 bg-gradient-to-t from-transparent to-indigo-500/30"></div>
            )}
          </div>
        )}
        
        {/* 프로필 아이콘 (어시스턴트만) */}
        {!isUser && !isPrevSameSender && (
          <div className="flex-shrink-0 mr-3 mt-1">
            <ProfileAvatar role={message.role} isGrouped={isGrouped} />
          </div>
        )}
        
        {/* 어시스턴트 메시지 들여쓰기 - 스레드 효과 */}
        {!isUser && isPrevSameSender && <div className="w-12"></div>}
        
        <div
          className={`flex ${
            isUser ? "flex-row-reverse" : "flex-row"
          } max-w-[85%] sm:max-w-[75%]`}
        >
          <div
            className={`flex flex-col ${
              isUser
                ? "items-end"
                : "items-start"
            }`}
          >
            {/* 메시지 시간 표시 */}
            {!isPrevSameSender && (
              <div className="text-xs text-gray-400 mb-1 px-1">
                {formatMessageTime(messageTime)}
              </div>
            )}
            
            {/* 목차 섹션 - 어시스턴트 메시지이고 제목이 있는 경우에만 표시 */}
            {showTableOfContents && (
              <TableOfContents headings={headings} onClickHeading={scrollToHeading} />
            )}
            
            {/* 메시지 내용 */}
            <div
              className={`message-bubble relative px-4 py-3 rounded-2xl ${
                isUser
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-800 text-gray-100"
              } ${isPrevSameSender ? "mt-1" : "mt-0"}`}
            >
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
          {isUser && !isPrevSameSender && (
          <div className="flex-shrink-0 ml-3 mt-1">
              <ProfileAvatar role={message.role} isGrouped={isGrouped} />
            </div>
          )}
          
          {/* 사용자 메시지 오른쪽 여백 처리 */}
          {isUser && isPrevSameSender && <div className="w-12"></div>}
          </div>
      </div>

      {/* 이미지 미리보기 모달 */}
      {showImagePreview && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-md flex items-center justify-center z-50 p-4 animate-fade-in">
          <div className="bg-gradient-to-br from-gray-800 to-gray-850 rounded-xl w-full max-w-3xl max-h-[90vh] flex flex-col overflow-hidden shadow-soft-2xl border border-gray-700/50">
            <div className="flex items-center justify-between p-4 border-b border-gray-700/50 bg-gradient-to-r from-gray-800/90 to-gray-850/90">
              <h3 className="font-medium flex items-center text-gray-200">
                <FiImage className="mr-2 text-indigo-500" />
                <span className="truncate max-w-md">{previewImage ? previewImage : "이미지 미리보기"}</span>
              </h3>
              <button 
                onClick={() => setShowImagePreview(false)}
                className="p-2 rounded-full hover:bg-gray-700 transition-colors text-gray-400"
              >
                <FiX size={20} />
              </button>
            </div>
            
            <div className="flex-1 overflow-auto p-6 custom-scrollbar bg-gradient-to-b from-gray-800/90 to-gray-850/95">
              {previewImage ? (
                <div className="relative group">
                  <img 
                    src={previewImage}
                    alt="미리보기"
                    className={`rounded-lg shadow-md transition-all duration-300 ${
                      showImagePreview ? 'max-w-none' : 'max-w-md'
                    }`}
                  />
                  <button
                    onClick={() => setShowImagePreview(false)}
                    className="absolute top-2 right-2 p-2 bg-gray-800/80 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    {showImagePreview ? (
                      <FiMinimize2 className="text-white" size={16} />
                    ) : (
                      <FiMaximize2 className="text-white" size={16} />
                    )}
                  </button>
      </div>
              ) : (
                <div className="text-center text-gray-400 py-8">
                  이미지를 불러올 수 없습니다.
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 피드백 모달 */}
      <FeedbackModal
        isOpen={showFeedbackModal}
        onClose={() => setShowFeedbackModal(false)}
        messageContent={message.content}
        feedbackType={currentFeedbackType}
        onSubmit={handleSubmitFeedback}
      />
      
      {/* 피드백 결과 토스트 */}
      <FeedbackToast
        isVisible={toast.visible}
        message={toast.message}
        type={toast.type}
        onClose={handleCloseToast}
      />
    </>
  );
}

export default memo(ChatMessage);
