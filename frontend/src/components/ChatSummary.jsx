import { useState, useEffect } from 'react';
import { FiFileText, FiChevronDown, FiChevronUp, FiLoader } from 'react-icons/fi';

// 대화 요약 컴포넌트
const ChatSummary = ({ messages = [], conversationId, isVisible = true }) => {
  const [summary, setSummary] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isExpanded, setIsExpanded] = useState(false);
  
  // 대화 내용이 변경될 때마다 요약 생성
  useEffect(() => {
    // 메시지가 3개 이상일 때만 요약 생성
    if (messages.length >= 3 && conversationId && isVisible) {
      generateSummary();
    } else {
      setSummary('');
    }
  }, [conversationId, isVisible]);
  
  // 대화 내용 요약 생성
  const generateSummary = async () => {
    // 이미 요약이 있거나 로딩 중이면 중복 요청 방지
    if (summary || isLoading) return;
    
    try {
      setIsLoading(true);
      setError(null);
      
      // 요약 API 호출
      const response = await fetch('/api/generate-title', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: messages.slice(-6) // 최근 6개 메시지만 사용
        }),
      });
      
      if (!response.ok) {
        throw new Error('요약 생성에 실패했습니다.');
      }
      
      const data = await response.json();
      
      if (data && data.title) {
        setSummary(data.title);
      } else {
        setSummary('현재 대화 내용');
      }
    } catch (err) {
      console.error('대화 요약 생성 중 오류:', err);
      setError('요약을 생성할 수 없습니다.');
      setSummary('현재 대화 내용');
    } finally {
      setIsLoading(false);
    }
  };
  
  // 표시할 내용이 없으면 렌더링하지 않음
  if (!isVisible || messages.length < 3) {
    return null;
  }
  
  return (
    <div className="mx-auto max-w-4xl px-4 mb-4 mt-2">
      <div className="bg-gray-800/60 backdrop-blur-sm border border-gray-700/50 rounded-lg overflow-hidden">
        <div 
          className="px-4 py-3 flex items-center justify-between cursor-pointer hover:bg-gray-750/50 transition-colors"
          onClick={() => setIsExpanded(!isExpanded)}
        >
          <div className="flex items-center">
            <div className="w-8 h-8 rounded-full bg-indigo-600/20 flex items-center justify-center mr-3">
              <FiFileText className="text-indigo-400" size={16} />
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-200">
                {isLoading ? '대화 요약 생성 중...' : summary || '현재 대화 내용'}
              </h3>
              <p className="text-xs text-gray-400 mt-0.5">
                {messages.length}개의 메시지 | {new Date().toLocaleDateString()}
              </p>
            </div>
          </div>
          
          <div className="flex items-center">
            {isLoading && (
              <FiLoader className="text-indigo-400 animate-spin mr-2" size={16} />
            )}
            {isExpanded ? (
              <FiChevronUp className="text-gray-400" size={18} />
            ) : (
              <FiChevronDown className="text-gray-400" size={18} />
            )}
          </div>
        </div>
        
        {isExpanded && (
          <div className="px-4 py-3 border-t border-gray-700/30 bg-gray-800/30">
            <div className="text-sm text-gray-300 leading-relaxed">
              {isLoading ? (
                <div className="flex items-center space-x-2 text-gray-400">
                  <FiLoader className="animate-spin" size={14} />
                  <span>대화 내용을 요약하고 있습니다...</span>
                </div>
              ) : error ? (
                <p className="text-red-400">{error}</p>
              ) : (
                <p>{summary || '이 대화에 대한 요약을 생성할 수 없습니다.'}</p>
              )}
            </div>
            
            <div className="mt-3 pt-3 border-t border-gray-700/30">
              <h4 className="text-xs font-medium text-gray-400 mb-2">대화 주제</h4>
              <div className="flex flex-wrap gap-2">
                {extractKeywords(messages).map((keyword, index) => (
                  <span 
                    key={index}
                    className="px-2 py-1 text-xs bg-indigo-900/30 text-indigo-300 rounded-full"
                  >
                    {keyword}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// 대화에서 키워드 추출 함수
const extractKeywords = (messages) => {
  // 실제 구현에서는 서버에서 키워드를 추출하거나 더 복잡한 로직을 사용할 수 있음
  // 여기서는 간단한 예시로 구현
  const userMessages = messages
    .filter(msg => msg.role === 'user')
    .map(msg => msg.content)
    .join(' ');
    
  // 불용어 목록
  const stopwords = ['이', '그', '저', '것', '이것', '저것', '어떤', '무슨', '어느', '아', '휴', '아이구', '아이쿠', 
                    '어', '나', '우리', '저희', '당신', '너', '너희', '그들', '그녀', '그것', '저것', '이것', '저기', 
                    '이런', '저런', '그런', '어떤', '무슨', '어느', '몇', '하다', '있다', '되다', '가다', '알다', '오다', 
                    '있다', '같다', '이다', '보다', '그렇다', '그러나', '그리고', '하지만', '또한', '그래서', '따라서', 
                    '하지만', '그런데', '그러면', '그러므로', '그러니까', '왜냐하면', '때문에', '그래도', '혹은', '또는'];
  
  // 간단한 키워드 추출 (2글자 이상, 불용어 제외)
  const words = userMessages.split(/\s+/).filter(word => 
    word.length >= 2 && !stopwords.includes(word)
  );
  
  // 단어 빈도 계산
  const wordFreq = {};
  words.forEach(word => {
    wordFreq[word] = (wordFreq[word] || 0) + 1;
  });
  
  // 빈도 기준 정렬 후 상위 5개 반환
  return Object.entries(wordFreq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([word]) => word);
};

export default ChatSummary; 