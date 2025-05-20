import { useState, useEffect } from 'react';

const TypeWriter = ({ text, speed = 2, onComplete }) => {
  const [displayText, setDisplayText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  
  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
        
        // 주기적으로 스크롤 이벤트 발생하는 부분 제거
        // 타이핑 중에는 스크롤 이벤트를 발생시키지 않음
      }, speed);
      
      return () => clearTimeout(timeout);
    } else if (!isComplete) {
      setIsComplete(true);
      
      // 타이핑 완료 시에만 스크롤 이벤트 발생
      // 이 부분은 유지하여 타이핑이 완료되면 스크롤이 내려가도록 함
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

export default TypeWriter; 