import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import rehypeRaw from 'rehype-raw';
import { 
  FiX, 
  FiCopy, 
  FiCheck, 
  FiMaximize2, 
  FiMinimize2, 
  FiFile, 
  FiFileText, 
  FiAlertTriangle, 
  FiBookmark, 
  FiExternalLink 
} from 'react-icons/fi';

/**
 * 출처 문서 미리보기 모달 컴포넌트
 */
export const SourcePreviewModal = ({ isOpen, onClose, source, content, image, isLoading, keywords }) => {
  if (!isOpen) return null;
  
  const [copySuccess, setCopySuccess] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  
  // 오류 메시지인지 확인하는 함수
  const isErrorMessage = (text) => {
    if (!text) return false;
    return text.startsWith("소스를 불러오는 중 오류") || 
           text.startsWith("내용을 불러올 수 없습니다") ||
           text.includes("오류가 발생했습니다");
  };
  
  // 오류 메시지에서 복수 줄 처리
  const formatErrorMessage = (text) => {
    if (!text || !isErrorMessage(text)) return text;
    
    // 줄바꿈 기호가 포함된 경우 분리하여 렌더링
    const lines = text.split('\n');
    if (lines.length <= 1) return text;
    
    return (
      <>
        {lines.map((line, i) => (
          <React.Fragment key={i}>
            {line}
            {i < lines.length - 1 && <br />}
          </React.Fragment>
        ))}
      </>
    );
  };
  
  const handleCopyContent = () => {
    if (!content) return;
    
    navigator.clipboard.writeText(content);
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2000);
  };
  
  // HTML 태그가 포함된 콘텐츠인지 확인
  const hasHtmlTags = (text) => {
    return text && typeof text === 'string' && (
      text.includes('<span class="highlight-strong">') || 
      text.includes('<span class="highlight-medium">')
    );
  };
  
  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };
  
  // 키워드 하이라이트 함수
  const highlightKeywordsInContent = (text, keywords) => {
    if (!text || !keywords || keywords.length === 0) return text;
    
    // 정규식 특수문자 이스케이프
    const escapedKeywords = keywords.map(kw => String(kw).replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    
    // 문자열 내 모든 키워드를 강조 표시
    let highlightedText = text;
    escapedKeywords.forEach(keyword => {
      const regex = new RegExp(`(${keyword})`, 'gi');
      highlightedText = highlightedText.replace(regex, '**$1**');
    });
    
    return highlightedText;
  };
  
  // 렌더링 콘텐츠 결정
  const renderContent = () => {
    if (isLoading) {
      return (
        <div className="flex flex-col items-center justify-center h-full py-12">
          <div className="w-12 h-12 border-t-2 border-b-2 border-indigo-500 rounded-full animate-spin mb-4"></div>
          <p className="text-gray-400 text-sm font-medium">문서 내용을 불러오는 중...</p>
          <p className="text-gray-500 text-xs mt-2">잠시만 기다려주세요</p>
        </div>
      );
    }
    
    if (image) {
      return (
        <div className="flex justify-center items-center h-full">
          <img 
            src={image} 
            alt="문서 이미지" 
            className="max-w-full max-h-[80vh] object-contain rounded-lg shadow-lg"
          />
        </div>
      );
    }
    
    if (!content) {
      return (
        <div className="flex flex-col items-center justify-center h-full py-12">
          <FiFileText className="text-gray-400 mb-4" size={48} />
          <p className="text-gray-500 font-medium">내용이 없습니다</p>
          <p className="text-gray-400 text-sm mt-2">해당 문서에 텍스트 내용이 없거나 추출할 수 없습니다.</p>
        </div>
      );
    }
    
    if (isErrorMessage(content)) {
      return (
        <div className="flex flex-col items-center justify-center min-h-[200px] text-center py-12">
          <FiAlertTriangle className="text-yellow-500 mb-4" size={48} />
          <p className="text-gray-300 mb-3 font-medium text-lg">문서 내용을 불러올 수 없습니다</p>
          <div className="text-gray-500 text-sm max-w-md px-8">
            {formatErrorMessage(content.includes(":") ? content.split(":")[1].trim() : content)}
          </div>
          <button
            onClick={onClose}
            className="mt-6 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm transition-colors"
          >
            닫기
          </button>
        </div>
      );
    }
    
    // HTML 태그가 포함된 콘텐츠 처리
    if (hasHtmlTags(content)) {
      return (
        <div className="prose prose-sm dark:prose-invert max-w-none">
          <ReactMarkdown 
            remarkPlugins={[remarkGfm, remarkMath]} 
            rehypePlugins={[rehypeKatex, rehypeHighlight, rehypeRaw]}
          >
            {content}
          </ReactMarkdown>
        </div>
      );
    }
    
    // 일반 마크다운 콘텐츠 - 키워드 하이라이트 강화
    const enhancedContent = keywords && keywords.length > 0
      ? highlightKeywordsInContent(content, keywords)
      : content;
      
    return (
      <div className="prose prose-sm dark:prose-invert max-w-none px-1">
        <ReactMarkdown 
          remarkPlugins={[remarkGfm, remarkMath]} 
          rehypePlugins={[rehypeKatex, rehypeHighlight]}
        >
          {enhancedContent}
        </ReactMarkdown>
      </div>
    );
  };
  
  return (
    <div 
      className="source-preview-modal"
      style={{padding: isFullscreen ? 0 : undefined}}
      onClick={(e) => {
        // 배경 클릭 시 모달 닫기
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div 
        className={`source-preview-container ${isFullscreen ? 'h-full max-h-full rounded-none' : 'max-h-[90vh]'}`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 헤더 */}
        <div className="source-preview-header">
          <div className="flex items-center space-x-3 overflow-hidden">
            <span className="p-2 bg-indigo-500/20 backdrop-blur-sm rounded-full flex-shrink-0">
              <FiFile className="text-indigo-400" size={16} />
            </span>
            <div className="overflow-hidden">
              <h3 className="font-medium text-gray-200 truncate max-w-xl">
                {source?.title || source?.display_name || (source?.path && source.path.split('/').pop().replace(/^[^_]*_/, ''))}
              </h3>
              <div className="flex items-center text-xs text-gray-400 mt-0.5">
                {source?.path && (
                  <span className="truncate max-w-xs opacity-70">{source.path.split('/').pop()}</span>
                )}
                {source?.page && (
                  <span className="text-xs text-gray-300 bg-gray-700/60 backdrop-blur-sm px-2 py-0.5 rounded-full ml-2 flex-shrink-0">
                    페이지 {source.page}
                  </span>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-1">
            <button
              onClick={handleCopyContent}
              className="p-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-700/60 transition-all backdrop-blur-sm"
              title="내용 복사"
              disabled={!content || isLoading || isErrorMessage(content)}
            >
              {copySuccess ? <FiCheck className="text-green-400" /> : <FiCopy />}
            </button>
            <button
              onClick={toggleFullscreen}
              className="p-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-700/60 transition-all backdrop-blur-sm"
              title={isFullscreen ? "전체화면 종료" : "전체화면으로 보기"}
            >
              {isFullscreen ? <FiMinimize2 /> : <FiMaximize2 />}
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg text-gray-400 hover:text-gray-200 hover:bg-gray-700/60 transition-all backdrop-blur-sm"
            >
              <FiX />
            </button>
          </div>
        </div>
        
        {/* 본문 */}
        <div className="source-preview-content">
          {renderContent()}
        </div>
        
        {/* 키워드 표시 영역 */}
        {keywords && keywords.length > 0 && !isErrorMessage(content) && (
          <div className="source-preview-footer">
            <p className="text-xs text-gray-400 mb-2 font-medium">관련 키워드:</p>
            <div className="flex flex-wrap gap-2">
              {keywords.map((keyword, idx) => (
                <span 
                  key={idx}
                  className="keyword-tag"
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

/**
 * 출처 아이템 컴포넌트
 */
export const SourceItem = ({ source, onClick, isFiltered = false, isCited = false, isReference = false }) => {
  // 파일명 정제
  const displayName = source.display_name || 
    (source.path ? source.path.split('/').pop() : 
    (source.source ? source.source.split('/').pop() : '알 수 없는 출처'));
  
  // 페이지 정보
  const pageInfo = source.page && source.page > 0 ? `p.${source.page}` : '';
  
  // 관련성 점수
  const relevanceScore = typeof source.score === 'number' ? 
    source.score > 0.7 ? '높음' : 
    source.score > 0.4 ? '중간' : '낮음' : '';
  
  return (
    <div 
      onClick={() => onClick(source)}
      className={`source-item ${isCited ? 'cited' : ''} ${isReference ? 'reference' : ''} ${isFiltered ? 'border-l-2 border-yellow-400' : ''}`}
      title={`${isCited ? '인용 출처' : '참고 문서'} - ${displayName}${pageInfo ? ` (${pageInfo})` : ''}`}
    >
      <div className={`source-item-icon ${isCited ? 'cited' : ''} ${isReference ? 'reference' : ''}`}>
        {isCited ? (
          <FiBookmark size={14} />
        ) : isReference ? (
          <FiFileText size={14} />
        ) : (
          <FiFile size={14} />
        )}
      </div>
      <div className="source-item-text">
        <span className={`source-item-text ${isCited ? 'cited' : ''} ${isReference ? 'reference' : ''}`}>
          {displayName}
        </span>
        {pageInfo && (
          <span className="ml-1 text-gray-500 dark:text-gray-500">
            {pageInfo}
          </span>
        )}
      </div>
      <div className="source-item-view">
        <FiExternalLink size={14} />
      </div>
    </div>
  );
};

/**
 * 출처 관리 컴포넌트
 */
export const SourcesSection = ({ sources = [], citedSources = [], onViewSource }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [filterText, setFilterText] = useState("");
  
  // 인용된 출처만 필터링 (cited_sources가 있으면 우선 사용, 없으면 is_cited=true인 항목만 필터링)
  const displayCitedSources = citedSources.length > 0 
    ? citedSources 
    : sources.filter(s => s && s.is_cited === true);
  
  // 참고 문서 - 인용되지 않은 소스 중 상위 2개 (인용된 소스가 없을 때만)
  const referenceSources = [];
  if (displayCitedSources.length === 0 && sources.length > 0) {
    // 인용된 소스가 없을 때 상위 2개 문서를 참고 문서로 표시
    const sortedSources = [...sources]
      .filter(s => s && (s.path || s.source)) // 유효한 소스만 필터링
      .sort((a, b) => (b.score || 0) - (a.score || 0)) // 점수 기준 내림차순 정렬
      .slice(0, 2); // 상위 2개만 선택
    
    referenceSources.push(...sortedSources);
  }
  
  // 출처 정보 없는 경우 처리
  if (displayCitedSources.length === 0 && referenceSources.length === 0) {
    return null;
  }
  
  // 필터링 처리
  const filteredCitedSources = filterText
    ? displayCitedSources.filter(source => {
        const sourcePath = source.path || source.source || "";
        return sourcePath.toLowerCase().includes(filterText.toLowerCase());
      })
    : displayCitedSources;
    
  const filteredReferenceSources = filterText
    ? referenceSources.filter(source => {
        const sourcePath = source.path || source.source || "";
        return sourcePath.toLowerCase().includes(filterText.toLowerCase());
      })
    : referenceSources;
  
  const hasSourcesToDisplay = filteredCitedSources.length > 0 || filteredReferenceSources.length > 0;
  
  return (
    <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
      <div className="flex items-center justify-between mb-2">
        <button
          onClick={() => setIsVisible(!isVisible)}
          className="sources-button"
        >
          {isVisible ? (
            <FiChevronDown className="mr-1" size={14} />
          ) : (
            <FiChevronRight className="mr-1" size={14} />
          )}
          {filteredCitedSources.length > 0 ? `출처 문서 (${filteredCitedSources.length})` : 
           filteredReferenceSources.length > 0 ? `참고 문서 (${filteredReferenceSources.length})` : 
           '출처 정보'}
        </button>
        
        {isVisible && (filteredCitedSources.length + filteredReferenceSources.length) > 3 && (
          <div className="relative">
            <input
              type="text"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="출처 검색..."
              className="text-xs py-1 px-2 w-32 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 focus:outline-none focus:ring-1 focus:ring-indigo-500"
            />
            {filterText && (
              <button
                onClick={() => setFilterText('')}
                className="absolute right-1 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <FiX size={12} />
              </button>
            )}
          </div>
        )}
      </div>
      
      {isVisible && (
        <div className="sources-list">
          {/* 인용 출처 표시 */}
          {filteredCitedSources.length > 0 && (
            <>
              {filteredCitedSources.length > 0 && filteredReferenceSources.length > 0 && (
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 mt-1 px-2">
                  인용 출처:
                </div>
              )}
              {filteredCitedSources.map((source, index) => (
                <SourceItem 
                  key={`cited-${source.source || source.path}-${source.page}-${index}`}
                  source={source}
                  onClick={onViewSource}
                  isCited={true}
                />
              ))}
            </>
          )}
          
          {/* 참고 문서 표시 */}
          {filteredReferenceSources.length > 0 && (
            <>
              {filteredCitedSources.length > 0 && filteredReferenceSources.length > 0 && (
                <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 mt-2 px-2">
                  참고 문서:
                </div>
              )}
              {filteredReferenceSources.map((source, index) => (
                <SourceItem 
                  key={`ref-${source.source || source.path}-${source.page}-${index}`}
                  source={source}
                  onClick={onViewSource}
                  isReference={true}
                />
              ))}
            </>
          )}
          
          {/* 필터링 결과가 없는 경우 */}
          {!hasSourcesToDisplay && (
            <p className="text-xs text-gray-500 dark:text-gray-400 py-2 text-center">
              {filterText ? '검색 결과가 없습니다.' : '출처 정보가 없습니다.'}
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default { SourcePreviewModal, SourceItem, SourcesSection }; 