import React, { useState, useRef, useEffect } from 'react';
import { FiDatabase, FiSearch, FiCode, FiTable, FiSend, FiLoader, FiAlertCircle, FiHelpCircle, FiInfo } from 'react-icons/fi';

// 마크다운 테이블을 HTML 테이블로 변환하는 함수
const markdownTableToHtml = (markdown) => {
  if (!markdown || !markdown.startsWith('|')) return null;
  
  try {
    const rows = markdown.split('\n').filter(row => row.trim().startsWith('|'));
    if (rows.length < 2) return null; // 헤더와 구분선이 최소한 필요
    
    // 헤더 행 처리
    const headerRow = rows[0];
    const headers = headerRow
      .split('|')
      .slice(1, -1) // 첫 번째와 마지막 '|' 제거
      .map(header => header.trim());
    
    // 데이터 행 처리 (구분선 행 건너뛰기)
    const dataRows = rows.slice(2);
    const data = dataRows.map(row => {
      return row
        .split('|')
        .slice(1, -1) // 첫 번째와 마지막 '|' 제거
        .map(cell => cell.trim());
    });
    
    return (
      <div className="overflow-x-auto bg-gray-900 rounded-lg shadow-md">
        <table className="min-w-full divide-y divide-gray-700">
          <thead className="bg-gray-800">
            <tr>
              {headers.map((header, index) => (
                <th 
                  key={index} 
                  className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-700">
            {data.map((row, rowIndex) => (
              <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800/50'}>
                {row.map((cell, cellIndex) => (
                  <td 
                    key={cellIndex} 
                    className="px-4 py-2 text-sm text-gray-200"
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  } catch (e) {
    console.error("테이블 변환 오류:", e);
    return <div className="text-red-400 mt-2 text-sm">{markdown}</div>;
  }
};

const SQLQueryPage = () => {
  const [question, setQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [generatedSQL, setGeneratedSQL] = useState('');
  const [queryResult, setQueryResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [dbSchema, setDbSchema] = useState('');
  const [showSchema, setShowSchema] = useState(false);
  const [llmResponse, setLlmResponse] = useState('');
  const [showLLMResponse, setShowLLMResponse] = useState(false);
  const inputRef = useRef(null);
  
  // 초기 로딩 시 DB 스키마 정보 가져오기
  useEffect(() => {
    const fetchDbSchema = async () => {
      try {
        // 상대 경로로 변경
        const response = await fetch('/api/db-schema');
        if (!response.ok) {
          const text = await response.text();
          console.error('스키마 로딩 실패 응답:', text);
          throw new Error(`스키마 로딩 실패: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // 응답 형식 변경에 따른 처리 추가
        if (data.status === 'error') {
          console.error('DB 스키마 로딩 오류:', data.error);
          setErrorMessage('데이터베이스 스키마를 불러올 수 없습니다.');
          // 오류가 있어도 schema 필드에 메시지가 들어있으면 표시
          if (data.schema) {
            setDbSchema(data.schema);
          }
        } else {
          setDbSchema(data.schema || '스키마 정보가 없습니다.');
        }
      } catch (error) {
        console.error('DB 스키마 로딩 오류:', error);
        setErrorMessage('데이터베이스 스키마를 불러올 수 없습니다.');
      }
    };
    
    fetchDbSchema();
  }, []);
  
  // SQL 쿼리 실행 함수
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!question.trim()) {
      setErrorMessage('질문을 입력해주세요.');
      inputRef.current?.focus();
      return;
    }
    
    setIsLoading(true);
    setErrorMessage('');
    setGeneratedSQL('');
    setQueryResult(null);
    setLlmResponse('');
    setShowLLMResponse(false);
    
    try {
      // 상대 경로로 변경
      const response = await fetch('/api/sql-query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question.trim() }),
      });
      
      // 오류 응답 처리 개선
      if (!response.ok) {
        const errorText = await response.text();
        console.error('API 오류 응답:', errorText);
        throw new Error(errorText || '쿼리 실행 중 오류가 발생했습니다.');
      }
      
      const data = await response.json();
      setGeneratedSQL(data.sql);
      setQueryResult(data.results);
      
    } catch (error) {
      console.error('SQL 쿼리 오류:', error);
      setErrorMessage(error.message || '서버 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // SQL-LLM 통합 API 호출 함수 추가
  const handleSqlLlmQuery = async () => {
    if (!question.trim()) {
      setErrorMessage('질문을 입력해주세요.');
      inputRef.current?.focus();
      return;
    }
    
    setIsLoading(true);
    setErrorMessage('');
    setGeneratedSQL('');
    setQueryResult(null);
    setLlmResponse('');
    setShowLLMResponse(true);
    
    try {
      // 상대 경로로 변경
      const response = await fetch('/api/sql-and-llm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question.trim() }),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('SQL-LLM API 오류 응답:', errorText);
        throw new Error(errorText || 'SQL-LLM 쿼리 실행 중 오류가 발생했습니다.');
      }
      
      const data = await response.json();
      setGeneratedSQL(data.sql_query);
      setQueryResult(data.sql_result);
      setLlmResponse(data.bot_response);
      
    } catch (error) {
      console.error('SQL-LLM 쿼리 오류:', error);
      setErrorMessage(error.message || '서버 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };
  
  // 예시 질문 클릭 핸들러
  const handleExampleClick = (exampleQuestion) => {
    setQuestion(exampleQuestion);
    inputRef.current?.focus();
  };
  
  return (
    <div className="flex flex-col h-full bg-gray-900 text-gray-100">
      {/* 헤더 - 더 모던한 디자인으로 수정 */}
      <div className="h-16 flex items-center justify-between px-6 bg-gray-900 border-b border-gray-800 shadow-sm z-10">
        <div className="flex items-center">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 mr-4 flex items-center justify-center shadow-md">
            <FiDatabase size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent">
              SQL 쿼리 도우미
            </h1>
            <p className="text-xs text-gray-400 mt-0.5">데이터베이스 질의 인터페이스</p>
          </div>
        </div>
        
        <button 
          onClick={() => setShowSchema(!showSchema)}
          className="px-3 py-1.5 rounded-lg text-gray-400 hover:text-gray-200 transition-colors
                    flex items-center gap-2"
        >
          <FiInfo size={16} />
          <span>DB 스키마 {showSchema ? '숨기기' : '보기'}</span>
        </button>
      </div>
      
      <div className="flex-1 overflow-hidden flex flex-col md:flex-row">
        {/* 메인 영역 */}
        <div className="flex-1 p-4 pb-0 overflow-y-auto flex flex-col">
          {/* 폼 - 색상 수정 */}
          <form onSubmit={handleSubmit} className="mb-6">
            <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 mb-6 shadow-lg">
              <div className="flex items-start space-x-3 mb-4">
                <div className="mt-1 text-indigo-400">
                  <FiHelpCircle size={22} />
                </div>
                <div>
                  <h2 className="text-white font-medium text-lg mb-1">자연어로 SQL 질문하기</h2>
                  <p className="text-gray-400 text-sm">
                    데이터베이스에 질문을 자연어로 입력하면 SQL로 변환하여 결과를 보여드립니다.
                  </p>
                </div>
              </div>
              
              <div className="relative">
                <input
                  ref={inputRef}
                  type="text"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="데이터베이스에 자연어로 질문하세요. 예: '사용자 목록을 보여줘'"
                  className="w-full px-4 pr-12 py-3 bg-gray-800 border border-gray-700 focus:border-indigo-500 
                          focus:ring-2 focus:ring-indigo-500/20 rounded-lg outline-none transition-all"
                />
                <button
                  type="submit"
                  disabled={isLoading || !question.trim()}
                  className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 text-indigo-400 
                          hover:text-indigo-300 disabled:text-gray-500 disabled:hover:text-gray-500 transition-colors"
                >
                  {isLoading ? <FiLoader className="animate-spin" size={20} /> : <FiSearch size={20} />}
                </button>
              </div>
              
              {/* 버튼 영역 추가 */}
              <div className="flex mt-4 space-x-3">
                <button
                  type="submit"
                  disabled={isLoading || !question.trim()}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded-md text-white 
                         flex items-center gap-2 disabled:bg-gray-700 disabled:text-gray-400 transition-colors"
                >
                  <FiCode size={18} />
                  <span>SQL 생성</span>
                </button>
                
                <button
                  type="button"
                  onClick={handleSqlLlmQuery}
                  disabled={isLoading || !question.trim()}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-md text-white 
                         flex items-center gap-2 disabled:bg-gray-700 disabled:text-gray-400 transition-colors"
                >
                  <FiSend size={18} />
                  <span>AI 응답 요청</span>
                </button>
              </div>
            </div>
          </form>
          
          {/* 예시 질문 */}
          <div className="mb-6">
            <h3 className="text-sm font-medium text-gray-400 mb-2 flex items-center gap-1.5">
              <FiSearch size={14} />
              <span>예시 질문</span>
            </h3>
            <div className="flex flex-wrap gap-2">
              {[
                '모든 사용자 목록을 보여줘',
                '최근에 가입한 사용자 5명을 보여줘',
                '이름이 "김"으로 시작하는 사용자를 검색해줘',
                '가장 많은 게시물을 작성한 사용자는?',
                '사용자 자원 통계를 보여줘'
              ].map((example, index) => (
                <button
                  key={index}
                  onClick={() => handleExampleClick(example)}
                  className="text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 hover:border-gray-600 px-3 py-1.5 rounded-full transition-colors"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
          
          {/* 에러 메시지 */}
          {errorMessage && (
            <div className="p-4 bg-red-900/30 border border-red-800 rounded-lg mb-6 flex items-start gap-3">
              <FiAlertCircle className="text-red-500 mt-0.5 shrink-0" size={20} />
              <p className="text-red-300">{errorMessage}</p>
            </div>
          )}
          
          {/* LLM 응답 UI 추가 */}
          {showLLMResponse && llmResponse && (
            <div className="p-5 bg-gradient-to-b from-purple-900/20 to-indigo-900/10 border border-purple-800/50 rounded-lg mb-6 shadow-lg">
              <div className="flex items-start gap-3 mb-3">
                <div className="mt-1 p-1.5 bg-purple-500 rounded-full">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-4 h-4 text-white">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-11v4h2V9h-2z" />
                    <path d="M12 17a1 1 0 100-2 1 1 0 000 2z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-white font-medium text-lg">AI 응답</h3>
                  <p className="text-gray-300 mt-1">{llmResponse}</p>
                </div>
              </div>
            </div>
          )}
          
          {/* 생성된 SQL */}
          {generatedSQL && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-400 mb-2 flex items-center gap-1.5">
                <FiCode size={14} />
                <span>생성된 SQL 쿼리</span>
              </h3>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-3 overflow-x-auto">
                <pre className="text-sm text-indigo-300 font-mono">{generatedSQL}</pre>
              </div>
            </div>
          )}
          
          {/* 쿼리 결과 */}
          {queryResult && (
            <div className="mb-6">
              <h3 className="text-sm font-medium text-gray-400 mb-2 flex items-center gap-1.5">
                <FiTable size={14} />
                <span>쿼리 결과</span>
              </h3>
              
              {queryResult.startsWith('⚠️') || queryResult.startsWith('❌') ? (
                <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 text-yellow-400">
                  {queryResult}
                </div>
              ) : (
                markdownTableToHtml(queryResult) || (
                  <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 text-gray-300">
                    {queryResult}
                  </div>
                )
              )}
            </div>
          )}
        </div>
        
        {/* 사이드바 - DB 스키마 */}
        {showSchema && (
          <div className="w-full md:w-72 lg:w-96 overflow-y-auto bg-gray-900 border-l border-gray-800 p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-1.5">
              <FiDatabase size={14} />
              <span>데이터베이스 스키마</span>
            </h3>
            
            {dbSchema ? (
              <pre className="text-xs text-gray-400 font-mono bg-gray-800 border border-gray-700 p-3 rounded-lg overflow-x-auto whitespace-pre-wrap">
                {dbSchema}
              </pre>
            ) : (
              <div className="text-gray-500 text-sm italic">
                스키마 정보를 불러오는 중...
              </div>
            )}
            
            <div className="mt-6 border-t border-gray-800 pt-4">
              <h4 className="text-xs font-medium text-gray-400 mb-2">사용 팁</h4>
              <ul className="text-xs text-gray-500 space-y-2">
                <li className="flex items-start gap-2">
                  <span className="text-gray-400 mt-0.5">•</span>
                  <span>자연어로 데이터베이스에 질문할 수 있습니다. (예: "사용자 목록을 보여줘")</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-gray-400 mt-0.5">•</span>
                  <span>질문은 구체적일수록 더 정확한 결과를 얻을 수 있습니다.</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-gray-400 mt-0.5">•</span>
                  <span>테이블 이름과 필드명을 명확하게 언급하면 좋습니다.</span>
                </li>
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SQLQueryPage; 