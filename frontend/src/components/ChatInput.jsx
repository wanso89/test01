import { useState, useRef, useEffect } from 'react';
import { FiLoader, FiSend, FiPaperclip, FiX } from 'react-icons/fi';

function FileUpload({ onClose, categories, onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [category, setCategory] = useState(categories[0] || '메뉴얼');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log('handleSubmit 호출됨, file:', file);
    if (!file) {
      console.log('파일이 선택되지 않음');
      setUploadStatus('파일을 선택해주세요.');
      return;
    }
    
    setIsUploading(true);
    setUploadStatus('업로드 중...');
    console.log('업로드 시작, category:', category);
    
    const formData = new FormData();
    formData.append('files', file); // 다중 파일을 위해 'files'로 이름 변경
    formData.append('category', category);
    
    try {
      console.log('fetch 요청 전송 시작');
      const response = await fetch('http://172.10.2.70:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      console.log('fetch 응답 수신:', response.status);
      const data = await response.json();
      console.log('업로드 응답:', data); // 개발자 도구에서 확인 가능
      if (data.results && data.results.length > 0) {
        const result = data.results[0];
        if (result.status === 'success') {
          setUploadStatus('업로드 성공!');
          onUploadSuccess(result.message);
          setTimeout(() => {
            onClose();
          }, 2000);
        } else {
          setUploadStatus(`업로드 실패: ${result.message || '알 수 없는 오류'}`);
        }
      } else {
        setUploadStatus(`업로드 실패: 응답 데이터 오류`);
      }
    } catch (error) {
      console.error('업로드 실패:', error);
      setUploadStatus(`업로드 오류: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-96 max-w-full">
        <h2 className="text-xl font-bold mb-4">파일 업로드</h2>
        
        <div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">카테고리</label>
            <select 
              className="w-full p-2 border rounded-md"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              {categories.map((cat) => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1">파일 선택</label>
            <input 
              type="file" 
              onChange={(e) => setFile(e.target.files[0])}
              className="w-full p-2 border rounded-md"
            />
          </div>
          
          {uploadStatus && (
            <div className="mt-2 text-sm text-center text-green-600">
              {uploadStatus}
            </div>
          )}
          
          <div className="flex justify-end space-x-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border rounded-md"
            >
              취소
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!file || isUploading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md disabled:bg-blue-300"
            >
              {isUploading ? '업로드 중...' : '업로드'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function ChatInput({ onSend, disabled, onTyping }) {
  const [msg, setMsg] = useState('');
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [suggestions, setSuggestions] = useState([]);
  const [files, setFiles] = useState([]); // 다중 파일을 배열로 관리
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const categories = ['메뉴얼', '장애보고서', '기술문서', '기타']; // 카테고리 목록
  
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 120)}px`;
    }
  }, [msg]);

  useEffect(() => {
    if (msg.trim().length > 0 && !disabled) {
      onTyping(true); // 입력 중 상태 전달
    } else {
      onTyping(false); // 입력 중 상태 해제
    }
  }, [msg, disabled, onTyping]);

  // 파일 선택 핸들러 (다중 파일 선택 지원)
  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files); // FileList를 배열로 변환
    if (selectedFiles.length > 0) {
      const updatedFiles = selectedFiles.map(file => {
        if (file.type.startsWith('image/')) {
          return new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = (event) => {
              resolve({ file, preview: event.target.result });
            };
            reader.readAsDataURL(file);
          });
        }
        return { file, preview: null };
      });
      Promise.all(updatedFiles).then(newFiles => {
        setFiles(prevFiles => [...prevFiles, ...newFiles]); // 기존 파일에 추가
      });
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''; // 파일 입력 초기화
    }
  };

  // 입력 중 자동 완성 제안 생성 (히스토리 기반)
  useEffect(() => {
    if (msg.trim().length > 0) {
      const filteredSuggestions = history
        .filter(item => item.toLowerCase().includes(msg.toLowerCase()))
        .slice(0, 3); // 최대 3개 제안
      setSuggestions(filteredSuggestions);
    } else {
      setSuggestions([]);
    }
  }, [msg, history]);

  // 자동 완성 제안 선택 핸들러
  const handleSuggestionSelect = (suggestion) => {
    setMsg(suggestion);
    setSuggestions([]);
    textareaRef.current?.focus();
  };

  // 파일 업로드 성공 콜백
  const handleUploadSuccess = (message) => {
    setUploadStatus(message);
    setTimeout(() => {
      setUploadStatus('');
    }, 3000);
  };

  // 파일 첨부 버튼 클릭 시 모달 표시
  const handleFileUploadClick = () => {
    setShowFileUpload(true);
  };

  // 메시지 전송 함수
  const handleSendMessage = () => {
    if ((msg.trim() || files.length > 0) && !disabled) {
      let content = msg.trim();
      if (files.length > 0) {
        const fileNames = files.map(f => f.file.name).join(', ');
        content = content ? `${content} (첨부파일: ${fileNames})` : `첨부파일: ${fileNames}`;
      }
      onSend(content);
      setHistory(prev => [content, ...prev]); // 입력 히스토리에 추가
      setMsg('');
      setHistoryIndex(-1); // 히스토리 인덱스 초기화
      setSuggestions([]); // 제안 초기화
      setFiles([]); // 파일 초기화
    }
  };

  // 파일 제거 핸들러 (특정 파일만 제거)
  const handleRemoveFile = (indexToRemove) => {
    setFiles(prevFiles => prevFiles.filter((_, index) => index !== indexToRemove));
  };

  // handleKeyDown 함수
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if ((msg.trim() || files.length > 0) && !disabled) {
        handleSendMessage();
      }
    } else if (e.key === 'ArrowUp' && history.length > 0) {
      e.preventDefault();
      if (historyIndex < history.length - 1) {
        setHistoryIndex(prev => prev + 1);
        setMsg(history[historyIndex + 1]);
      }
    } else if (e.key === 'ArrowDown' && history.length > 0) {
      e.preventDefault();
      if (historyIndex > 0) {
        setHistoryIndex(prev => prev - 1);
        setMsg(history[historyIndex - 1]);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setMsg('');
      }
    } else if (e.key === 'Tab' && suggestions.length > 0) {
      e.preventDefault();
      handleSuggestionSelect(suggestions[0]); // 첫 번째 제안 선택
    }
  };

  return (
    <div className="flex flex-col p-4 border-t bg-white dark:bg-gray-800">
      <form
        className="flex"
        onSubmit={e => {
          e.preventDefault();
          handleSendMessage();
        }}
      >
        <div className="flex-1">
          <textarea
            ref={textareaRef}
            className="flex-1 p-3 rounded-full border-2 border-gray-200 dark:border-gray-700 focus:ring-2 focus:ring-blue-400 bg-white dark:bg-gray-700 text-gray-800 dark:text-white transition resize-none overflow-hidden"
            placeholder={disabled ? "전송 중..." : "메시지를 입력하세요. Shift+Enter로 줄바꿈, Enter로 전송됩니다."}
            value={msg}
            onChange={e => setMsg(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={disabled}
            style={{ minHeight: 40, maxHeight: 120, scrollbarWidth: 'none', msOverflowStyle: 'none', width: '100%' }}
          />
          {suggestions.length > 0 && (
            <div className="absolute bottom-16 left-0 w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded shadow-lg z-10 max-h-40 overflow-y-auto">
              {suggestions.map((suggestion, index) => (
                <div
                  key={index}
                  className="px-3 py-2 text-gray-800 dark:text-gray-100 hover:bg-blue-100 dark:hover:bg-blue-800 transition cursor-pointer"
                  onClick={() => handleSuggestionSelect(suggestion)}
                >
                  {suggestion}
                </div>
              ))}
            </div>
          )}
          {files.length > 0 && (
            <div className="mt-2 max-h-24 overflow-y-auto bg-gray-50 dark:bg-gray-700 rounded border border-gray-200 dark:border-gray-600 p-2">
              {files.map((fileItem, index) => (
                <div key={index} className="flex items-center justify-between mb-1 bg-gray-100 dark:bg-gray-800 rounded px-2 py-1">
                  {fileItem.preview ? (
                    <div className="flex items-center">
                      <img src={fileItem.preview} alt="미리보기" className="w-8 h-8 object-cover rounded mr-2" />
                      <span 
                        className="text-gray-800 dark:text-gray-100 text-xs overflow-wrap-break-word" 
                        style={{ overflowWrap: 'break-word' }} 
                        title={fileItem.file.name}
                      >
                        {fileItem.file.name}
                      </span>
                    </div>
                  ) : (
                    <span 
                      className="text-gray-800 dark:text-gray-100 text-xs overflow-wrap-break-word" 
                      style={{ overflowWrap: 'break-word' }} 
                      title={fileItem.file.name}
                    >
                      {fileItem.file.name}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => handleRemoveFile(index)}
                    className="ml-2 text-gray-500 dark:text-gray-400 hover:text-red-500 dark:hover:text-red-400"
                    title="파일 제거"
                  >
                    <FiX size={12} />
                  </button>
                </div>
              ))}
            </div>
          )}
          {uploadStatus && (
            <div className="mt-2 text-sm text-center text-green-600">
              {uploadStatus}
            </div>
          )}
        </div>
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          onChange={handleFileChange}
          accept="image/*,application/pdf,.doc,.docx"
          multiple // 다중 파일 선택 가능
        />
        <button
          type="button"
          onClick={handleFileUploadClick}
          className="ml-2 px-3 py-2 rounded-full text-gray-600 dark:text-gray-300 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
          disabled={disabled}
          title="파일 첨부"
          style={{ minWidth: 40, minHeight: 40, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          <FiPaperclip size={18} />
        </button>
        <button
          type="submit"
          className={`ml-2 px-6 py-2 rounded-full text-white font-bold shadow transition transform duration-200 ${
            disabled 
              ? 'bg-gray-400 dark:bg-gray-600 cursor-not-allowed' 
              : 'bg-gradient-to-r from-blue-500 to-indigo-500 hover:from-blue-600 hover:to-indigo-600 hover:scale-105'
          }`}
          disabled={disabled}
        >
          {disabled ? <FiLoader className="animate-spin" size={18} /> : <FiSend size={18} />}
        </button>
      </form>
      {showFileUpload && (
        <FileUpload 
          onClose={() => setShowFileUpload(false)} 
          categories={categories} 
          onUploadSuccess={handleUploadSuccess} 
        />
      )}
    </div>
  );
}
export default ChatInput;
