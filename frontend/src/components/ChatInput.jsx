import { useState, useRef, useEffect, forwardRef } from 'react';
import { FiLoader, FiSend, FiPaperclip, FiX, FiCheck, FiImage, FiMessageSquare } from 'react-icons/fi';

function FileUpload({ onClose, categories, onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [category, setCategory] = useState(categories[0] || '메뉴얼');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [dragActive, setDragActive] = useState(false);
  
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };
  
  const handleDragLeave = () => {
    setDragActive(false);
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setUploadStatus('파일을 선택해주세요.');
      return;
    }
    
    setIsUploading(true);
    setUploadStatus('업로드 중...');
    
    const formData = new FormData();
    formData.append('files', file);
    formData.append('category', category);
    
    try {
      const response = await fetch('http://172.10.2.70:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      
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
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white dark:bg-slate-900 rounded-xl p-6 w-full max-w-md shadow-xl animate-slide-up">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-medium text-slate-800 dark:text-slate-200">파일 업로드</h2>
          <button 
            onClick={onClose}
            className="text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 rounded-full p-1 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <FiX size={20} />
          </button>
        </div>
        
        <div>
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1.5 text-slate-700 dark:text-slate-300">카테고리</label>
            <select 
              className="input w-full"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              {categories.map((cat) => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium mb-1.5 text-slate-700 dark:text-slate-300">파일 선택</label>
            <div 
              className={`drop-area ${dragActive ? 'active' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              {file ? (
                <div className="flex items-center justify-center space-x-2">
                  <FiCheck className="text-green-500" size={20} />
                  <span className="text-slate-800 dark:text-slate-200 text-sm truncate max-w-[200px]">
                    {file.name}
                  </span>
                </div>
              ) : (
                <div className="text-slate-500 dark:text-slate-400">
                  <FiPaperclip className="mx-auto mb-2" size={24} />
                  <p className="text-sm">파일을 드래그하거나 클릭하여 선택하세요</p>
                </div>
              )}
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="hidden"
              />
            </div>
          </div>
          
          {uploadStatus && (
            <div className={`mt-2 text-sm text-center p-2 rounded-md ${
              uploadStatus.includes('성공') 
                ? 'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-400' 
                : uploadStatus.includes('실패') || uploadStatus.includes('오류')
                  ? 'text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-400'
                  : 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 dark:text-blue-400'
            }`}>
              {uploadStatus}
            </div>
          )}
          
          <div className="flex justify-end space-x-3 mt-5">
            <button
              type="button"
              onClick={onClose}
              className="btn btn-secondary"
            >
              취소
            </button>
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!file || isUploading}
              className="btn btn-primary flex items-center justify-center gap-2"
            >
              {isUploading ? (
                <>
                  <FiLoader className="animate-spin" size={16} />
                  <span>업로드 중...</span>
                </>
              ) : (
                <span>업로드</span>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

const ChatInput = forwardRef(function ChatInput({ onSend, disabled, onTyping }, ref) {
  const [msg, setMsg] = useState('');
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [suggestions, setSuggestions] = useState([]);
  const [files, setFiles] = useState([]);
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const categories = ['메뉴얼', '장애보고서', '기술문서', '기타'];
  
  useEffect(() => {
    if (ref) {
      if (typeof ref === 'function') {
        ref(textareaRef.current);
      } else {
        ref.current = textareaRef.current;
      }
    }
  }, [ref]);
  
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
      onTyping(true);
    } else {
      onTyping(false);
    }
  }, [msg, disabled, onTyping]);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
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
        setFiles(prevFiles => [...prevFiles, ...newFiles]);
      });
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  useEffect(() => {
    if (msg.trim().length > 0) {
      const filteredSuggestions = history
        .filter(item => item.toLowerCase().includes(msg.toLowerCase()))
        .slice(0, 3);
      setSuggestions(filteredSuggestions);
    } else {
      setSuggestions([]);
    }
  }, [msg, history]);

  const handleSuggestionSelect = (suggestion) => {
    setMsg(suggestion);
    setSuggestions([]);
    textareaRef.current?.focus();
  };

  const handleUploadSuccess = (message) => {
    setUploadStatus(message);
    setTimeout(() => {
      setUploadStatus('');
    }, 3000);
  };

  const handleFileUploadClick = () => {
    setShowFileUpload(true);
  };

  const handleSendMessage = () => {
    if (msg.trim() && !disabled) {
      onSend(msg.trim());
      setHistory(prev => {
        const updatedHistory = prev.includes(msg.trim()) 
          ? prev 
          : [...prev, msg.trim()];
        return updatedHistory.slice(-20);
      });
      setMsg('');
      setHistoryIndex(-1);
      textareaRef.current?.focus();
    }
  };

  const handleRemoveFile = (indexToRemove) => {
    setFiles(files.filter((_, index) => index !== indexToRemove));
  };

  const handleKeyDown = (e) => {
    if (e.key === 'ArrowUp' && history.length > 0 && msg === '') {
      e.preventDefault();
      const newIndex = historyIndex < history.length - 1 ? historyIndex + 1 : historyIndex;
      setHistoryIndex(newIndex);
      setMsg(history[history.length - 1 - newIndex] || '');
    } else if (e.key === 'ArrowDown' && historyIndex > -1) {
      e.preventDefault();
      const newIndex = historyIndex - 1;
      setHistoryIndex(newIndex);
      setMsg(newIndex >= 0 ? history[history.length - 1 - newIndex] : '');
    }
    
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="px-4 py-3 relative">
      {uploadStatus && (
        <div className="absolute -top-14 left-0 right-0 mx-auto w-max px-4 py-2 rounded-lg 
          bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-200 
          shadow-lg border border-green-100 dark:border-green-800 animate-fade-in-up z-10">
          <div className="flex items-center gap-2">
            <FiCheck className="text-green-500 dark:text-green-400" size={16} />
            <span className="text-sm font-medium">{uploadStatus}</span>
          </div>
        </div>
      )}
  
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {files.map((fileObj, index) => (
            <div 
              key={index} 
              className="relative bg-white dark:bg-slate-800 rounded-md p-2 pr-8 text-sm 
                flex items-center gap-2 shadow-sm border border-slate-200 dark:border-slate-700 animate-fade-in"
            >
              <FiImage size={14} className="text-blue-500 dark:text-blue-400" />
              <span className="truncate max-w-[200px] text-slate-700 dark:text-slate-300">
                {fileObj.file.name}
              </span>
              <button
                onClick={() => handleRemoveFile(index)}
                className="absolute right-1.5 top-1.5 text-slate-400 hover:text-red-500 
                  dark:text-slate-400 dark:hover:text-red-400 p-0.5 rounded-full 
                  hover:bg-red-50 dark:hover:bg-red-900/20 custom-transition-colors"
                title="파일 제거"
              >
                <FiX size={14} />
              </button>
            </div>
          ))}
        </div>
      )}
  
      <div className={`relative flex items-end rounded-xl overflow-hidden custom-transition-all
        ${isFocused 
          ? 'shadow-lg dark:shadow-slate-800/30 ring-2 ring-blue-500/30 dark:ring-blue-500/20' 
          : 'shadow-md dark:shadow-slate-800/20'}
        ${disabled ? 'opacity-80' : ''}
      `}>
        <textarea
          ref={textareaRef}
          value={msg}
          onChange={(e) => setMsg(e.target.value)}
          onKeyDown={handleKeyDown}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          placeholder={disabled ? "메시지를 처리 중입니다..." : "메시지를 입력하세요..."}
          disabled={disabled}
          className="flex-1 resize-none p-4 pr-20 h-14 max-h-[140px] bg-white dark:bg-slate-800 
            outline-none text-slate-800 dark:text-slate-200 placeholder-slate-400 
            dark:placeholder-slate-500 border border-slate-200 dark:border-slate-700 rounded-xl"
          style={{ overflowY: 'auto' }}
        />
        
        <div className="absolute bottom-2.5 right-3 flex items-center gap-1.5">
          <button
            onClick={handleFileUploadClick}
            disabled={disabled}
            className="p-2 rounded-lg text-slate-500 hover:text-slate-700 hover:bg-slate-100 
              dark:text-slate-400 dark:hover:text-slate-200 dark:hover:bg-slate-700/70 custom-transition-colors"
            title="파일 첨부"
          >
            <FiPaperclip size={18} />
          </button>
          
          <button
            onClick={handleSendMessage}
            disabled={disabled || !msg.trim()}
            className={`p-2 rounded-lg custom-transition-colors ${
              msg.trim() && !disabled
                ? 'text-blue-600 hover:bg-blue-50 dark:text-blue-400 dark:hover:bg-blue-900/20'
                : 'text-slate-400 dark:text-slate-600 cursor-not-allowed'
            }`}
            title="전송"
          >
            {disabled ? (
              <FiLoader size={18} className="animate-spin" />
            ) : (
              <FiSend size={18} />
            )}
          </button>
        </div>
      </div>
  
      {suggestions.length > 0 && !disabled && (
        <div className="absolute bottom-full left-0 right-0 mb-2 bg-white dark:bg-slate-800 
          border border-slate-200 dark:border-slate-700 rounded-lg shadow-lg overflow-hidden z-10 animate-fade-in-up">
          {suggestions.map((suggestion, index) => (
            <button
              key={index}
              className="w-full text-left px-4 py-2.5 hover:bg-slate-100 dark:hover:bg-slate-700 
                text-slate-800 dark:text-slate-200 text-sm truncate border-b 
                border-slate-100 dark:border-slate-700 last:border-0"
              onClick={() => handleSuggestionSelect(suggestion)}
            >
              <div className="flex items-center gap-2">
                <FiMessageSquare size={14} className="text-blue-500 dark:text-blue-400" />
                <span>{suggestion}</span>
              </div>
            </button>
          ))}
        </div>
      )}
  
      {showFileUpload && (
        <FileUpload
          onClose={() => setShowFileUpload(false)}
          categories={categories}
          onUploadSuccess={handleUploadSuccess}
        />
      )}
      
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        multiple
        className="hidden"
      />
    </div>
  );
});

export default ChatInput;
