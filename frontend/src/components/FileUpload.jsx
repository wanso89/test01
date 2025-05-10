import { useState, useRef, useEffect } from "react";
import { FiLoader, FiX, FiPaperclip, FiCheck, FiFolder, FiFile, FiUploadCloud } from 'react-icons/fi';
import ReactDOM from 'react-dom';

function FileUpload({ onClose, categories = ['메뉴얼'], onUploadSuccess, initialCategory = '메뉴얼', initialFiles = [], containerSelector, showCategories = false }) {
  const [files, setFiles] = useState([]);
  const [category, setCategory] = useState(initialCategory || '메뉴얼');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [fileResults, setFileResults] = useState([]);
  
  const fileInputRef = useRef(null);

  // initialFiles가 전달된 경우 상태를 초기화
  useEffect(() => {
    if (initialFiles && initialFiles.length > 0) {
      setFiles(initialFiles);
    }
  }, [initialFiles]);

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
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const newFiles = Array.from(e.dataTransfer.files);
      setFiles(prevFiles => [...prevFiles, ...newFiles]);
    }
  };
  
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles = Array.from(e.target.files);
      setFiles(prevFiles => [...prevFiles, ...newFiles]);
    }
  };
  
  const removeFile = (index) => {
    setFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length === 0) {
      setUploadStatus('파일을 선택해주세요.');
      return;
    }
    
    setIsUploading(true);
    setUploadStatus('업로드 중...');
    setUploadProgress(0);
    setFileResults([]);
    
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    
    // 카테고리는 항상 '메뉴얼'로 고정
    formData.append('category', '메뉴얼'); 
    
    try {
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 5, 90)); // 진행 표시용 (실제 진행도는 아님)
      }, 500);
      
      const response = await fetch('http://172.10.2.70:8000/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      clearInterval(progressInterval);
      const data = await response.json();
      
      // 파일별 결과 저장
      if (data.results && data.results.length > 0) {
        setFileResults(data.results);
        setUploadProgress(100);
        
        const successCount = data.results.filter(r => r.status === 'success').length;
        const skipCount = data.results.filter(r => r.status === 'skipped').length;
        
        if (successCount + skipCount === files.length) {
          setUploadStatus(`${files.length}개 파일 처리 완료! (${successCount}개 인덱싱, ${skipCount}개 건너뜀)`);
          
          if (onUploadSuccess) {
            onUploadSuccess(files, category);
          }
          
          // 3초 후 모달 닫기 (사용자가 결과를 볼 수 있도록)
          setTimeout(() => {
            onClose();
          }, 3000);
        } else {
          setUploadStatus(`${successCount}/${files.length} 파일 업로드 성공, ${skipCount}개 건너뜀`);
          
          if (onUploadSuccess) {
            onUploadSuccess(files.slice(0, successCount), category);
          }
        }
      } else {
        setUploadStatus(`업로드 실패: 응답 데이터 오류`);
      }
    } catch (error) {
      console.error('업로드 실패:', error);
      setUploadStatus(`업로드 오류: ${error.message}`);
      setUploadProgress(0);
    } finally {
      setIsUploading(false);
    }
  };
  
  // 파일 크기 표시 형식화 (예: 2.5MB)
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  // 파일 결과 상태에 따른 아이콘 및 색상 클래스
  const getStatusClasses = (status) => {
    switch(status) {
      case 'success':
        return { icon: <FiCheck size={16} />, className: 'text-green-500 bg-green-100 dark:bg-green-900/30' };
      case 'error':
        return { icon: <FiX size={16} />, className: 'text-red-500 bg-red-100 dark:bg-red-900/30' };
      case 'skipped':
        return { icon: <FiCheck size={16} />, className: 'text-blue-500 bg-blue-100 dark:bg-blue-900/30' };
      case 'processing':
      default:
        return { icon: <FiLoader size={16} className="animate-spin" />, className: 'text-indigo-500 bg-indigo-100 dark:bg-indigo-900/30' };
    }
  };

  // 모달 콘텐츠
  const modalContent = (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 w-full max-w-md shadow-xl overflow-hidden relative animate-fade-in-up">
        {/* 배경 장식 */}
        <div className="absolute -right-20 -top-20 w-64 h-64 bg-gradient-to-br from-indigo-500/20 to-purple-500/10 rounded-full blur-3xl"></div>
        <div className="absolute -left-20 -bottom-20 w-64 h-64 bg-gradient-to-tr from-blue-500/10 to-indigo-500/20 rounded-full blur-3xl"></div>
        
        <div className="relative z-10">
          <div className="flex justify-between items-center mb-5">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-100 flex items-center">
              <FiUploadCloud className="mr-2 text-indigo-500" size={20} />
              <span>파일 업로드</span>
            </h2>
            <button 
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 bg-gray-100 dark:bg-gray-700 rounded-full p-2 hover:bg-gray-200 dark:hover:bg-gray-600 transition-all"
              aria-label="닫기"
              disabled={isUploading}
            >
              <FiX size={18} />
            </button>
          </div>
          
          {/* 카테고리 선택 섹션은 숨김 처리 */}
          {showCategories && (
            <div className="mb-5">
              <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">카테고리</label>
              <div className="grid grid-cols-1 gap-2">
                <button
                  type="button"
                  className="px-4 py-2.5 rounded-xl text-sm font-medium transition-all bg-gradient-to-r from-indigo-600 to-indigo-500 text-white shadow-md"
                  disabled={true}
                >
                  <div className="flex items-center justify-center">
                    <FiCheck size={16} className="mr-1.5" />
                    <span>메뉴얼</span>
                  </div>
                </button>
              </div>
            </div>
          )}
          
          <div className="mb-6">
            <label className="block text-sm font-medium mb-2 text-gray-700 dark:text-gray-300">파일 선택</label>
            <div 
              className={`drop-area rounded-xl p-6 text-center cursor-pointer transition-all
                ${dragActive 
                  ? 'border-2 border-dashed border-indigo-500 bg-indigo-50 dark:bg-indigo-900/20' 
                  : 'border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-700/50 hover:bg-gray-100 dark:hover:bg-gray-700'
                } ${isUploading ? 'opacity-75 pointer-events-none' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => !isUploading && fileInputRef.current?.click()}
            >
              {files.length > 0 ? (
                <div className="py-2 space-y-2">
                  <div className="flex justify-center items-center mb-2">
                    <span className="bg-indigo-100 dark:bg-indigo-900/30 text-indigo-800 dark:text-indigo-300 text-xs font-semibold py-1 px-3 rounded-full">
                      {files.length}개 파일 선택됨
                    </span>
                  </div>
                  <div className="max-h-40 overflow-y-auto custom-scrollbar">
                    {files.map((file, index) => (
                      <div key={index} className="flex items-center justify-between bg-white dark:bg-gray-700 rounded-lg p-2 mb-1.5 shadow-sm group hover:bg-gray-50 dark:hover:bg-gray-650 transition-colors">
                        <div className="flex items-center">
                          <div className="w-7 h-7 rounded-md bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center mr-2 flex-shrink-0">
                            <FiFile className="text-indigo-600 dark:text-indigo-400" size={14} />
                          </div>
                          <span className="text-gray-800 dark:text-gray-200 text-xs truncate max-w-[160px] font-medium">
                            {file.name}
                          </span>
                        </div>
                        <div className="flex items-center">
                          <span className="text-xs text-gray-500 dark:text-gray-400 mr-2">
                            {formatFileSize(file.size)}
                          </span>
                          {!isUploading && (
                            <button 
                              onClick={(e) => {
                                e.stopPropagation();
                                removeFile(index);
                              }} 
                              className="opacity-0 group-hover:opacity-100 p-1.5 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-all"
                            >
                              <FiX size={14} className="text-gray-500 dark:text-gray-400" />
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-gray-500 dark:text-gray-400 py-6">
                  <div className="mx-auto w-14 h-14 rounded-full bg-indigo-100 dark:bg-indigo-900/20 flex items-center justify-center mb-3 shadow-glow-sm">
                    <FiUploadCloud className="text-indigo-500" size={28} />
                  </div>
                  <p className="text-sm font-medium mb-1">파일을 여기에 드래그하세요</p>
                  <p className="text-xs text-gray-500 dark:text-gray-500">또는 클릭하여 여러 파일 선택</p>
                </div>
              )}
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
                multiple
                disabled={isUploading}
              />
            </div>
          </div>
          
          {/* 업로드 진행 상태 표시 */}
          {isUploading && (
            <div className="mb-4">
              <div className="h-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-indigo-600 to-blue-500 rounded-full transition-all duration-300 animate-pulse"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <p className="text-xs text-center mt-2 text-gray-500 dark:text-gray-400">
                업로드 중... 파일 크기에 따라 다소 시간이 소요될 수 있습니다.
              </p>
            </div>
          )}
          
          {/* 파일별 처리 결과 표시 */}
          {fileResults.length > 0 && !isUploading && (
            <div className="mb-4 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
              <div className="max-h-48 overflow-y-auto custom-scrollbar">
                {fileResults.map((result, index) => {
                  const statusInfo = getStatusClasses(result.status);
                  return (
                    <div key={index} className="flex items-center p-2 border-b border-gray-200 dark:border-gray-700 last:border-0">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center mr-2 ${statusInfo.className}`}>
                        {statusInfo.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-gray-800 dark:text-gray-200 truncate">{result.filename}</p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">{result.message}</p>
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap ml-2">
                        {result.processing_time ? `${result.processing_time}초` : ''}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          
          {uploadStatus && !isUploading && (
            <div className={`mb-4 p-3 rounded-lg ${uploadStatus.includes('실패') || uploadStatus.includes('오류') ? 'bg-red-100 dark:bg-red-900/20 text-red-800 dark:text-red-200' : 'bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200'}`}>
              <p className="text-sm">{uploadStatus}</p>
            </div>
          )}
          
          <div className="flex justify-end space-x-3">
            <button
              onClick={onClose}
              className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 transition-all"
              disabled={isUploading}
            >
              취소
            </button>
            <button
              onClick={handleSubmit}
              disabled={files.length === 0 || isUploading}
              className={`px-4 py-2 rounded-lg transition-all flex items-center ${
                files.length === 0 || isUploading
                  ? 'bg-indigo-400 dark:bg-indigo-700 cursor-not-allowed text-white opacity-70'
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white'
              }`}
            >
              {isUploading ? (
                <>
                  <FiLoader className="animate-spin mr-2" size={16} />
                  <span>처리 중...</span>
                </>
              ) : (
                <>
                  <FiUploadCloud className="mr-2" size={16} />
                  <span>업로드</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  // ReactDOM.createPortal을 사용하여 모달을 렌더링
  const container = containerSelector ? document.querySelector(containerSelector) : document.body;
  return ReactDOM.createPortal(modalContent, container);
}

export default FileUpload;
