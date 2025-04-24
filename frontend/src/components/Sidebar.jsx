import { useState } from 'react'
import FileUpload from './FileUpload'

function Sidebar({ categories, selectedCategory, onSelectCategory }) {
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false)

  return (
    <div className="flex flex-col h-full">
      {/* 헤더 */}
      <div className="p-4 border-b">
        <h1 className="text-xl font-bold">RAG 챗봇</h1>
      </div>

      {/* 카테고리 목록 */}
      <div className="flex-1 overflow-y-auto p-4">
        <h2 className="text-sm font-semibold text-gray-500 mb-2">카테고리</h2>
        <ul>
          {categories.map((category) => (
            <li key={category}>
              <button
                className={`w-full text-left p-2 rounded-md mb-1 ${
                  selectedCategory === category
                    ? 'bg-blue-100 text-blue-700'
                    : 'hover:bg-gray-100'
                }`}
                onClick={() => onSelectCategory(category)}
              >
                {category}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* 파일 업로드 버튼 */}
      <div className="p-4 border-t">
        <button
          className="w-full py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          onClick={() => setIsUploadModalOpen(true)}
        >
          파일 업로드
        </button>
      </div>

      {/* 파일 업로드 모달 */}
      {isUploadModalOpen && (
        <FileUpload 
          onClose={() => setIsUploadModalOpen(false)} 
          categories={categories}
        />
      )}
    </div>
  )
}

export default Sidebar
