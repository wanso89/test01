import { useState } from 'react'

function FileUpload({ onClose, categories }) {
  const [file, setFile] = useState(null)
  const [category, setCategory] = useState(categories[0] || '메뉴얼')
  const [isUploading, setIsUploading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) return
    
    setIsUploading(true)
    
    const formData = new FormData()
    formData.append('file', file)
    formData.append('category', category)
    
    try {
      const response = await fetch('http://172.10.2.70:8000/api/upload', {
        method: 'POST',
        body: formData,
      })
      
      const data = await response.json()
      if (data.status === 'success') {
        onClose()
        // 필요하다면 성공 메시지 표시
      }
    } catch (error) {
      console.error('업로드 실패:', error)
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-96 max-w-full">
        <h2 className="text-xl font-bold mb-4">파일 업로드</h2>
        
        <form onSubmit={handleSubmit}>
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
          
          <div className="flex justify-end space-x-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 border rounded-md"
            >
              취소
            </button>
            <button
              type="submit"
              disabled={!file || isUploading}
              className="px-4 py-2 bg-blue-600 text-white rounded-md disabled:bg-blue-300"
            >
              {isUploading ? '업로드 중...' : '업로드'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default FileUpload
