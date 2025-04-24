import { useState, useEffect } from 'react'
import Sidebar from './components/Sidebar'
import ChatContainer from './components/ChatContainer'
import './App.css'

function App() {
  const [selectedCategory, setSelectedCategory] = useState('메뉴얼')
  const [categories, setCategories] = useState(['메뉴얼', '장애보고서'])
  const [isSidebarOpen, setIsSidebarOpen] = useState(true)

  // 카테고리 목록 가져오기
  useEffect(() => {
    const fetchCategories = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/categories')
        const data = await response.json()
        if (data.categories && data.categories.length > 0) {
          setCategories(data.categories)
        }
      } catch (error) {
        console.error('카테고리 로딩 실패:', error)
      }
    }

    fetchCategories()
  }, [])

  return (
    <div className="flex h-screen bg-gray-100">
      {/* 사이드바 토글 버튼 (모바일용) */}
      <button 
        className="fixed z-50 p-2 bg-blue-600 rounded-md text-white md:hidden left-4 top-4"
        onClick={() => setIsSidebarOpen(!isSidebarOpen)}
      >
        {isSidebarOpen ? '✕' : '☰'}
      </button>

      {/* 사이드바 */}
      <div className={`${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'} transition-transform duration-300 fixed md:static md:translate-x-0 z-40 w-64 h-full bg-white shadow-lg`}>
        <Sidebar 
          categories={categories} 
          selectedCategory={selectedCategory}
          onSelectCategory={setSelectedCategory}
        />
      </div>

      {/* 메인 컨텐츠 */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <ChatContainer selectedCategory={selectedCategory} />
      </div>
    </div>
  )
}

export default App
