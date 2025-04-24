import Sidebar from './components/Sidebar'
import ChatContainer from './components/ChatContainer'

function App() {
  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900 transition-colors">
      <Sidebar />
      <main className="flex-1 flex flex-col">
        <header className="p-4 border-b bg-white dark:bg-gray-800 flex items-center justify-between">
          <h1 className="text-2xl font-bold text-gray-800 dark:text-white">RAG 챗봇</h1>
          {/* 다크모드 토글 버튼 예시 */}
          <button
            onClick={() => {
              document.documentElement.classList.toggle('dark')
            }}
            className="ml-2 px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-100"
          >🌙</button>
        </header>
        <ChatContainer />
      </main>
    </div>
  )
}
export default App
