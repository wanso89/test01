import Sidebar from './components/Sidebar'
import ChatContainer from './components/ChatContainer'

function App() {
  return (
    <div className="flex h-screen bg-gray-100 dark:bg-gray-900">
      <Sidebar />
      <main className="flex-1 flex flex-col">
        <header className="p-4 border-b bg-white dark:bg-gray-800">
          <h1 className="text-2xl font-bold text-gray-800 dark:text-white">테스트중~</h1>
        </header>
        <ChatContainer />
      </main>
    </div>
  )
}
export default App
