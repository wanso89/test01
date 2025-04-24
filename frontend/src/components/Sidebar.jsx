function Sidebar() {
  return (
    <aside className="w-72 bg-white dark:bg-gray-800 border-r flex flex-col">
      <div className="p-4 border-b">
        <h2 className="font-bold text-lg text-gray-700 dark:text-white">카테고리</h2>
      </div>
      <nav className="flex-1 p-4 space-y-2">
        <button className="w-full text-left p-2 rounded hover:bg-blue-100 dark:hover:bg-gray-700 text-blue-700 dark:text-blue-300 font-semibold bg-blue-50 dark:bg-gray-700 mb-2">
          메뉴얼
        </button>
        <button className="w-full text-left p-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300">
          장애보고서
        </button>
      </nav>
      <div className="p-4 border-t">
        <button className="w-full py-2 bg-blue-600 text-white rounded hover:bg-blue-700">+ 새 대화</button>
      </div>
    </aside>
  )
}
export default Sidebar
