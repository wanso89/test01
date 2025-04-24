import { useState } from 'react'

function ChatInput({ onSendMessage, isLoading }) {
  const [message, setMessage] = useState('')
  
  const handleSubmit = (e) => {
    e.preventDefault()
    if (!message.trim() || isLoading) return
    
    onSendMessage(message)
    setMessage('')
  }
  
  return (
    <form onSubmit={handleSubmit} className="flex space-x-2">
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="메시지를 입력하세요..."
        className="flex-1 p-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        disabled={isLoading}
      />
      <button
        type="submit"
        disabled={!message.trim() || isLoading}
        className="px-4 py-2 bg-blue-600 text-white rounded-md disabled:bg-blue-300"
      >
        전송
      </button>
    </form>
  )
}

export default ChatInput
