import ChatMessage from './ChatMessage'
import ChatInput from './ChatInput'
import { useRef, useEffect, useState } from 'react'

function ChatContainer() {
  const [messages, setMessages] = useState([
    // 예시 메시지
    { role: 'assistant', content: '안녕하세요! 무엇을 도와드릴까요?' }
  ])
  const messagesEndRef = useRef(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50 dark:bg-gray-900">
        {messages.map((msg, i) => (
          <ChatMessage key={i} message={msg} />
        ))}
        <div ref={messagesEndRef} />
      </div>
      <ChatInput onSend={msg => setMessages(m => [...m, { role: 'user', content: msg }])} />
    </div>
  )
}
export default ChatContainer
