import { useState, useEffect, useRef } from 'react'
import ChatMessage from './ChatMessage'
import ChatInput from './ChatInput'

function ChatContainer({ selectedCategory }) {
  const [messages, setMessages] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  // 메시지가 추가될 때마다 스크롤을 아래로 이동
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSendMessage = async (message) => {
    if (!message.trim()) return

    // 사용자 메시지 추가
    const userMessage = {
      role: 'user',
      content: message,
      category: selectedCategory
    }
    
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)
    
    try {
      // 백엔드 API 호출
      const response = await fetch('http://172.10.2.70:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: message,
          category: selectedCategory,
          history: messages.map(msg => ({
            role: msg.role,
            content: msg.content
          }))
        }),
      })
      
      if (!response.ok) {
        throw new Error('API 응답 오류')
      }
      
      const data = await response.json()
      
      const botResponse = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources || []
      }
      
      setMessages(prev => [...prev, botResponse])
    } catch (error) {
      console.error('메시지 전송 실패:', error)
      
      // 오류 메시지 추가
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '죄송합니다. 메시지 처리 중 오류가 발생했습니다.',
        error: true
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* 헤더 */}
      <div className="p-4 bg-white shadow-sm">
        <h2 className="text-lg font-semibold">
          {selectedCategory} 카테고리 대화
        </h2>
      </div>
      
      {/* 메시지 목록 */}
      <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-400">
            <p>새로운 대화를 시작하세요</p>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message, index) => (
              <ChatMessage key={index} message={message} />
            ))}
            {isLoading && (
              <div className="flex justify-center p-4">
                <div className="animate-pulse text-gray-500">응답 생성 중...</div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      
      {/* 입력 영역 */}
      <div className="p-4 bg-white border-t">
        <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
      </div>
    </div>
  )
}

export default ChatContainer
