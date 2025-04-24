import { useState } from 'react'
function ChatInput({ onSend }) {
  const [msg, setMsg] = useState('')
  return (
    <form
      className="flex p-4 border-t bg-white dark:bg-gray-800"
      onSubmit={e => {
        e.preventDefault()
        if (msg.trim()) {
          onSend(msg)
          setMsg('')
        }
      }}>
      <input
        className="flex-1 p-2 rounded border focus:outline-none"
        placeholder="메시지를 입력하세요..."
        value={msg}
        onChange={e => setMsg(e.target.value)}
      />
      <button
        type="submit"
        className="ml-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
      >
        전송
      </button>
    </form>
  )
}
export default ChatInput
