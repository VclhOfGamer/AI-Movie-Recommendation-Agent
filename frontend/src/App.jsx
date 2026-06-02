import UserIdInput from './components/UserIdInput'
import ChatWindow from './components/ChatWindow'
import { useChat } from './hooks/useChat'

export default function App() {
  const { messages, sessionId, userId, loading, sendMessage, startSession, endSession } = useChat()

  if (!sessionId) {
    return <UserIdInput onStart={startSession} loading={loading} />
  }

  return (
    <ChatWindow
      messages={messages}
      loading={loading}
      userId={userId}
      onSend={sendMessage}
      onEnd={endSession}
    />
  )
}