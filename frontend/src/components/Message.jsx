import ReactMarkdown from 'react-markdown'

export default function Message({ message }) {
  const isUser = message.role === 'user'
  return (
    <div style={{ ...styles.wrap, justifyContent: isUser ? 'flex-end' : 'flex-start' }}>
      {!isUser && <div style={styles.avatar}>🤖</div>}
      <div style={{ ...styles.bubble, ...(isUser ? styles.userBubble : styles.botBubble) }}>
        {isUser ? (
          <span>{message.content}</span>
        ) : (
          <div className="message-content">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
      </div>
      {isUser && <div style={styles.avatar}>👤</div>}
    </div>
  )
}

const styles = {
  wrap: {
    display: 'flex',
    gap: '10px',
    alignItems: 'flex-start',
    marginBottom: '4px',
  },
  avatar: {
    width: '30px',
    height: '30px',
    borderRadius: '50%',
    background: 'var(--surface-2)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '15px',
    flexShrink: 0,
    marginTop: '2px',
  },
  bubble: {
    maxWidth: '75%',
    padding: '10px 14px',
    borderRadius: '14px',
    fontSize: '14px',
    lineHeight: '1.6',
  },
  userBubble: {
    background: 'var(--user-bubble)',
    borderBottomRightRadius: '4px',
    color: 'var(--text)',
  },
  botBubble: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderBottomLeftRadius: '4px',
    color: 'var(--text)',
  },
}