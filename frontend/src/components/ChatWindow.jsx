import { useEffect, useRef, useState } from 'react'
import Message from './Message'

const SUGGESTIONS = [
  "What should I watch tonight?",
  "I want a dark psychological thriller with a twist",
  "What's my blind spot? What genres am I missing?",
  "What do people with similar taste to mine think about ...?",
]

export default function ChatWindow({ messages, loading, userId, onSend, onEnd }) {
  const [input, setInput] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSend = () => {
    if (!input.trim() || loading) return
    onSend(input.trim())
    setInput('')
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div style={styles.wrap}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.headerIcon}>🎬</span>
          <div>
            <div style={styles.headerTitle}>AI AGENT</div>
            <div style={styles.headerSub}>User {userId}</div>
          </div>
        </div>
        <button style={styles.endBtn} onClick={onEnd} title="End session">
          End session
        </button>
      </div>

      {/* Messages */}
      <div style={styles.messages}>
        {messages.map((msg) => (
          <Message key={msg.id} message={msg} />
        ))}
        {loading && (
          <div style={styles.thinking}>
            <div style={styles.thinkingDots}>
              <span />
              <span />
              <span />
            </div>
            <span style={styles.thinkingText}>Looking through the data…</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggestions (show only when few messages) */}
      {messages.length <= 1 && (
        <div style={styles.suggestions}>
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              style={styles.suggestion}
              onClick={() => { onSend(s) }}
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Input */}
      <div style={styles.inputRow}>
        <textarea
          style={styles.textarea}
          rows={1}
          placeholder="Ask about movies…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
        />
        <button
          style={{ ...styles.sendBtn, opacity: (!input.trim() || loading) ? 0.4 : 1 }}
          onClick={handleSend}
          disabled={!input.trim() || loading}
        >
          ↑
        </button>
      </div>
    </div>
  )
}

const styles = {
  wrap: {
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    maxWidth: '780px',
    margin: '0 auto',
    background: 'var(--bg)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '14px 20px',
    borderBottom: '1px solid var(--border)',
    background: 'var(--surface)',
    flexShrink: 0,
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: '10px' },
  headerIcon: { fontSize: '22px' },
  headerTitle: { fontWeight: 700, fontSize: '15px', color: '#fff' },
  headerSub: { fontSize: '12px', color: 'var(--text-muted)' },
  endBtn: {
    background: 'transparent',
    border: '1px solid var(--border)',
    color: 'var(--text-muted)',
    borderRadius: '8px',
    padding: '6px 12px',
    fontSize: '13px',
    transition: 'color 0.15s',
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '20px 16px',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
  },
  thinking: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '4px 0',
  },
  thinkingDots: {
    display: 'flex',
    gap: '4px',
  },
  thinkingText: { fontSize: '13px', color: 'var(--text-muted)', fontStyle: 'italic' },
  suggestions: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    padding: '0 16px 12px',
  },
  suggestion: {
    background: 'var(--surface-2)',
    border: '1px solid var(--border)',
    borderRadius: '20px',
    padding: '6px 12px',
    fontSize: '13px',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    transition: 'color 0.15s, border-color 0.15s',
  },
  inputRow: {
    display: 'flex',
    gap: '8px',
    padding: '12px 16px 16px',
    borderTop: '1px solid var(--border)',
    background: 'var(--surface)',
    flexShrink: 0,
    alignItems: 'flex-end',
  },
  textarea: {
    flex: 1,
    background: 'var(--surface-2)',
    border: '1px solid var(--border)',
    borderRadius: '12px',
    padding: '10px 14px',
    color: 'var(--text)',
    fontSize: '14px',
    resize: 'none',
    lineHeight: '1.5',
    maxHeight: '120px',
    overflow: 'auto',
  },
  sendBtn: {
    width: '40px',
    height: '40px',
    borderRadius: '10px',
    background: 'var(--accent)',
    color: '#fff',
    fontSize: '18px',
    fontWeight: 700,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
    transition: 'opacity 0.15s',
  },
}