import { useState } from 'react'

const SUGGESTED_USERS = [
  { id: 1, label: 'User 1', desc: 'Action/Comedy · 190 ratings' },
  { id: 15, label: 'User 15', desc: 'Sci-fi · 85 ratings' },
  { id: 30, label: 'User 30', desc: 'Sparse · 18 ratings' },
]

export default function UserIdInput({ onStart, loading }) {
  const [value, setValue] = useState('')
  const [err, setErr] = useState('')

  const handleSubmit = () => {
    const uid = parseInt(value.trim(), 10)
    if (!uid || uid < 1) {
      setErr('Please enter a valid user ID (1–610)')
      return
    }
    setErr('')
    onStart(uid)
  }

  const handleKey = (e) => {
    if (e.key === 'Enter') handleSubmit()
  }

  return (
    <div style={styles.wrap}>
      <div style={styles.card}>
        <div style={styles.icon}>🎬</div>
        <h1 style={styles.title}>TrustedAI</h1>
        <p style={styles.sub}>Movie recommendations backed by real data</p>

        <div style={styles.field}>
          <label style={styles.label}>Enter your user ID</label>
          <input
            style={styles.input}
            type="number"
            min="1"
            max="610"
            placeholder="e.g. 1"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKey}
            autoFocus
          />
          {err && <p style={styles.err}>{err}</p>}
        </div>

        <div style={styles.suggestions}>
          <span style={styles.suggestLabel}>Try these:</span>
          <div style={styles.pills}>
            {SUGGESTED_USERS.map((u) => (
              <button
                key={u.id}
                style={styles.pill}
                onClick={() => { setValue(String(u.id)); onStart(u.id) }}
              >
                <span style={styles.pillId}>{u.label}</span>
                <span style={styles.pillDesc}>{u.desc}</span>
              </button>
            ))}
          </div>
        </div>

        <button
          style={{ ...styles.btn, opacity: loading ? 0.6 : 1 }}
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? 'Starting…' : 'Start chatting →'}
        </button>
      </div>
    </div>
  )
}

const styles = {
  wrap: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '24px',
    background: 'var(--bg)',
  },
  card: {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: '20px',
    padding: '40px 36px',
    width: '100%',
    maxWidth: '420px',
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  icon: { fontSize: '36px', textAlign: 'center' },
  title: { fontSize: '26px', fontWeight: 700, textAlign: 'center', color: '#fff' },
  sub: { textAlign: 'center', color: 'var(--text-muted)', fontSize: '14px', marginTop: '-8px' },
  field: { display: 'flex', flexDirection: 'column', gap: '6px' },
  label: { fontSize: '13px', color: 'var(--text-muted)', fontWeight: 500 },
  input: {
    background: 'var(--surface-2)',
    border: '1px solid var(--border)',
    borderRadius: '10px',
    padding: '12px 14px',
    color: 'var(--text)',
    fontSize: '16px',
    transition: 'border-color 0.15s',
  },
  err: { color: '#f87171', fontSize: '13px' },
  suggestions: { display: 'flex', flexDirection: 'column', gap: '8px' },
  suggestLabel: { fontSize: '12px', color: 'var(--text-muted)' },
  pills: { display: 'flex', flexDirection: 'column', gap: '6px' },
  pill: {
    background: 'var(--surface-2)',
    border: '1px solid var(--border)',
    borderRadius: '8px',
    padding: '10px 14px',
    color: 'var(--text)',
    textAlign: 'left',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    transition: 'border-color 0.15s, background 0.15s',
    cursor: 'pointer',
  },
  pillId: { fontWeight: 600, fontSize: '14px' },
  pillDesc: { fontSize: '12px', color: 'var(--text-muted)' },
  btn: {
    background: 'var(--accent)',
    color: '#fff',
    borderRadius: '10px',
    padding: '13px',
    fontSize: '15px',
    fontWeight: 600,
    transition: 'opacity 0.15s',
  },
}