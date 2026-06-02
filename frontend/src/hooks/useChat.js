import { useState, useCallback, useRef } from 'react'

const API = '/api'

export function useChat() {
  const [messages, setMessages] = useState([])
  const [sessionId, setSessionId] = useState(null)
  const [userId, setUserId] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [profile, setProfile] = useState(null)

  const startSession = useCallback(async (uid) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API}/session/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: uid }),
      })
      if (!res.ok) throw new Error('Failed to start session')
      const data = await res.json()
      setSessionId(data.session_id)
      setUserId(uid)
      setProfile(data.profile || null)
      setMessages([])

      // Welcome message
      const welcome = data.returning_user
        ? `Welcome back! I remember you from last time. What are you in the mood for?`
        : `Hi! I'm your movie assistant. I have access to your rating history and a dataset of 5,135 movies. What would you like to watch?`
      setMessages([{ role: 'assistant', content: welcome, id: 'welcome' }])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const sendMessage = useCallback(async (text) => {
    if (!sessionId || !text.trim()) return

    const userMsg = { role: 'user', content: text, id: Date.now().toString() }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Request failed')
      }
      const data = await res.json()
      const botMsg = { role: 'assistant', content: data.reply, id: `bot-${Date.now()}` }
      setMessages((prev) => [...prev, botMsg])
    } catch (e) {
      setError(e.message)
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `Sorry, something went wrong: ${e.message}`, id: `err-${Date.now()}` },
      ])
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  const endSession = useCallback(async () => {
    if (!sessionId) return
    try {
      await fetch(`${API}/session/end`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      })
    } catch (_) {}
    setSessionId(null)
    setUserId(null)
    setMessages([])
    setProfile(null)
  }, [sessionId])

  return { messages, sessionId, userId, loading, error, profile, startSession, sendMessage, endSession }
}