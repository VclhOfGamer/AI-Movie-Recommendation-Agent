"""
main.py — FastAPI backend.

Endpoints:
  POST /chat           — send a message, get a reply
  POST /session/start  — start a session for a user
  POST /session/end    — end session, persist learned profile
  GET  /user/{user_id}/profile — fetch stored profile
  GET  /health         — sanity check

Token-efficiency changes vs original:
  - update_profile_from_session now uses Groq (llama-3.1-8b) instead of OpenAI gpt-4o-mini
  - History passed to summariser strips tool messages (role != user/assistant)
    and truncates each message to 200 chars to keep the summarisation prompt small
"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent
from tools.user_history import get_user_history

PROFILES_DIR = Path(__file__).parent / "profiles"
PROFILES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Movie Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id → {user_id, history, created_at}
sessions: dict[str, dict] = {}


# ── Pydantic models ─────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    user_id: int


class ChatRequest(BaseModel):
    session_id: str
    message: str


class EndSessionRequest(BaseModel):
    session_id: str
    summary: Optional[str] = None


# ── Helper: load/save user profiles ─────────────────────────────────────────

def load_profile(user_id: int) -> dict:
    path = PROFILES_DIR / f"user_{user_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_profile(user_id: int, profile: dict):
    path = PROFILES_DIR / f"user_{user_id}.json"
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False))


def _build_summary_convo(history: list[dict], max_msg_chars: int = 200, max_total_chars: int = 2000) -> str:
    """
    Build a compact conversation string for the summariser.
    - Skips tool messages (role == 'tool') — they're raw data, not conversation
    - Truncates each message to max_msg_chars
    - Stops adding messages once total exceeds max_total_chars
    """
    lines = []
    total = 0
    for m in history:
        if m.get("role") not in ("user", "assistant"):
            continue
        content = (m.get("content") or "")
        if not isinstance(content, str):
            continue
        snippet = content[:max_msg_chars]
        line = f"{m['role'].upper()}: {snippet}"
        total += len(line)
        if total > max_total_chars:
            break
        lines.append(line)
    return "\n".join(lines)


def update_profile_from_session(user_id: int, history: list[dict], session_summary: str | None):
    """
    After a session ends, extract learned preferences via Groq (free, no extra API key).
    Uses llama-3.1-8b-instant — fast and cheap for this simple extraction task.
    """
    from groq import Groq
    groq_client = Groq()  # reads GROQ_API_KEY from env

    if not history:
        return

    convo_text = _build_summary_convo(history)
    if not convo_text:
        return

    prompt = f"""Based on this movie recommendation conversation, extract key user preferences as a JSON object.

Conversation:
{convo_text}

Return ONLY a JSON object (no markdown, no explanation) with these fields (omit any you can't infer):
- liked_genres: list of genres user expressed interest in
- disliked_genres: list of genres user wants to avoid
- preferred_themes: list of themes/vibes they enjoy
- constraints: any constraints mentioned (e.g. "no animated movies")
- notes: any other relevant preference notes (1 sentence max)"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        raw = (response.choices[0].message.content or "{}").strip()
        raw = raw.lstrip("```json").lstrip("```").rstrip("```").strip()
        new_prefs = json.loads(raw)
    except Exception:
        new_prefs = {}

    profile = load_profile(user_id)
    profile["user_id"] = user_id
    profile["last_session"] = datetime.utcnow().isoformat()
    profile["session_count"] = profile.get("session_count", 0) + 1

    existing = profile.get("learned_preferences", {})
    for key, val in new_prefs.items():
        if isinstance(val, list):
            existing[key] = list(dict.fromkeys(existing.get(key, []) + val))
        else:
            existing[key] = val

    profile["learned_preferences"] = existing
    if session_summary:
        profile["last_session_summary"] = session_summary

    save_profile(user_id, profile)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/session/start")
def start_session(req: StartSessionRequest):
    session_id = str(uuid.uuid4())
    profile = load_profile(req.user_id)

    sessions[session_id] = {
        "user_id": req.user_id,
        "history": [],
        "created_at": datetime.utcnow().isoformat(),
        "profile": profile,
    }

    return {
        "session_id": session_id,
        "user_id": req.user_id,
        "returning_user": bool(profile),
        "profile": profile,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Call /session/start first.")

    user_id = session["user_id"]
    history = session["history"]
    profile = session.get("profile", {})

    reply, updated_history = run_agent(
        user_id=user_id,
        message=req.message,
        conversation_history=history,
        user_profile=profile,
    )

    session["history"] = updated_history

    return {
        "reply": reply,
        "session_id": req.session_id,
        "message_count": len(updated_history),
    }


@app.post("/session/end")
def end_session(req: EndSessionRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    user_id = session["user_id"]
    history = session["history"]

    update_profile_from_session(user_id, history, req.summary)
    del sessions[req.session_id]

    return {"status": "ok", "user_id": user_id, "messages_in_session": len(history)}


@app.get("/user/{user_id}/profile")
def get_profile(user_id: int):
    return load_profile(user_id)


@app.get("/user/{user_id}/history")
def get_history(user_id: int):
    return get_user_history(user_id)