"""
agent.py — LLaMA agent via Groq with tool calling.
"""
import json
import re
import os
from groq import Groq
from tools.search_plots import search_plots
from tools.collaborative import (
    get_cf_recommendations, get_similar_users_opinion,
    find_similar_users, get_top_rated_movies, get_general_movie_opinion
)
from tools.user_history import get_user_history, get_user_taste_summary
from tools.movie_details import get_movie_details

client = Groq()

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOOL_ROUNDS = 3
MAX_HISTORY_TURNS = 6  
# ── Max characters per tool result injected into context ──
MAX_TOOL_RESULT_CHARS = 1200

# ── System prompt cache: { user_id: (profile_hash, prompt_str) } ──
_prompt_cache: dict[int, tuple[int, str]] = {}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_plots",
            "description": (
                "Semantic search over movie plot summaries. "
                "Use when user describes a vibe, theme, or genre — "
                "e.g. 'scary horror movie', 'feel-good road trip', 'dark thriller with a twist'. "
                "Works for BOTH new and existing users."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language description of desired movie"},
                    "n_results": {"type": "integer", "description": "Number of results (default 5, max 6)"},
                    "genre_filter": {"type": "string", "description": "Optional genre filter (e.g. 'Horror', 'Comedy')"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cf_recommendations",
            "description": (
                "Personalized recommendations using collaborative filtering. "
                "For existing users: finds similar users and surfaces movies they loved. "
                "For NEW users with no history: automatically falls back to globally top-rated movies. "
                "Always safe to call regardless of user history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "top_k_movies": {"type": "integer", "description": "Number of movies (default 6, max 8)"},
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_rated_movies",
            "description": (
                "Get globally top-rated movies by community score. "
                "Use for new users, or when user asks for popular/well-rated movies. "
                "Supports optional genre filtering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_ratings": {"type": "integer", "description": "Minimum number of ratings required (default 20)"},
                    "top_k": {"type": "integer", "description": "How many to return (default 6, max 8)"},
                    "genre_filter": {"type": "string", "description": "Optional genre (e.g. 'Horror', 'Drama', 'Action')"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_general_movie_opinion",
            "description": (
                "Get community opinion on a specific movie: avg rating, distribution, % positive, tags. "
                "Use when user asks 'what do people think about X?', 'is X good?', 'how was X received?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {"type": "integer", "description": "Movie ID"},
                },
                "required": ["movie_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_users_opinion",
            "description": (
                "What do users with similar taste think about a specific movie? "
                "Use only when user asks 'what do people LIKE ME think of X?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "movie_id": {"type": "integer"},
                },
                "required": ["user_id", "movie_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_history",
            "description": (
                "Get a user's rating history, genre profile, and blind spots. "
                "Use to understand taste or explain why they'd like a specific movie."
            ),
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "integer"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": (
                "Get full details for a movie: plot, genres, rating stats, tags. "
                "Use to look up a movie by title (fuzzy match) and get its ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {"type": "integer", "description": "Movie ID (use if known)"},
                    "title_query": {"type": "string", "description": "Movie title to search (fuzzy match)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_users",
            "description": "Find the most similar users to this user based on rating history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer"},
                    "top_k": {"type": "integer"},
                },
                "required": ["user_id"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "search_plots": search_plots,
    "get_cf_recommendations": get_cf_recommendations,
    "get_similar_users_opinion": get_similar_users_opinion,
    "get_user_history": get_user_history,
    "get_movie_details": get_movie_details,
    "find_similar_users": find_similar_users,
    "get_top_rated_movies": get_top_rated_movies,
    "get_general_movie_opinion": get_general_movie_opinion,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _trim_tool_result(raw: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    """
    Truncate tool JSON to max_chars so a large payload (e.g. full rating history)
    doesn't blow up the context.  Appends a notice when truncated so the model
    knows data was cut.
    """
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + f'\n... [truncated, {len(raw) - max_chars} chars omitted]'


def _trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """
    Keep only the last max_turns user/assistant pairs.
    Tool messages interleaved inside a turn are kept together with their assistant
    message so the context stays valid for the Groq API.

    Strategy: walk backwards collecting complete "turns" (assistant + its tool
    results, then the preceding user message) until we have max_turns pairs.
    """
    if not history:
        return history

    # Collect indices of user messages (each marks the start of a turn)
    user_indices = [i for i, m in enumerate(history) if m["role"] == "user"]

    if len(user_indices) <= max_turns:
        return history  # Already within limit

    # Keep only the last max_turns user messages and everything after the oldest kept
    cutoff = user_indices[-max_turns]
    return history[cutoff:]


# Schema for each tool's integer parameters — used to coerce string→int
# when a model passes "20" instead of 20.
_INTEGER_PARAMS: dict[str, set[str]] = {
    "search_plots":             {"n_results"},
    "get_cf_recommendations":   {"user_id", "top_k_movies"},
    "get_top_rated_movies":     {"min_ratings", "top_k"},
    "get_general_movie_opinion":{"movie_id"},
    "get_similar_users_opinion":{"user_id", "movie_id"},
    "get_user_history":         {"user_id"},
    "get_movie_details":        {"movie_id"},
    "find_similar_users":       {"user_id", "top_k"},
}


def _coerce_args(name: str, arguments: dict) -> dict:
    """
    Coerce any integer parameters that arrived as strings.
    Some models (including newer Llama variants on Groq) emit numeric args
    as JSON strings ("20" instead of 20), which causes Groq schema validation
    to reject the request with tool_use_failed before our code even runs.
    """
    int_keys = _INTEGER_PARAMS.get(name, set())
    coerced = {}
    for k, v in arguments.items():
        if k in int_keys and isinstance(v, str):
            try:
                coerced[k] = int(v)
            except ValueError:
                coerced[k] = v  # leave as-is; let the function raise a useful error
        else:
            coerced[k] = v
    return coerced


def execute_tool(name: str, arguments: dict) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**_coerce_args(name, arguments))
        raw = json.dumps(result, ensure_ascii=False, default=str)
        return _trim_tool_result(raw)
    except Exception as e:
        return json.dumps({"error": str(e)})


def parse_legacy_tool_calls(content: str) -> list[dict] | None:
    """
    Some LLaMA versions emit tool calls as plain text in the format:
      <function=tool_name {"arg": "val"}></function>
    This parser extracts and normalises them into the standard format.
    Returns None if no legacy tool calls found.
    """
    pattern = r'<function=(\w+)\s*(\{.*?\})\s*></function>'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None

    calls = []
    for i, (name, args_str) in enumerate(matches):
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}
        calls.append({
            "id": f"legacy_call_{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return calls if calls else None


def is_new_user(user_id: int) -> bool:
    from data.loader import get_rating_matrix
    matrix = get_rating_matrix()
    return user_id not in matrix.index


def build_system_prompt(user_id: int, user_profile: dict | None = None) -> str:
    """
    Build (or return cached) system prompt for this user.

    Cache key = (user_id, hash of learned_preferences).
    The prompt is only rebuilt when the profile actually changes, avoiding
    repeated calls to get_user_taste_summary() on every request.
    """
    prefs = user_profile.get("learned_preferences") if user_profile else None
    cache_key = hash(json.dumps(prefs, sort_keys=True)) if prefs else 0

    if user_id in _prompt_cache:
        cached_key, cached_prompt = _prompt_cache[user_id]
        if cached_key == cache_key:
            return cached_prompt

    new_user = is_new_user(user_id)

    if new_user:
        user_context = (
            f"User {user_id} is a BRAND NEW USER with no rating history. "
            "Use get_top_rated_movies (with genre_filter if they specify a genre) "
            "and search_plots to answer their requests. "
            "Tell them you don't know their taste yet and encourage them to share preferences."
        )
    else:
        taste_summary = get_user_taste_summary(user_id)
        user_context = f"User {user_id} has rating history. {taste_summary}"

    profile_section = ""
    if prefs:
        profile_section = f"\n\nLearned from previous sessions: {json.dumps(prefs)}"

    prompt = f"""You are a knowledgeable, enthusiastic movie companion — like a friend who has seen everything and genuinely loves talking about films. You have access to a real dataset of 5,135 movies and 74,000 ratings.

{user_context}{profile_section}

## Tool selection guide
- Genre/theme/vibe request → search_plots (always works, any user)
- "What should I watch?" → get_cf_recommendations
- "What do people think about X?" → get_movie_details (find ID) then get_general_movie_opinion
- "What do people LIKE ME think of X?" → get_movie_details then get_similar_users_opinion
- "Why would I like X?" → get_user_history + get_movie_details
- "Popular / top rated?" → get_top_rated_movies
- New user → get_top_rated_movies + search_plots

## How to respond
Always call tools first — never invent ratings or plots. Then craft a response that feels like natural conversation, not a data report.

**Weave numbers into prose, don't list them.** Instead of:
  ❌ "Average: 3.96/5, 141 ratings, 85.8% positive, distribution: 1×0.5, 1×1.0..."
  ✅ "People really enjoy it — it holds a 3.96/5 from over 140 ratings, and nearly 86% of viewers gave it a thumbs up."

**Lead with the takeaway, support with data.** What's the headline? Is it beloved, divisive, a hidden gem, overrated? Say that first, then back it up with a number or two.

**For opinion questions** ("what do people think of X?"): give a verdict + one or two data points + flavour from the tags or plot if relevant. Skip the full rating distribution unless it reveals something interesting (e.g. very polarising).

**For recommendations**: briefly say why this movie fits, not just what it's about. Connect it to what you know about the user's taste when possible.

**For "why would I like X?"**: reason out loud — reference specific genres or movies from their history that overlap.

**For blind spots / taste analysis**: be conversational and a little playful. "You've basically ignored Horror entirely — which is either very wise or a big gap depending on who you ask."

Keep responses concise. 2–4 sentences for simple opinion questions, a short paragraph per recommendation."""

    _prompt_cache[user_id] = (cache_key, prompt)
    return prompt


def run_agent(
    user_id: int,
    message: str,
    conversation_history: list[dict],
    user_profile: dict | None = None,
) -> tuple[str, list[dict]]:
    system_prompt = build_system_prompt(user_id, user_profile)
    conversation_history.append({"role": "user", "content": message})

    # Apply sliding window BEFORE building the messages list sent to the API.
    # The full history is still returned to the caller so the session isn't lost —
    # we only trim what we send to Groq.
    trimmed_history = _trim_history(conversation_history, MAX_HISTORY_TURNS)
    messages = [{"role": "system", "content": system_prompt}] + trimmed_history

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=1024,
        )

        msg = response.choices[0].message
        content = msg.content or ""

        # ── Check for standard tool calls ──────────────────────────────────
        tool_calls = msg.tool_calls

        # ── Fallback: parse legacy <function=...> format ───────────────────
        if not tool_calls and content:
            legacy = parse_legacy_tool_calls(content)
            if legacy:
                tool_calls = legacy
                content = re.sub(r'<function=\w+\s*\{.*?\}\s*></function>', '', content, flags=re.DOTALL).strip()

        # ── No tool calls → final answer ───────────────────────────────────
        if not tool_calls:
            conversation_history.append({"role": "assistant", "content": content})
            return content, conversation_history

        # ── Execute tool calls ─────────────────────────────────────────────
        def get_attr(tc, key):
            if isinstance(tc, dict):
                return tc.get(key)
            return getattr(tc, key, None)

        def get_fn_attr(tc, key):
            fn = get_attr(tc, "function")
            if isinstance(fn, dict):
                return fn.get(key)
            return getattr(fn, key, None)

        assistant_msg = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": get_attr(tc, "id"),
                    "type": "function",
                    "function": {
                        "name": get_fn_attr(tc, "name"),
                        "arguments": get_fn_attr(tc, "arguments"),
                    },
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)

        for tc in tool_calls:
            tc_id = get_attr(tc, "id")
            name = get_fn_attr(tc, "name")
            raw_args = get_fn_attr(tc, "arguments")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}

            result = execute_tool(name, args)  # already trimmed inside execute_tool
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })

    fallback = "I looked through the data but couldn't find a confident answer. Could you rephrase?"
    conversation_history.append({"role": "assistant", "content": fallback})
    return fallback, conversation_history