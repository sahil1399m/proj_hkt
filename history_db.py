"""
history_db.py — Chat history persistence via Supabase PostgreSQL
"""
from __future__ import annotations

import logging
from typing import Any, Optional
from datetime import datetime, timezone

from supabase import Client
from auth import get_supabase

logger = logging.getLogger(__name__)


# ── Types ─────────────────────────────────────────────────────────────────────

class ChatSession:
    __slots__ = ("id", "title", "created_at", "updated_at")

    def __init__(self, row: dict[str, Any]) -> None:
        self.id: str        = row["id"]
        self.title: str     = row.get("title", "New Chat")
        self.created_at: str = row.get("created_at", "")
        self.updated_at: str = row.get("updated_at", "")

    def updated_label(self) -> str:
        try:
            dt   = datetime.fromisoformat(self.updated_at.replace("Z", "+00:00"))
            now  = datetime.now(timezone.utc)
            diff = now - dt
            h    = diff.total_seconds() / 3600
            if h < 1:      return "just now"
            if h < 24:     return f"{int(h)}h ago"
            if diff.days < 7: return f"{diff.days}d ago"
            return dt.strftime("%d %b")
        except Exception:
            return ""


class ChatMessage:
    __slots__ = ("id", "session_id", "role", "content", "metadata", "created_at")

    def __init__(self, row: dict[str, Any]) -> None:
        self.id: str         = row["id"]
        self.session_id: str = row["session_id"]
        self.role: str       = row["role"]
        self.content: str    = row["content"]
        self.metadata: dict  = row.get("metadata") or {}
        self.created_at: str = row.get("created_at", "")


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(user_id: str, title: str = "New Chat") -> Optional[str]:
    try:
        sb: Client = get_supabase()
        res = sb.table("chat_sessions").insert({"user_id": user_id, "title": title}).execute()
        if res.data:
            sid = res.data[0]["id"]
            logger.info("Created session %s for user %s", sid, user_id)
            return sid
        return None
    except Exception as exc:
        logger.error("create_session error: %s", exc)
        return None


def get_sessions(user_id: str, limit: int = 50) -> list[ChatSession]:
    try:
        sb: Client = get_supabase()
        res = (
            sb.table("chat_sessions")
            .select("id, title, created_at, updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return [ChatSession(row) for row in (res.data or [])]
    except Exception as exc:
        logger.error("get_sessions error: %s", exc)
        return []


def delete_session(session_id: str, user_id: str) -> bool:
    try:
        get_supabase().table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()
        logger.info("Deleted session %s", session_id)
        return True
    except Exception as exc:
        logger.error("delete_session error: %s", exc)
        return False


def rename_session(session_id: str, user_id: str, new_title: str) -> bool:
    try:
        get_supabase().table("chat_sessions").update({"title": new_title[:80]}).eq("id", session_id).eq("user_id", user_id).execute()
        return True
    except Exception as exc:
        logger.error("rename_session error: %s", exc)
        return False


# ── Message CRUD ──────────────────────────────────────────────────────────────

def save_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    try:
        sb: Client = get_supabase()
        res = (
            sb.table("chat_messages")
            .insert({
                "session_id": session_id,
                "role":       role,
                "content":    content,
                "metadata":   metadata or {},
            })
            .execute()
        )
        if res.data:
            if role == "user":
                _auto_title(session_id, content)
            return res.data[0]["id"]
        return None
    except Exception as exc:
        logger.error("save_message error: %s", exc)
        return None


def get_messages(session_id: str) -> list[ChatMessage]:
    try:
        sb: Client = get_supabase()
        res = (
            sb.table("chat_messages")
            .select("id, session_id, role, content, metadata, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .execute()
        )
        return [ChatMessage(row) for row in (res.data or [])]
    except Exception as exc:
        logger.error("get_messages error: %s", exc)
        return []


def save_turn(
    session_id: str,
    user_query: str,
    assistant_answer: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Save a full user→assistant turn in one call."""
    save_message(session_id, "user",      user_query)
    save_message(session_id, "assistant", assistant_answer, metadata)


# ── Private helpers ───────────────────────────────────────────────────────────

def _auto_title(session_id: str, first_message: str) -> None:
    """Set session title from first user message if still 'New Chat'."""
    try:
        title = first_message.strip()[:60]
        get_supabase().table("chat_sessions").update({"title": title}).eq("id", session_id).eq("title", "New Chat").execute()
    except Exception as exc:
        logger.debug("auto_title skipped: %s", exc)


# ── Streamlit session-state helpers (called from app.py) ─────────────────────

def ensure_active_session(user_id: str) -> str:
    """
    Return active session_id from st.session_state,
    or create a new one if none exists.
    """
    import streamlit as st

    if not st.session_state.get("active_session_id"):
        session_id = create_session(user_id)
        st.session_state["active_session_id"] = session_id
        st.session_state.pop("chat_sessions_cache", None)
    return st.session_state["active_session_id"]


def switch_session(session_id: str) -> None:
    """
    Switch to a past session — loads its messages into session_state.
    Called when user clicks a session in the sidebar.
    """
    import streamlit as st

    st.session_state["active_session_id"] = session_id
    messages = get_messages(session_id)
    st.session_state["messages"] = [
        {"role": m.role, "content": m.content, "metadata": m.metadata}
        for m in messages
    ]


def start_new_chat(user_id: str) -> None:
    """
    Clear current chat and create a fresh session.
    Called from the '+ New Chat' button.
    """
    import streamlit as st

    st.session_state["active_session_id"] = None
    st.session_state["messages"] = []
    st.session_state.pop("chat_sessions_cache", None)
    ensure_active_session(user_id)