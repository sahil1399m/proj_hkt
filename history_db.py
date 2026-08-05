"""
history_db.py — Chat history persistence via Supabase PostgreSQL

Includes UI helper utilities for the sidebar:
  - session_display_meta()  → icon, relative time, truncated title
  - group_sessions_by_date() → groups sessions into Today / Yesterday / Earlier
"""
from __future__ import annotations

import logging
from typing import Any, Optional
from datetime import datetime, timezone, timedelta

from supabase import Client
from auth import get_supabase

logger = logging.getLogger(__name__)


# ── Types ──────────────────────────────────────────────────────────────────────

class ChatSession:
    __slots__ = ("id", "title", "created_at", "updated_at")

    def __init__(self, row: dict[str, Any]) -> None:
        self.id:         str = row["id"]
        self.title:      str = row.get("title", "New Chat")
        self.created_at: str = row.get("created_at", "")
        self.updated_at: str = row.get("updated_at", "")

    # ── Time helpers ──────────────────────────────────────────────────────

    def _updated_dt(self) -> Optional[datetime]:
        try:
            return datetime.fromisoformat(self.updated_at.replace("Z", "+00:00"))
        except Exception:
            return None

    def updated_label(self) -> str:
        """Human-readable relative time — e.g. 'just now', '3h ago', '2d ago'."""
        dt = self._updated_dt()
        if not dt:
            return ""
        diff = datetime.now(timezone.utc) - dt
        h = diff.total_seconds() / 3600
        if h < 1:         return "just now"
        if h < 24:        return f"{int(h)}h ago"
        if diff.days < 7: return f"{diff.days}d ago"
        return dt.strftime("%d %b")

    # ── Display helpers ───────────────────────────────────────────────────

    def display_title(self, max_chars: int = 36) -> str:
        """Title truncated to fit the sidebar, with ellipsis."""
        t = self.title.strip() or "New Chat"
        return (t[:max_chars] + "…") if len(t) > max_chars else t

    def display_meta(self) -> dict[str, str]:
        """
        Returns a dict ready for sidebar rendering:
          icon        – emoji representing the topic (heuristic)
          title       – truncated title
          time_label  – relative time string
          date_group  – 'Today' | 'Yesterday' | 'This Week' | 'Earlier'
        """
        return {
            "icon":       _topic_icon(self.title),
            "title":      self.display_title(),
            "time_label": self.updated_label(),
            "date_group": _date_group(self._updated_dt()),
        }


class ChatMessage:
    __slots__ = ("id", "session_id", "role", "content", "metadata", "created_at")

    def __init__(self, row: dict[str, Any]) -> None:
        self.id:         str  = row["id"]
        self.session_id: str  = row["session_id"]
        self.role:       str  = row["role"]
        self.content:    str  = row["content"]
        self.metadata:   dict = row.get("metadata") or {}
        self.created_at: str  = row.get("created_at", "")


# ── Private UI helpers ─────────────────────────────────────────────────────────

def _topic_icon(title: str) -> str:
    """Pick a representative emoji for the session based on keywords in the title."""
    t = title.lower()
    if any(k in t for k in ("fee", "fees", "tuition", "payment")):       return "💰"
    if any(k in t for k in ("scholar", "scholarship")):                   return "🎓"
    if any(k in t for k in ("seat", "matrix", "admission", "cap")):       return "📋"
    if any(k in t for k in ("circular", "gr", "government")):             return "📜"
    if any(k in t for k in ("college", "university", "institution")):     return "🏫"
    if any(k in t for k in ("date", "deadline", "schedule", "timetable")): return "📅"
    if any(k in t for k in ("document", "certificate", "proof")):         return "📄"
    if any(k in t for k in ("rule", "regulation", "act", "policy")):      return "⚖️"
    if any(k in t for k in ("result", "merit", "rank", "score")):         return "🏆"
    if any(k in t for k in ("new chat", "untitled")):                     return "💬"
    return "💬"


def _date_group(dt: Optional[datetime]) -> str:
    """Return one of: 'Today', 'Yesterday', 'This Week', 'Earlier'."""
    if not dt:
        return "Earlier"
    now  = datetime.now(timezone.utc)
    diff = now - dt
    if diff.days == 0:  return "Today"
    if diff.days == 1:  return "Yesterday"
    if diff.days < 7:   return "This Week"
    return "Earlier"


# ── Session CRUD ───────────────────────────────────────────────────────────────

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
        get_supabase().table("chat_sessions").delete() \
            .eq("id", session_id).eq("user_id", user_id).execute()
        logger.info("Deleted session %s", session_id)
        return True
    except Exception as exc:
        logger.error("delete_session error: %s", exc)
        return False


def rename_session(session_id: str, user_id: str, new_title: str) -> bool:
    try:
        get_supabase().table("chat_sessions") \
            .update({"title": new_title[:80]}) \
            .eq("id", session_id).eq("user_id", user_id).execute()
        return True
    except Exception as exc:
        logger.error("rename_session error: %s", exc)
        return False


# ── Message CRUD ───────────────────────────────────────────────────────────────

def save_message(
    session_id: str,
    role:       str,
    content:    str,
    metadata:   Optional[dict[str, Any]] = None,
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
    session_id:       str,
    user_query:       str,
    assistant_answer: str,
    metadata:         Optional[dict[str, Any]] = None,
) -> None:
    """Save a complete user → assistant turn in a single call."""
    save_message(session_id, "user",      user_query)
    save_message(session_id, "assistant", assistant_answer, metadata)


# ── UI grouping helper ─────────────────────────────────────────────────────────

def group_sessions_by_date(
    sessions: list[ChatSession],
) -> dict[str, list[ChatSession]]:
    """
    Groups a flat list of ChatSession objects into labelled buckets for the sidebar.

    Returns an ordered dict with keys: 'Today', 'Yesterday', 'This Week', 'Earlier'
    (only buckets that have at least one session are included).

    Usage in app.py sidebar:
        from history_db import get_sessions, group_sessions_by_date

        sessions = get_sessions(user_id)
        groups   = group_sessions_by_date(sessions)

        for group_label, group_sessions in groups.items():
            st.caption(group_label)
            for s in group_sessions:
                meta = s.display_meta()
                if st.button(f"{meta['icon']}  {meta['title']}", key=s.id):
                    switch_session(s.id)
    """
    ORDER  = ["Today", "Yesterday", "This Week", "Earlier"]
    groups: dict[str, list[ChatSession]] = {g: [] for g in ORDER}

    for s in sessions:
        groups[_date_group(s._updated_dt())].append(s)

    # Drop empty buckets, preserve ORDER
    return {k: v for k in ORDER for v in [groups[k]] if v}


# ── Private helpers ────────────────────────────────────────────────────────────

def _auto_title(session_id: str, first_message: str) -> None:
    """Set session title from the first user message if still 'New Chat'."""
    try:
        title = first_message.strip()[:60]
        get_supabase().table("chat_sessions") \
            .update({"title": title}) \
            .eq("id", session_id).eq("title", "New Chat").execute()
    except Exception as exc:
        logger.debug("auto_title skipped: %s", exc)


# ── Streamlit session-state helpers ───────────────────────────────────────────

def ensure_active_session(user_id: str) -> str:
    """
    Return the active session_id from st.session_state,
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
    Switch to a past session — loads its messages into st.session_state.
    Call this when the user clicks a session in the sidebar.
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
    Clear the current chat and create a fresh session.
    Called from the '+ New Chat' button in the sidebar.
    """
    import streamlit as st

    st.session_state["active_session_id"] = None
    st.session_state["messages"]          = []
    st.session_state.pop("chat_sessions_cache", None)
    ensure_active_session(user_id)
