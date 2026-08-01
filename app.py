"""
app.py — HTE Knowledge Assistant — Hackathon Edition
=====================================================
Sidebar dashboard · Live streaming answers · Real pipeline-stage spinner
Default query chips · Best-in-class dark theme
"""

from __future__ import annotations
import json
import logging
import os
import time
from typing import Any

import streamlit as st
import plotly.graph_objects as go

from auth import get_current_user, is_authenticated, render_auth_page, logout
from history_db import (
    ensure_active_session, get_sessions, save_turn,
    start_new_chat, switch_session, delete_session, ChatSession,
)
from crag import stream_crag_pipeline
from chroma_loader import ensure_chroma_downloaded

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="HTE Knowledge Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Init state ────────────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "messages":            [],
        "active_session_id":   None,
        "show_pipeline":       False,
        "translate_marathi":   False,
        "chat_sessions_cache": None,
        "last_result":         None,
        "pending_query":       "",
        "crawler_started":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── HuggingFace ChromaDB bootstrap (runs once per cold start) ────────────────
if "chroma_ready" not in st.session_state:
    with st.spinner("⏳ Loading knowledge base — this only happens on first launch…"):
        ok = ensure_chroma_downloaded()
    st.session_state["chroma_ready"] = ok
    if not ok:
        st.error(
            "❌ Could not load the knowledge base from HuggingFace. "
            "Check that **HF_DATASET_REPO** is set in Streamlit secrets."
        )
        st.stop()

# ── Full CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══════════════════════════════
   GLOBAL RESET
══════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: #09090b !important;
    color: #e4e4e7 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
/* Hide only the decoration bar and footer, NOT the header collapse button */
footer                                { display: none !important; }
[data-testid="stDecoration"]          { display: none !important; }
[data-testid="stToolbar"]             { display: none !important; }

/* Keep the header but make it invisible except for the sidebar toggle */
header[data-testid="stHeader"]        { background: transparent !important; border: none !important; }
header[data-testid="stHeader"] > *:not([data-testid="stSidebarCollapsedControl"]) { opacity: 0 !important; pointer-events: none !important; }

/* Sidebar collapse/expand button — always visible and styled */
[data-testid="stSidebarCollapsedControl"] {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 0 8px 8px 0 !important;
    top: 50% !important;
    color: #a1a1aa !important;
}
[data-testid="stSidebarCollapsedControl"]:hover {
    background: #27272a !important;
    border-color: #3f3f46 !important;
}

/* Sidebar nav collapse arrow (≥ Streamlit 1.28) */
[data-testid="collapsedControl"] {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 0 8px 8px 0 !important;
    color: #a1a1aa !important;
}
[data-testid="collapsedControl"]:hover {
    background: #27272a !important;
}

.block-container {
    max-width: 860px !important;
    padding: 2.5rem 2rem 8rem !important;
    margin: 0 auto !important;
}

/* Ensure main area doesn't go under sidebar */
[data-testid="stAppViewContainer"] > section:nth-child(2) {
    padding-left: 0 !important;
}

/* ══════════════════════════════
   SIDEBAR
══════════════════════════════ */
[data-testid="stSidebar"] {
    background: #0c0c10 !important;
    border-right: 1px solid #27272a !important;
    min-width: 260px !important;
    max-width: 300px !important;
    z-index: 999 !important;
}
[data-testid="stSidebar"][aria-expanded="true"]  { display: block !important; }
[data-testid="stSidebar"] > div:first-child       { padding: 0 !important; overflow-y: auto !important; }

/* Ensure toggles inside sidebar are visible */
[data-testid="stSidebar"] [data-testid="stToggle"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stToggle"] label {
    color: #a1a1aa !important;
    font-size: 13px !important;
    visibility: visible !important;
}
[data-testid="stSidebar"] [data-testid="stToggle"] input {
    visibility: visible !important;
    opacity: 1 !important;
}

.sb-brand {
    display: flex; align-items: center; gap: 12px;
    padding: 24px 20px 20px;
    border-bottom: 1px solid #18181b;
    margin-bottom: 4px;
}
.sb-logo {
    width: 38px; height: 38px; border-radius: 10px;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0;
    box-shadow: 0 2px 12px rgba(59,130,246,0.25);
}
.sb-name { font-size: 13.5px; font-weight: 600; color: #f4f4f5; }
.sb-sub  { font-size: 10.5px; color: #52525b; margin-top: 2px; }

.sb-section {
    padding: 16px 20px 0;
}
.sb-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #3f3f46; margin-bottom: 10px;
}

/* KB stats cards */
.sb-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 4px; }
.sb-stat {
    background: #18181b; border: 1px solid #27272a;
    border-radius: 8px; padding: 10px 12px; text-align: center;
}
.sb-stat-val { font-size: 20px; font-weight: 700; color: #60a5fa; font-family: 'JetBrains Mono', monospace; }
.sb-stat-key { font-size: 10px; color: #52525b; margin-top: 2px; }

/* Model pills */
.sb-pills { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 4px; }
.sb-pill {
    padding: 3px 8px; border-radius: 4px; font-size: 10px;
    font-weight: 600; border: 1px solid; white-space: nowrap;
}
.pill-ibm   { background:#00082a; color:#60a5fa; border-color:#1d3a6a; }
.pill-gem   { background:#071510; color:#34d399; border-color:#065f46; }
.pill-groq  { background:#140e00; color:#fbbf24; border-color:#5a4200; }
.pill-crag  { background:#160a28; color:#a78bfa; border-color:#4c1d95; }
.pill-chroma{ background:#111115; color:#94a3b8; border-color:#27272a; }

/* History items */
.hist-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 10px; border-radius: 7px;
    cursor: pointer; margin-bottom: 2px;
    transition: background 0.15s;
    font-size: 12.5px; color: #71717a;
    border: 1px solid transparent;
    position: relative;
}
.hist-item:hover { background: #18181b; border-color: #27272a; color: #a1a1aa; }
.hist-item.active {
    background: #1e2a3d; border-color: #2d4a70; color: #93c5fd;
}
.hist-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #3f3f46; flex-shrink: 0;
}
.hist-item.active .hist-dot { background: #3b82f6; }
.hist-text { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }

/* Sidebar button override */
[data-testid="stSidebar"] .stButton > button {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    color: #a1a1aa !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #27272a !important;
    border-color: #3f3f46 !important;
    color: #f4f4f5 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
}
[data-testid="stSidebar"] [data-testid="stToggle"] label {
    color: #a1a1aa !important;
    font-size: 13px !important;
}

/* ══════════════════════════════
   TOP BAR
══════════════════════════════ */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 16px;
    border-bottom: 1px solid #18181b;
    margin-bottom: 24px;
}
.topbar-title { font-size: 18px; font-weight: 700; color: #f4f4f5; }
.topbar-badge {
    background: #18181b; border: 1px solid #27272a;
    border-radius: 5px; padding: 3px 10px;
    font-size: 11px; color: #71717a; margin-left: 10px;
}
.topbar-user {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: #52525b;
}
.topbar-avatar {
    width: 28px; height: 28px; border-radius: 8px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; color: white;
}

/* ══════════════════════════════
   WELCOME + CHIPS
══════════════════════════════ */
.welcome-wrap { padding: 48px 0 32px; text-align: center; }
.welcome-icon-wrap {
    width: 68px; height: 68px; border-radius: 18px; margin: 0 auto 20px;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 30px;
    box-shadow: 0 8px 32px rgba(59,130,246,0.2);
}
.welcome-title {
    font-size: 26px; font-weight: 700; color: #f4f4f5;
    letter-spacing: -0.5px; margin-bottom: 10px;
}
.welcome-sub {
    font-size: 15px; color: #71717a;
    line-height: 1.6; margin-bottom: 36px;
}
.chips-label {
    font-size: 11px; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #3f3f46; margin-bottom: 16px;
}

/* chip buttons */
.stButton > button[kind="secondary"] {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    color: #a1a1aa !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    min-height: 60px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    line-height: 1.4 !important;
    transition: all 0.18s !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #27272a !important;
    border-color: #3f3f46 !important;
    color: #f4f4f5 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
}

/* ══════════════════════════════
   CONFIDENCE BANNERS
══════════════════════════════ */
.conf-banner {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; border-radius: 8px;
    font-size: 12.5px; font-weight: 500;
    margin-bottom: 14px; letter-spacing: 0.1px;
}
.conf-high   { background:#071510; border:1px solid #065f46; color:#34d399; }
.conf-mid    { background:#140e00; border:1px solid #7a5c00; color:#fbbf24; }
.conf-low    { background:#130508; border:1px solid #6a1010; color:#f87171; }
.conf-dot    { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.dot-green   { background:#34d399; box-shadow:0 0 6px rgba(52,211,153,0.5); }
.dot-yellow  { background:#fbbf24; box-shadow:0 0 6px rgba(251,191,36,0.5); }
.dot-red     { background:#f87171; box-shadow:0 0 6px rgba(248,113,113,0.5); }

/* ══════════════════════════════
   CHAT MESSAGES
══════════════════════════════ */
.msg-user-wrap {
    display: flex; justify-content: flex-end;
    margin: 28px 0 8px;
}
.msg-user {
    background: #1e293b;
    border: 1px solid #2d4a70;
    border-radius: 18px 18px 4px 18px;
    padding: 14px 20px;
    max-width: 82%;
    color: #e2e8f0;
    font-size: 15px; line-height: 1.65;
}

.msg-ai-header {
    display: flex; align-items: center; gap: 10px;
    margin: 4px 0 10px;
}
.msg-ai-avatar {
    width: 30px; height: 30px; border-radius: 8px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
}
.msg-ai-name { font-size: 13px; color: #71717a; font-weight: 600; }
.msg-ai-model {
    font-size: 10.5px; color: #3f3f46;
    font-family: 'JetBrains Mono', monospace;
    background: #18181b; border: 1px solid #27272a;
    border-radius: 4px; padding: 2px 7px;
}
.msg-ai-body {
    font-size: 15px; line-height: 1.78;
    color: #d4d4d8; padding-left: 40px;
}
.msg-ai-body h1,.msg-ai-body h2 {
    font-size: 17px; font-weight: 700;
    color: #f4f4f5; margin: 20px 0 10px;
    padding-bottom: 6px; border-bottom: 1px solid #27272a;
}
.msg-ai-body h3 { font-size: 15px; font-weight: 600; color: #e4e4e7; margin: 16px 0 8px; }
.msg-ai-body ul, .msg-ai-body ol { padding-left: 20px; margin: 8px 0; }
.msg-ai-body li { margin-bottom: 6px; color: #d4d4d8; }
.msg-ai-body strong { color: #f4f4f5; font-weight: 600; }
.msg-ai-body a { color: #60a5fa; text-decoration: none; }
.msg-ai-body a:hover { text-decoration: underline; }
.msg-ai-body code {
    background: #18181b; border: 1px solid #27272a;
    border-radius: 5px; padding: 2px 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; color: #fb923c;
}
/* Tables inside answer */
.msg-ai-body table {
    border-collapse: collapse; width: 100%;
    margin: 16px 0; font-size: 13.5px;
}
.msg-ai-body th {
    background: #18181b; color: #a1a1aa;
    padding: 10px 14px; text-align: left;
    font-weight: 600; font-size: 12px;
    letter-spacing: 0.5px; text-transform: uppercase;
    border: 1px solid #27272a;
}
.msg-ai-body td {
    padding: 9px 14px; border: 1px solid #1f1f22;
    color: #d4d4d8; vertical-align: top;
}
.msg-ai-body tr:nth-child(even) td { background: #0f0f12; }
.msg-ai-body tr:hover td { background: #18181b; }
.msg-ai-body blockquote {
    border-left: 3px solid #3b82f6;
    margin: 12px 0; padding: 10px 16px;
    background: #0f172a; border-radius: 0 8px 8px 0;
    color: #94a3b8; font-style: italic;
}

/* ══════════════════════════════
   SOURCES
══════════════════════════════ */
.src-wrap {
    margin: 18px 0 0 40px;
    background: #0c0c10;
    border: 1px solid #18181b;
    border-radius: 10px;
    padding: 16px;
}
.src-lbl {
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 1.2px; color: #3f3f46;
    font-weight: 700; margin-bottom: 10px;
}
.src-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 7px;
}
.src-d, .src-t, .src-w {
    display: flex; align-items: flex-start; gap: 9px;
    padding: 9px 12px; border-radius: 8px; border: 1px solid;
    text-decoration: none; transition: all 0.15s; min-width: 0;
}
.src-d  { background:#0f1729; border-color:#1d3a6a; }
.src-t  { background:#071510; border-color:#065f46; }
.src-w  { background:#111115; border-color:#27272a; cursor:pointer; }
.src-w:hover { background:#18181b; border-color:#3f3f46; }
.src-icon { font-size:15px; flex-shrink:0; margin-top:1px; }
.src-info { min-width:0; flex:1; }
.src-name {
    font-size:12px; font-weight:500;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.src-d .src-name { color:#60a5fa; }
.src-t .src-name { color:#34d399; }
.src-w .src-name { color:#a1a1aa; }
.src-meta { font-size:10.5px; color:#3f3f46; margin-top:2px; font-family:'JetBrains Mono',monospace; }

/* ══════════════════════════════
   LOADING ANIMATION
══════════════════════════════ */
.loading-wrap {
    background: #0c0c10;
    border: 1px solid #18181b;
    border-radius: 14px;
    padding: 16px 20px;
    margin: 8px 0 16px;
}
.loading-row {
    display: flex; align-items: center; gap: 12px;
}
.loading-avatar {
    width: 28px; height: 28px; border-radius: 7px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0;
}
.loading-msg { font-size: 13px; color: #d4d4d8; font-weight: 500; }
.spinner {
    width: 15px; height: 15px;
    border: 2px solid #27272a;
    border-top-color: #60a5fa;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
    animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.stream-cursor {
    display: inline-block;
    width: 8px; height: 16px;
    background: #60a5fa;
    margin-left: 2px;
    vertical-align: text-bottom;
    animation: blink 0.9s steps(1) infinite;
}
@keyframes blink { 50% { opacity: 0; } }

/* ══════════════════════════════
   INPUT
══════════════════════════════ */
[data-testid="stTextArea"] textarea {
    background: #18181b !important;
    border: 1px solid #27272a !important;
    border-radius: 12px !important;
    color: #f4f4f5 !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 16px !important;
    line-height: 1.55 !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
    resize: none !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.15), 0 4px 16px rgba(0,0,0,0.25) !important;
}
[data-testid="stTextArea"] textarea::placeholder { color: #3f3f46 !important; }

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    border: none !important; color: white !important;
    border-radius: 12px !important;
    min-height: 60px !important;
    font-weight: 700 !important; font-size: 15px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 16px rgba(59,130,246,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.35) !important;
}

[data-testid="stHorizontalBlock"] { align-items: flex-end !important; }

/* ══════════════════════════════
   EXPANDER
══════════════════════════════ */
[data-testid="stExpander"] {
    border: 1px solid #18181b !important;
    border-radius: 8px !important;
    background: #0c0c10 !important;
    margin: 6px 0 6px 40px !important;
}
[data-testid="stExpander"] summary {
    font-size: 12px !important; color: #52525b !important;
    padding: 8px 14px !important;
}

/* ══════════════════════════════
   MISC
══════════════════════════════ */
hr { border-color: #18181b !important; margin: 12px 0 !important; }
.marathi-box {
    background: #071510; border: 1px solid #065f46;
    border-left: 3px solid #34d399;
    border-radius: 0 10px 10px 10px;
    padding: 16px 20px; margin: 14px 0 0 40px;
    font-size: 15px; line-height: 1.9; color: #86efac;
}
</style>
""", unsafe_allow_html=True)

# ── Crawler ───────────────────────────────────────────────────────────────────
if not st.session_state["crawler_started"]:
    try:
        from crawler import start_scheduler
        start_scheduler()
    except Exception:
        pass
    st.session_state["crawler_started"] = True

# ── Auth guard ────────────────────────────────────────────────────────────────
if not is_authenticated():
    render_auth_page()
    st.stop()

user = get_current_user()
assert user is not None
ensure_active_session(user.user_id)

if st.session_state.get("chat_sessions_cache") is None:
    st.session_state["chat_sessions_cache"] = get_sessions(user.user_id, limit=40)

sessions: list[ChatSession] = st.session_state.get("chat_sessions_cache") or []

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-logo">🏛️</div>
      <div>
        <div class="sb-name">HTE Assistant</div>
        <div class="sb-sub">Maharashtra Higher &amp; Technical Ed.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">', unsafe_allow_html=True)
    if st.button("＋  New Chat", use_container_width=True, type="primary", key="new_chat_btn"):
        start_new_chat(user.user_id)
        st.session_state["chat_sessions_cache"] = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Knowledge Base</div>', unsafe_allow_html=True)
    kb_count = 0
    if st.session_state.get("chroma_ready"):
        try:
            import chromadb as _cc
            _col = _cc.PersistentClient(path=os.getenv("CHROMA_PATH", "./chroma_db")).get_collection(
                os.getenv("CHROMA_COLLECTION", "hte_documents"))
            kb_count = _col.count()
        except Exception:
            pass

    st.markdown(f"""
    <div class="sb-stats">
      <div class="sb-stat">
        <div class="sb-stat-val">{kb_count:,}</div>
        <div class="sb-stat-key">Chunks indexed</div>
      </div>
      <div class="sb-stat">
        <div class="sb-stat-val">{len(sessions)}</div>
        <div class="sb-stat-key">Your chats</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">AI Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-pills">
      <span class="sb-pill pill-ibm">IBM Granite</span>
      <span class="sb-pill pill-gem">Gemini</span>
      <span class="sb-pill pill-groq">Groq Llama</span>
      <span class="sb-pill pill-crag">CRAG</span>
      <span class="sb-pill pill-chroma">ChromaDB</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-section"><div class="sb-label">Settings</div>', unsafe_allow_html=True)
    st.session_state["translate_marathi"] = st.toggle(
        "🔤 Marathi Translation",
        value=st.session_state.get("translate_marathi", False),
        key="marathi_tog",
    )
    st.session_state["show_pipeline"] = st.toggle(
        "🔬 Pipeline Trace",
        value=st.session_state.get("show_pipeline", False),
        key="pipeline_tog",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    try:
        from crawler import trigger_crawl_now
        st.markdown('<div class="sb-section">', unsafe_allow_html=True)
        if st.button("🔄 Crawl Sources", use_container_width=True, key="crawl_btn"):
            with st.spinner("Crawling HTE sources…"):
                s = trigger_crawl_now()
            st.success(f"✓ {s.get('ingested',0)} docs · {s.get('total_chunks',0)} chunks")
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    except Exception:
        pass

    st.markdown('<div class="sb-section"><div class="sb-label">Chat History</div>', unsafe_allow_html=True)
    if not sessions:
        st.markdown('<div style="font-size:12px;color:#3f3f46;padding:4px 0;">No chats yet.</div>', unsafe_allow_html=True)
    else:
        for sess in sessions[:25]:
            active = sess.id == st.session_state.get("active_session_id")
            hc1, hc2 = st.columns([10, 1])
            with hc1:
                td  = sess.title[:46] + "…" if len(sess.title) > 46 else sess.title
                cls = "hist-item active" if active else "hist-item"
                st.markdown(
                    f'<div class="{cls}">'
                    f'<div class="hist-dot"></div>'
                    f'<span class="hist-text">{td}</span></div>',
                    unsafe_allow_html=True,
                )
                if st.button("", key=f"h_{sess.id}", use_container_width=True,
                             help=sess.title, type="secondary"):
                    switch_session(sess.id)
                    st.session_state["chat_sessions_cache"] = None
                    st.rerun()
            with hc2:
                if st.button("✕", key=f"hd_{sess.id}", help="Delete"):
                    delete_session(sess.id, user.user_id)
                    if sess.id == st.session_state.get("active_session_id"):
                        start_new_chat(user.user_id)
                    st.session_state["chat_sessions_cache"] = None
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    short_em = user.email.split("@")[0][:14]
    st.markdown(f"""
    <div class="topbar-user" style="padding:8px 20px 4px;">
      <div class="topbar-avatar">{short_em[0].upper()}</div>
      <span style="font-size:12px;color:#52525b;">{short_em}</span>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True, key="logout_btn"):
        logout()
        st.rerun()


# ══════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════

st.markdown(f"""
<div class="topbar">
  <div style="display:flex;align-items:center;gap:8px;">
    <span class="topbar-title">Ask HTE</span>
    <span class="topbar-badge">Maharashtra Education</span>
  </div>
  <div class="topbar-user">
    <div class="topbar-avatar">{user.email[0].upper()}</div>
    <span>{user.email.split('@')[0][:16]}</span>
  </div>
</div>
""", unsafe_allow_html=True)

DEFAULT_QUERIES = [
    {"icon": "💰", "text": "What is the fee structure for diploma engineering colleges in Maharashtra?"},
    {"icon": "🎓", "text": "How do I apply for EBC freeship scholarship?"},
    {"icon": "📋", "text": "What are the MHT-CET 2025 eligibility criteria?"},
    {"icon": "📅", "text": "When does CAP Round 1 registration close for 2025-26?"},
    {"icon": "🏫", "text": "What hostel facilities are available for girl students?"},
    {"icon": "📜", "text": "How does the college affiliation renewal process work?"},
    {"icon": "🔢", "text": "What is the fee for SC/ST students in government colleges?"},
    {"icon": "📊", "text": "Show me the engineering seat matrix for CAP 2025"},
]

messages = st.session_state.get("messages", [])

# ══════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════

def _conf_banner(label: str, score: float, model: str) -> str:
    pct       = int(score * 100)
    model_tag = (
        f'<span style="font-size:10px;color:#3f3f46;'
        f'font-family:\'JetBrains Mono\',monospace;margin-left:10px;">{model}</span>'
    )
    if label == "HIGH":
        return (
            f'<div class="conf-banner conf-high">'
            f'<span class="conf-dot dot-green"></span>'
            f'High confidence ({pct}%) — Grounded in official HTE documents{model_tag}</div>'
        )
    elif label == "MEDIUM":
        return (
            f'<div class="conf-banner conf-mid">'
            f'<span class="conf-dot dot-yellow"></span>'
            f'Medium confidence ({pct}%) — Official documents + verified web sources{model_tag}</div>'
        )
    else:
        return (
            f'<div class="conf-banner conf-low">'
            f'<span class="conf-dot dot-red"></span>'
            f'Low document match ({pct}%) — Sourced from verified government websites{model_tag}</div>'
        )


def _sources_html(metadata: dict[str, Any]) -> str:
    cits = metadata.get("citations", {})
    docs = cits.get("documents", [])
    webs = cits.get("web", [])
    if not docs and not webs:
        return ""

    parts = ['<div class="src-wrap">']

    if docs:
        parts.append('<div class="src-lbl">📄 Document Sources</div><div class="src-grid">')
        for c in docs:
            src   = c.get("source", "")
            pg    = c.get("page", "")
            et    = c.get("element_type", "text")
            cat   = c.get("category", "")
            score = c.get("score", 0.0)
            disp  = src[:30] + "…" if len(src) > 30 else src
            icon  = "📊" if et == "table" else "📄"
            cls   = "src-t" if et == "table" else "src-d"
            type_lbl = "TABLE" if et == "table" else (cat.upper() if cat else "DOC")
            score_lbl = f" · {score:.2f}" if score else ""
            parts.append(
                f'<div class="{cls}">'
                f'<div class="src-icon">{icon}</div>'
                f'<div class="src-info">'
                f'<div class="src-name" title="{src}">{disp}</div>'
                f'<div class="src-meta">p.{pg} · {type_lbl}{score_lbl}</div>'
                f'</div></div>'
            )
        parts.append('</div>')

    if webs:
        parts.append('<div class="src-lbl" style="margin-top:12px;">🌐 Web Sources</div><div class="src-grid">')
        for w in webs:
            url   = w.get("url", "#")
            title = w.get("title", "") or url
            score = w.get("score", 0.0)
            try:
                from urllib.parse import urlparse as _up
                domain = _up(url).netloc.replace("www.", "")
            except Exception:
                domain = url[:30]
            disp_t    = title[:38] + "…" if len(title) > 38 else title
            score_lbl = f" · {score:.2f}" if score else ""
            parts.append(
                f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="src-w">'
                f'<div class="src-icon">🔗</div>'
                f'<div class="src-info">'
                f'<div class="src-name" title="{title}">{disp_t}</div>'
                f'<div class="src-meta">{domain}{score_lbl}</div>'
                f'</div></a>'
            )
        parts.append('</div>')

    parts.append('</div>')
    return "".join(parts)


def _render_pipeline(metadata: dict[str, Any]) -> None:
    trace = metadata.get("pipeline_trace", {})
    if not trace:
        return

    def _kv(d: dict) -> str:
        bits = []
        for k, v in d.items():
            bits.append(
                f'<span style="color:#60a5fa">{v}</span>'
                f'<span style="color:#27272a"> {k}</span>'
            )
        return " · ".join(bits)

    with st.expander("🔬 Pipeline trace", expanded=False):
        stages = [
            ("Query analysis", "🔤", trace.get("query", {})),
            ("Retrieval",      "🧮", trace.get("retrieval", {})),
            ("CrossEncoder",   "⚖️",  trace.get("crossencoder", {})),
            ("CRAG branch",    "🔀", trace.get("corrective_branch", {})),
            ("Context",        "📦", trace.get("context", {})),
            ("LLM generation", "🤖", trace.get("llm_generation", {})),
        ]
        rows = []
        for name, icon, data in stages:
            if not data:
                continue
            rows.append(
                f'<div style="display:flex;align-items:flex-start;gap:10px;padding:7px 12px;'
                f'background:#111115;border:1px solid #18181b;border-radius:6px;margin-bottom:3px;'
                f'font-family:\'JetBrains Mono\',monospace;font-size:11px;">'
                f'<span style="color:#10b981;flex-shrink:0">{icon}</span>'
                f'<span style="color:#52525b;min-width:120px">{name}</span>'
                f'<span>{_kv(data)}</span></div>'
            )
        retrieval_s = metadata.get("retrieval_s")
        if retrieval_s:
            rows.append(
                f'<div style="color:#3f3f46;font-size:11px;text-align:right;margin-top:4px;'
                f'font-family:\'JetBrains Mono\',monospace;">⏱ {retrieval_s}s retrieval</div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)


def _render_chart(metadata: dict[str, Any]) -> None:
    docs = metadata.get("citations", {}).get("documents", [])
    if len(docs) < 2:
        return
    with st.expander("📊 Document relevance scores", expanded=False):
        names  = [f"DOC {d['index']}: {d['source'][:22]}…" if len(d['source']) > 22
                  else f"DOC {d['index']}: {d['source']}" for d in docs]
        scores = [d.get("score", 0) for d in docs]
        colors = ["#3b82f6" if s > -3 else "#8b5cf6" if s > -6.5 else "#ef4444" for s in scores]
        fig = go.Figure(go.Bar(
            x=scores, y=names, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}<br>Score: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#52525b", size=11, family="JetBrains Mono"),
            xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#18181b"),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=12, t=4, b=4),
            height=max(90, len(docs) * 32),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_message(role: str, content: str, metadata: dict[str, Any]) -> None:
    if role == "user":
        st.markdown(
            f'<div class="msg-user-wrap"><div class="msg-user">{content}</div></div>',
            unsafe_allow_html=True,
        )
        return

    conf_label = metadata.get("confidence_label", "")
    conf_score = metadata.get("confidence_score", 0.0)
    model_used = metadata.get("model_used", "")

    if conf_label:
        st.markdown(_conf_banner(conf_label, conf_score, model_used), unsafe_allow_html=True)

    model_html = (
        "" if not model_used
        else f'<span class="msg-ai-model">{model_used}</span>'
    )
    st.markdown(
        f'<div class="msg-ai-header">'
        f'<div class="msg-ai-avatar">🏛️</div>'
        f'<span class="msg-ai-name">HTE Assistant</span>'
        f'{model_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="msg-ai-body">', unsafe_allow_html=True)
    st.markdown(content)
    st.markdown("</div>", unsafe_allow_html=True)

    src = _sources_html(metadata)
    if src:
        st.markdown(src, unsafe_allow_html=True)

    if metadata.get("translation_applied") and metadata.get("translated_answer"):
        st.markdown(
            f'<div class="marathi-box">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:1px;color:#34d399;'
            f'text-transform:uppercase;margin-bottom:10px;">🔤 मराठी अनुवाद</div>'
            f'{metadata["translated_answer"]}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.get("show_pipeline", False):
        _render_pipeline(metadata)
        _render_chart(metadata)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# CHAT AREA
# ══════════════════════════════════════════════════════

if messages:
    for msg in messages:
        render_message(msg["role"], msg["content"], msg.get("metadata", {}))
else:
    st.markdown("""
    <div class="welcome-wrap">
      <div class="welcome-icon-wrap">🏛️</div>
      <div class="welcome-title">Ask anything about HTE Maharashtra</div>
      <div class="welcome-sub">
        Your AI assistant for engineering &amp; diploma admissions, fees,<br>
        scholarships, circulars, regulations, and more — grounded in official documents.
      </div>
      <div class="chips-label">Popular Questions</div>
    </div>
    """, unsafe_allow_html=True)

    col_pairs = [DEFAULT_QUERIES[i:i+2] for i in range(0, len(DEFAULT_QUERIES), 2)]
    for pair in col_pairs:
        cols = st.columns(2)
        for ci, q in enumerate(pair):
            with cols[ci]:
                label = f"{q['icon']}  {q['text'][:56]}{'…' if len(q['text'])>56 else ''}"
                if st.button(label, key=f"chip_{q['text'][:22]}", use_container_width=True):
                    st.session_state["pending_query"] = q["text"]
                    st.rerun()

# ══════════════════════════════════════════════════════
# INPUT BAR
# ══════════════════════════════════════════════════════

prefill = st.session_state.get("pending_query", "") or ""
if prefill:
    st.session_state["pending_query"] = ""

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

cq, cb = st.columns([8, 1])
with cq:
    query = st.text_area(
        "q",
        value=prefill,
        placeholder="Ask about admissions, fees, scholarships, circulars, regulations…",
        height=76,
        key="query_input",
        label_visibility="collapsed",
    )
with cb:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    ask_btn = st.button("Ask →", type="primary", use_container_width=True, key="ask_btn")


# ══════════════════════════════════════════════════════
# SUBMIT + LIVE STREAMING  ← THE FIXED SECTION
# ══════════════════════════════════════════════════════

META_PREFIX   = "<<<META>>>"
STATUS_PREFIX = "<<<STATUS>>>"

def _parse_stream(user_query: str, lang_out: str):
    """
    Consumes stream_crag_pipeline() and returns (answer: str, meta: dict).

    Root cause of the original bug:
      stream_crag_pipeline() yields the META line as:
          "\n<<<META>>>{...json...}"
      Because the LLM's last real token and the META sentinel are sometimes
      emitted in the same yield, a simple `startswith` check on the raw token
      missed it — the leading "\n" made the check fail, so the raw JSON string
      fell through into full_answer_parts and got rendered as text.

    Fix: strip each token before checking for control prefixes, AND guard
    against the META sentinel appearing mid-token (e.g. "last word\n<<<META>>>…").
    """
    full_answer_parts: list[str] = []
    meta: dict[str, Any]         = {}
    status_ph = st.empty()
    answer_ph = st.empty()
    started_streaming = False

    def _show_status(msg: str) -> None:
        status_ph.markdown(
            f'<div class="loading-wrap"><div class="loading-row">'
            f'<span class="spinner"></span>'
            f'<span class="loading-msg">{msg}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    def _show_streaming(text: str, done: bool = False) -> None:
        cursor = "" if done else '<span class="stream-cursor"></span>'
        answer_ph.markdown(
            f'<div class="msg-ai-header">'
            f'<div class="msg-ai-avatar">🏛️</div>'
            f'<span class="msg-ai-name">HTE Assistant</span></div>'
            f'<div class="msg-ai-body">{text}{cursor}</div>',
            unsafe_allow_html=True,
        )

    _show_status("Generating semantic embeddings…")

    try:
        for raw_token in stream_crag_pipeline(user_query, language_out=lang_out):
            # ── Control token detection ──────────────────────────────────
            # Strip leading whitespace/newlines before checking prefixes so
            # "\n<<<META>>>{...}" is caught correctly.
            stripped = raw_token.strip()

            # Handle the case where META is appended to the last answer token
            # e.g.  "some text\n<<<META>>>{...}"
            if META_PREFIX in raw_token:
                # Split on the prefix; anything before it is answer text
                before, _, after = raw_token.partition(META_PREFIX)
                before = before.strip()
                if before:
                    full_answer_parts.append(before)
                try:
                    meta = json.loads(after.strip())
                except Exception as exc:
                    logger.error("META parse error: %s | raw: %r", exc, after)
                continue  # done with this token

            if stripped.startswith(STATUS_PREFIX):
                try:
                    data = json.loads(stripped[len(STATUS_PREFIX):])
                    if not started_streaming:
                        _show_status(data.get("msg", "Working…"))
                except Exception:
                    pass
                continue

            # ── Regular answer token ─────────────────────────────────────
            if not started_streaming:
                started_streaming = True
                status_ph.empty()

            full_answer_parts.append(raw_token)
            _show_streaming("".join(full_answer_parts))

    except Exception as exc:
        status_ph.empty()
        err_msg = f"⚠️ Pipeline error: {exc}"
        full_answer_parts = [err_msg]
        _show_streaming(err_msg, done=True)

    answer = "".join(full_answer_parts).strip()
    _show_streaming(answer, done=True)
    answer_ph.empty()   # hand off to render_message for final formatted render
    status_ph.empty()

    return answer, meta


if ask_btn and query.strip():
    user_query = query.strip()

    # Show the user's message immediately
    st.markdown(
        f'<div class="msg-user-wrap"><div class="msg-user">{user_query}</div></div>',
        unsafe_allow_html=True,
    )

    lang_out = "marathi" if st.session_state.get("translate_marathi", False) else "english"

    # ── Stream + parse ────────────────────────────────────────────────────
    answer, meta = _parse_stream(user_query, lang_out)

    # ── Optional Marathi translation ──────────────────────────────────────
    translated_answer   = ""
    translation_applied = False
    if st.session_state.get("translate_marathi", False) and answer:
        try:
            from translator import translate_to_marathi as _tr
            translated_answer   = _tr(answer)
            translation_applied = bool(translated_answer and translated_answer != answer)
        except Exception:
            pass

    # ── Build metadata for render + history ──────────────────────────────
    msg_meta: dict[str, Any] = {
        "confidence_label":    meta.get("confidence_label", ""),
        "confidence_score":    meta.get("confidence_score", 0.0),
        "model_used":          meta.get("model_used", ""),
        "translation_applied": translation_applied,
        "translated_answer":   translated_answer,
        # ↓ This is what was broken — meta wasn't parsed so citations was empty
        "citations":           meta.get("citations", {"documents": [], "web": []}),
        "pipeline_trace":      meta.get("pipeline_trace", {}),
        "retrieval_s":         meta.get("retrieval_s"),
    }

    # ── Final render (confidence banner + formatted answer + sources) ─────
    render_message("assistant", answer, msg_meta)

    # ── Persist to session + DB ───────────────────────────────────────────
    st.session_state["messages"].append({"role": "user",      "content": user_query, "metadata": {}})
    st.session_state["messages"].append({"role": "assistant", "content": answer,     "metadata": msg_meta})

    sid = ensure_active_session(user.user_id)
    save_turn(session_id=sid, user_query=user_query,
              assistant_answer=answer, metadata=msg_meta)
    st.session_state["chat_sessions_cache"] = None

elif ask_btn and not query.strip():
    st.warning("Please enter a question before clicking Ask.")