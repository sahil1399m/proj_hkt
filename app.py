"""
app.py — HTE Knowledge Assistant
Fixed: STATUS token leak, confidence threshold, clean response rendering
"""
from __future__ import annotations
import json
import logging
import os
from typing import Any

import streamlit as st
import plotly.graph_objects as go

from auth import get_current_user, is_authenticated, render_auth_page, logout
from history_db import (
    ensure_active_session, get_sessions, save_turn,
    start_new_chat, switch_session, delete_session, ChatSession,
)
from chroma_loader import ensure_chroma_downloaded, get_chunk_count
from crag import stream_crag_pipeline, _reset_col
from translator import translate_to_marathi, translate_to_hindi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="HTE Knowledge Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def _init_state() -> None:
    defaults = {
        "messages":            [],
        "active_session_id":   None,
        "show_pipeline":       False,
        "translate_lang":      "none",
        "chat_sessions_cache": None,
        "last_result":         None,
        "pending_query":       "",
        "crawler_started":     False,
        "show_panel":          False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

if "chroma_ready" not in st.session_state:
    with st.spinner("⏳ Loading knowledge base from HuggingFace…"):
        ok = ensure_chroma_downloaded()
    if ok:
        _reset_col()
    st.session_state["chroma_ready"] = ok
    if not ok:
        st.error("❌ Could not load the knowledge base. See details below.")
        st.markdown("### Debug Info")
        st.code(f"""
HF_DATASET_REPO : {os.getenv("HF_DATASET_REPO", "NOT SET")}
HF_TOKEN set    : {bool(os.getenv("HF_TOKEN"))}
CHROMA_PATH     : {os.getenv("CHROMA_PATH", "NOT SET (will use /tmp/chroma_db)")}
CHROMA_COLLECTION: {os.getenv("CHROMA_COLLECTION", "hte_documents")}
        """)
        st.info("👆 Check your Streamlit Cloud secrets. Most common fix: set CHROMA_PATH = \"/tmp/chroma_db\"")
        st.stop()

# ══════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}

html,body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"]{
    background:#09090b!important;
    color:#e4e4e7!important;
    font-family:'Inter',-apple-system,sans-serif!important;
}

footer,[data-testid="stDecoration"],[data-testid="stToolbar"],
header[data-testid="stHeader"],[data-testid="stSidebar"]{
    display:none!important;
}

.block-container{
    max-width:900px!important;
    padding:1.5rem 1.5rem 8rem!important;
    margin:0 auto!important;
}

/* ── Buttons ── */
.stButton>button{
    background:#18181b!important;border:1px solid #27272a!important;
    color:#a1a1aa!important;border-radius:8px!important;
    font-size:12px!important;font-weight:500!important;transition:all 0.15s!important;
    white-space:nowrap!important;
}
.stButton>button:hover{
    background:#27272a!important;color:#f4f4f5!important;border-color:#3f3f46!important;
}
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#3b82f6 0%,#8b5cf6 100%)!important;
    border:none!important;color:white!important;border-radius:12px!important;
    min-height:60px!important;font-weight:700!important;font-size:15px!important;
}
.stButton>button[kind="primary"]:hover{
    opacity:0.92!important;transform:translateY(-1px)!important;
}
.stButton>button[kind="secondary"]{
    background:#18181b!important;border:1px solid #27272a!important;
    color:#a1a1aa!important;border-radius:10px!important;padding:12px 14px!important;
    min-height:60px!important;text-align:left!important;font-size:13px!important;
    font-weight:500!important;transition:all 0.18s!important;
}
.stButton>button[kind="secondary"]:hover{
    background:#27272a!important;color:#f4f4f5!important;
}

/* ── Panel ── */
.panel-box{
    background:#0c0c10;border:1px solid #27272a;
    border-radius:14px;padding:20px;margin-bottom:20px;
}
.panel-lbl{
    font-size:10px;font-weight:700;letter-spacing:1.2px;
    text-transform:uppercase;color:#3f3f46;margin:14px 0 8px;
}
.panel-lbl:first-child{margin-top:0;}
.pstat-row{display:flex;gap:10px;}
.pstat{
    background:#18181b;border:1px solid #27272a;
    border-radius:8px;padding:8px 14px;text-align:center;flex:1;
}
.pstat-val{font-size:18px;font-weight:700;color:#60a5fa;font-family:'JetBrains Mono',monospace;}
.pstat-key{font-size:10px;color:#52525b;margin-top:2px;}

/* ── AI Stack Pills ── */
.sb-pills{display:flex;flex-wrap:wrap;gap:5px;}
.sb-pill{padding:3px 8px;border-radius:4px;font-size:10px;font-weight:600;border:1px solid;white-space:nowrap;}
.pill-ibm{background:#00082a;color:#60a5fa;border-color:#1d3a6a;}
.pill-gem{background:#071510;color:#34d399;border-color:#065f46;}
.pill-groq{background:#140e00;color:#fbbf24;border-color:#5a4200;}
.pill-crag{background:#160a28;color:#a78bfa;border-color:#4c1d95;}
.pill-chroma{background:#111115;color:#94a3b8;border-color:#27272a;}

/* ── History ── */
.hist-item{
    display:flex;align-items:center;gap:8px;padding:7px 10px;
    border-radius:7px;margin-bottom:2px;font-size:12.5px;
    color:#71717a;border:1px solid transparent;
}
.hist-dot{width:6px;height:6px;border-radius:50%;background:#3f3f46;flex-shrink:0;}
.hist-item.active{background:#1e2a3d;border-color:#2d4a70;color:#93c5fd;}
.hist-item.active .hist-dot{background:#3b82f6;}
.hist-text{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;}

/* ── User message ── */
.msg-user-wrap{display:flex;justify-content:flex-end;margin:28px 0 8px;}
.msg-user{
    background:#1e293b;border:1px solid #2d4a70;
    border-radius:18px 18px 4px 18px;padding:14px 20px;
    max-width:82%;color:#e2e8f0;font-size:15px;line-height:1.65;
}

/* ── AI message header ── */
.msg-ai-header{display:flex;align-items:center;gap:10px;margin:4px 0 12px;}
.msg-ai-avatar{
    width:32px;height:32px;border-radius:8px;
    background:linear-gradient(135deg,#3b82f6,#8b5cf6);
    display:flex;align-items:center;justify-content:center;
    font-size:16px;flex-shrink:0;
}
.msg-ai-name{font-size:13px;color:#71717a;font-weight:600;}
.msg-ai-model{
    font-size:10.5px;color:#3f3f46;font-family:'JetBrains Mono',monospace;
    background:#18181b;border:1px solid #27272a;border-radius:4px;padding:2px 7px;
}

/* ── AI answer body ── */
.msg-ai-body{
    font-size:15px;line-height:1.85;color:#d4d4d8;
    padding-left:42px;
}
.msg-ai-body h1,
.msg-ai-body h2{
    font-size:18px;font-weight:700;color:#f4f4f5;
    margin:22px 0 10px;padding-bottom:8px;
    border-bottom:1px solid #27272a;
}
.msg-ai-body h3{font-size:15px;font-weight:600;color:#e4e4e7;margin:16px 0 8px;}
.msg-ai-body p{margin:8px 0;color:#d4d4d8;}
.msg-ai-body ul,.msg-ai-body ol{padding-left:22px;margin:10px 0;}
.msg-ai-body li{margin-bottom:7px;color:#d4d4d8;line-height:1.7;}
.msg-ai-body strong{color:#f4f4f5;font-weight:600;}
.msg-ai-body a{color:#60a5fa;text-decoration:none;}
.msg-ai-body a:hover{text-decoration:underline;}
.msg-ai-body code{
    background:#1e1e24;border:1px solid #27272a;border-radius:5px;
    padding:2px 7px;font-family:'JetBrains Mono',monospace;
    font-size:13px;color:#fb923c;
}
.msg-ai-body pre{
    background:#1e1e24;border:1px solid #27272a;border-radius:8px;
    padding:14px;margin:12px 0;overflow-x:auto;
}
.msg-ai-body pre code{border:none;background:transparent;padding:0;}

/* ── Tables in answer ── */
.msg-ai-body table{
    border-collapse:collapse;width:100%;
    margin:16px 0;font-size:13.5px;
    border-radius:8px;overflow:hidden;
}
.msg-ai-body th{
    background:#1e2a3d;color:#93c5fd;padding:11px 16px;
    text-align:left;font-weight:600;font-size:12px;
    letter-spacing:0.5px;text-transform:uppercase;
    border-bottom:2px solid #2d4a70;
}
.msg-ai-body td{
    padding:10px 16px;border-bottom:1px solid #1f1f22;
    color:#d4d4d8;vertical-align:top;
}
.msg-ai-body tr:nth-child(even) td{background:#0f0f14;}
.msg-ai-body tr:hover td{background:#18181b;}
.msg-ai-body blockquote{
    border-left:3px solid #3b82f6;margin:12px 0;padding:10px 16px;
    background:#0f172a;border-radius:0 8px 8px 0;color:#94a3b8;font-style:italic;
}

/* ── Confidence banners ── */
.conf-banner{
    display:flex;align-items:center;gap:10px;
    padding:11px 16px;border-radius:9px;
    font-size:13px;font-weight:500;margin-bottom:16px;
    line-height:1.4;
}
.conf-high{background:#071510;border:1px solid #065f46;color:#34d399;}
.conf-mid {background:#140e00;border:1px solid #7a5c00;color:#fbbf24;}
.conf-low {background:#130508;border:1px solid #6a1010;color:#f87171;}
.conf-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.dot-green {background:#34d399;box-shadow:0 0 8px rgba(52,211,153,0.6);}
.dot-yellow{background:#fbbf24;box-shadow:0 0 8px rgba(251,191,36,0.6);}
.dot-red   {background:#f87171;box-shadow:0 0 8px rgba(248,113,113,0.6);}
.conf-model{
    margin-left:auto;font-size:10px;color:#52525b;
    font-family:'JetBrains Mono',monospace;
    background:#18181b;border:1px solid #27272a;
    border-radius:4px;padding:2px 8px;white-space:nowrap;
}

/* ── Sources section ── */
.src-wrap{
    margin:20px 0 0 42px;
    background:#0c0c10;border:1px solid #1f1f22;
    border-radius:12px;padding:18px;
}
.src-header{
    display:flex;align-items:center;gap:8px;
    font-size:11px;font-weight:700;letter-spacing:1px;
    text-transform:uppercase;color:#52525b;margin-bottom:14px;
}
.src-header-line{flex:1;height:1px;background:#1f1f22;}
.src-section-lbl{
    font-size:10px;font-weight:600;letter-spacing:0.8px;
    text-transform:uppercase;color:#3f3f46;margin-bottom:10px;
}
.src-grid{
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(220px,1fr));
    gap:8px;margin-bottom:14px;
}
.src-d,.src-t,.src-w{
    display:flex;align-items:flex-start;gap:10px;
    padding:10px 14px;border-radius:9px;border:1px solid;
    text-decoration:none;transition:all 0.15s;min-width:0;
}
.src-d{background:#0d1526;border-color:#1d3a6a;}
.src-t{background:#071510;border-color:#065f46;}
.src-w{background:#111115;border-color:#27272a;cursor:pointer;}
.src-d:hover{background:#112040;border-color:#2d4a70;}
.src-t:hover{background:#0a1f10;border-color:#065f46;}
.src-w:hover{background:#18181b;border-color:#3f3f46;}
.src-icon{font-size:16px;flex-shrink:0;margin-top:1px;}
.src-info{min-width:0;flex:1;}
.src-name{
    font-size:12.5px;font-weight:500;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.src-d .src-name{color:#60a5fa;}
.src-t .src-name{color:#34d399;}
.src-w .src-name{color:#a1a1aa;}
.src-meta{
    font-size:10px;color:#3f3f46;margin-top:3px;
    font-family:'JetBrains Mono',monospace;
}
.src-score-high{color:#34d399!important;}
.src-score-mid {color:#fbbf24!important;}
.src-score-low {color:#f87171!important;}

/* ── Loading spinner ── */
.loading-wrap{
    background:#0c0c10;border:1px solid #18181b;
    border-radius:14px;padding:16px 20px;margin:8px 0 16px;
}
.loading-row{display:flex;align-items:center;gap:12px;}
.loading-msg{font-size:13px;color:#a1a1aa;font-weight:500;}
.spinner{
    width:16px;height:16px;border:2px solid #27272a;
    border-top-color:#60a5fa;border-radius:50%;
    display:inline-block;flex-shrink:0;
    animation:spin 0.7s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg);}}
.stream-cursor{
    display:inline-block;width:2px;height:18px;
    background:#60a5fa;margin-left:3px;
    vertical-align:text-bottom;
    animation:blink 0.9s steps(1) infinite;
}
@keyframes blink{50%{opacity:0;}}

/* ── Translation box ── */
.trans-box{
    background:#071510;border:1px solid #065f46;
    border-left:3px solid #34d399;border-radius:0 10px 10px 10px;
    padding:18px 20px;margin:16px 0 0 42px;
    font-size:15px;line-height:1.9;color:#86efac;
}
.trans-label{
    font-size:11px;font-weight:700;letter-spacing:1px;
    color:#34d399;text-transform:uppercase;margin-bottom:12px;
}

/* ── Input ── */
[data-testid="stTextArea"] textarea{
    background:#18181b!important;border:1px solid #27272a!important;
    border-radius:12px!important;color:#f4f4f5!important;
    font-size:15px!important;padding:16px!important;
    line-height:1.55!important;resize:none!important;
    transition:border-color 0.15s!important;
}
[data-testid="stTextArea"] textarea:focus{
    border-color:#3b82f6!important;
    box-shadow:0 0 0 2px rgba(59,130,246,0.15)!important;
}
[data-testid="stTextArea"] textarea::placeholder{color:#3f3f46!important;}

[data-testid="stHorizontalBlock"]{align-items:flex-end!important;}

/* ── Expander ── */
[data-testid="stExpander"]{
    border:1px solid #18181b!important;border-radius:8px!important;
    background:#0c0c10!important;margin:6px 0 6px 42px!important;
}
[data-testid="stExpander"] summary{
    font-size:12px!important;color:#52525b!important;padding:8px 14px!important;
}

/* ── Toggle ── */
[data-testid="stToggle"] label{color:#a1a1aa!important;font-size:13px!important;}

/* ── Misc ── */
hr{border-color:#18181b!important;margin:12px 0!important;}

.topbar-avatar{
    width:28px;height:28px;border-radius:8px;
    background:linear-gradient(135deg,#3b82f6,#8b5cf6);
    display:flex;align-items:center;justify-content:center;
    font-size:12px;font-weight:700;color:white;
}

/* ── Welcome screen ── */
.welcome-wrap{padding:48px 0 32px;text-align:center;}
.welcome-icon-wrap{
    width:72px;height:72px;border-radius:20px;margin:0 auto 20px;
    background:linear-gradient(135deg,#3b82f6 0%,#8b5cf6 100%);
    display:flex;align-items:center;justify-content:center;font-size:32px;
    box-shadow:0 8px 32px rgba(59,130,246,0.25);
}
.welcome-title{
    font-size:27px;font-weight:700;color:#f4f4f5;
    letter-spacing:-0.5px;margin-bottom:12px;
}
.welcome-sub{font-size:15px;color:#71717a;line-height:1.7;margin-bottom:36px;}
.chips-label{
    font-size:11px;font-weight:700;letter-spacing:1.2px;
    text-transform:uppercase;color:#3f3f46;margin-bottom:16px;
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

# ── Auth ──────────────────────────────────────────────────────────────────────
if not is_authenticated():
    render_auth_page()
    st.stop()

user = get_current_user()
assert user is not None
ensure_active_session(user.user_id)

if st.session_state.get("chat_sessions_cache") is None:
    st.session_state["chat_sessions_cache"] = get_sessions(user.user_id, limit=40)

sessions: list[ChatSession] = st.session_state.get("chat_sessions_cache") or []
kb_count = get_chunk_count() if st.session_state.get("chroma_ready") else 0
short_em = user.email.split("@")[0][:14]

# ══════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════
c_logo, c_gap, c_nc, c_lang, c_pp, c_menu, c_user = st.columns([3, 1, 1.2, 2.5, 1.2, 1.2, 1.5])

with c_logo:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;padding:10px 0 6px;">
      <span style="font-size:18px;font-weight:700;color:#f4f4f5;">🏛️ Ask HTE</span>
      <span style="background:#18181b;border:1px solid #27272a;border-radius:5px;
                   padding:3px 8px;font-size:11px;color:#71717a;">Maharashtra</span>
    </div>""", unsafe_allow_html=True)

with c_nc:
    if st.button("＋ New", key="tnc", use_container_width=True):
        start_new_chat(user.user_id)
        st.session_state["chat_sessions_cache"] = None
        st.rerun()

with c_lang:
    lang_choice = st.radio(
        "Lang",
        options=["none", "marathi", "hindi"],
        format_func=lambda x: {"none": "🌐 English", "marathi": "🔤 Marathi", "hindi": "🔤 Hindi"}[x],
        horizontal=True,
        label_visibility="collapsed",
        key="translate_lang",
    )

with c_pp:
    pp_on = st.session_state.get("show_pipeline", False)
    if st.button(f"🔬 {'ON' if pp_on else 'Trace'}", key="tpp",
                 use_container_width=True, help="Toggle Pipeline Trace"):
        st.session_state["show_pipeline"] = not pp_on
        st.rerun()

with c_menu:
    panel_open = st.session_state.get("show_panel", False)
    if st.button("✕ Close" if panel_open else "☰ Menu", key="tmenu",
                 use_container_width=True):
        st.session_state["show_panel"] = not panel_open
        st.rerun()

with c_user:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:6px;padding:10px 0 6px;justify-content:flex-end;">
      <div class="topbar-avatar">{short_em[0].upper()}</div>
      <span style="font-size:11px;color:#52525b;">{short_em}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# PANEL
# ══════════════════════════════════════════════════════
if st.session_state.get("show_panel", False):
    st.markdown('<div class="panel-box">', unsafe_allow_html=True)

    st.markdown('<div class="panel-lbl">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pstat-row">
      <div class="pstat"><div class="pstat-val">{kb_count:,}</div><div class="pstat-key">Chunks indexed</div></div>
      <div class="pstat"><div class="pstat-val">{len(sessions)}</div><div class="pstat-key">Your chats</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="panel-lbl">AI Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-pills">
      <span class="sb-pill pill-ibm">IBM Granite 4</span>
      <span class="sb-pill pill-gem">Gemini Embed</span>
      <span class="sb-pill pill-groq">Groq Fallback</span>
      <span class="sb-pill pill-crag">CRAG Pipeline</span>
      <span class="sb-pill pill-chroma">ChromaDB</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="panel-lbl">Settings</div>', unsafe_allow_html=True)
    new_pp = st.toggle("🔬 Pipeline Trace",
                       value=st.session_state.get("show_pipeline", False),
                       key="panel_pp")
    if new_pp != st.session_state.get("show_pipeline", False):
        st.session_state["show_pipeline"] = new_pp
        st.rerun()

    try:
        from crawler import trigger_crawl_now
        if st.button("🔄 Crawl Sources", use_container_width=True, key="panel_crawl"):
            with st.spinner("Crawling…"):
                s = trigger_crawl_now()
            st.success(f"✓ {s.get('ingested',0)} docs · {s.get('total_chunks',0)} chunks")
            st.rerun()
    except Exception:
        pass

    st.markdown('<div class="panel-lbl">Chat History</div>', unsafe_allow_html=True)
    if not sessions:
        st.markdown('<div style="font-size:12px;color:#3f3f46;padding:4px 0 8px;">No chats yet.</div>',
                    unsafe_allow_html=True)
    else:
        for sess in sessions[:20]:
            active = sess.id == st.session_state.get("active_session_id")
            hc1, hc2 = st.columns([10, 1])
            with hc1:
                td  = sess.title[:50] + "…" if len(sess.title) > 50 else sess.title
                cls = "hist-item active" if active else "hist-item"
                st.markdown(f'<div class="{cls}"><div class="hist-dot"></div>'
                            f'<span class="hist-text">{td}</span></div>',
                            unsafe_allow_html=True)
                if st.button("↩ Load", key=f"ph_{sess.id}", use_container_width=True):
                    switch_session(sess.id)
                    st.session_state["chat_sessions_cache"] = None
                    st.session_state["show_panel"] = False
                    st.rerun()
            with hc2:
                if st.button("✕", key=f"phd_{sess.id}"):
                    delete_session(sess.id, user.user_id)
                    if sess.id == st.session_state.get("active_session_id"):
                        start_new_chat(user.user_id)
                    st.session_state["chat_sessions_cache"] = None
                    st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    lo1, lo2 = st.columns([3, 1])
    with lo1:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:6px;padding:4px 0;">
          <div class="topbar-avatar">{short_em[0].upper()}</div>
          <span style="font-size:12px;color:#52525b;">{user.email}</span>
        </div>""", unsafe_allow_html=True)
    with lo2:
        if st.button("Sign out", key="panel_logout"):
            logout()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# RENDER HELPERS
# ══════════════════════════════════════════════════════

def _conf_banner(label: str, score: float, model: str) -> str:
    """
    FIX: Green if score > 20%, Yellow if 10-20%, Red if < 10%.
    Previously only showed green for HIGH label (which required logit >= -3.0).
    Now uses actual percentage for colour decision.
    """
    pct = int(score * 100)
    model_tag = f'<span class="conf-model">{model}</span>' if model else ""

    # Colour based on actual score percentage, not just label
    if pct > 20:
        cls  = "conf-high"
        dot  = "dot-green"
        text = f"High confidence ({pct}%) — Grounded in official HTE documents"
    elif pct > 10:
        cls  = "conf-mid"
        dot  = "dot-yellow"
        text = f"Medium confidence ({pct}%) — Official documents + verified web sources"
    else:
        cls  = "conf-low"
        dot  = "dot-red"
        text = f"Low document match ({pct}%) — Answer sourced from government websites"

    return (
        f'<div class="conf-banner {cls}">'
        f'<span class="conf-dot {dot}"></span>'
        f'<span>{text}</span>'
        f'{model_tag}'
        f'</div>'
    )


def _score_color(score: float) -> str:
    if score > -3:
        return "src-score-high"
    if score > -6.5:
        return "src-score-mid"
    return "src-score-low"


def _sources_html(metadata: dict[str, Any]) -> str:
    cits = metadata.get("citations", {})
    docs = cits.get("documents", [])
    webs = cits.get("web", [])
    if not docs and not webs:
        return ""

    parts = [
        '<div class="src-wrap">',
        '<div class="src-header">',
        '<span>Sources &amp; References</span>',
        f'<span style="font-size:10px;color:#3f3f46;font-family:\'JetBrains Mono\',monospace;">'
        f'{len(docs)} doc{"s" if len(docs)!=1 else ""}'
        f'{f" · {len(webs)} web" if webs else ""}</span>',
        '<span class="src-header-line"></span>',
        '</div>',
    ]

    if docs:
        parts.append('<div class="src-section-lbl">📄 Document Sources</div><div class="src-grid">')
        for c in docs:
            src   = c.get("source", "")
            pg    = c.get("page", "")
            et    = c.get("element_type", "text")
            cat   = c.get("category", "")
            score = c.get("score", 0.0)
            disp  = src[:32] + "…" if len(src) > 32 else src
            icon  = "📊" if et == "table" else "📄"
            cls   = "src-t" if et == "table" else "src-d"
            type_lbl  = "TABLE" if et == "table" else (cat.upper() if cat else "DOC")
            score_cls = _score_color(score)
            parts.append(
                f'<div class="{cls}">'
                f'<div class="src-icon">{icon}</div>'
                f'<div class="src-info">'
                f'<div class="src-name" title="{src}">{disp}</div>'
                f'<div class="src-meta">p.{pg} · {type_lbl} · '
                f'<span class="{score_cls}">{score:.3f}</span></div>'
                f'</div></div>'
            )
        parts.append('</div>')

    if webs:
        parts.append('<div class="src-section-lbl" style="margin-top:14px;">🌐 Web Sources</div><div class="src-grid">')
        for w in webs:
            url   = w.get("url", "#")
            title = w.get("title", "") or url
            score = w.get("score", 0.0)
            try:
                from urllib.parse import urlparse as _up
                domain = _up(url).netloc.replace("www.", "")
            except Exception:
                domain = url[:30]
            disp_t    = title[:40] + "…" if len(title) > 40 else title
            score_cls = _score_color(score)
            parts.append(
                f'<a href="{url}" target="_blank" rel="noopener noreferrer" class="src-w">'
                f'<div class="src-icon">🔗</div>'
                f'<div class="src-info">'
                f'<div class="src-name" title="{title}">{disp_t}</div>'
                f'<div class="src-meta">{domain} · '
                f'<span class="{score_cls}">{score:.3f}</span></div>'
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
        return " · ".join(
            f'<span style="color:#60a5fa">{v}</span>'
            f'<span style="color:#52525b"> {k}</span>'
            for k, v in d.items()
        )
    with st.expander("🔬 Pipeline trace", expanded=False):
        stages = [
            ("Query",        "🔤", trace.get("query", {})),
            ("Retrieval",    "🧮", trace.get("retrieval", {})),
            ("CrossEncoder", "⚖️",  trace.get("crossencoder", {})),
            ("CRAG branch",  "🔀", trace.get("corrective_branch", {})),
            ("Context",      "📦", trace.get("context", {})),
            ("LLM",          "🤖", trace.get("llm_generation", {})),
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
                f'<span style="color:#52525b;min-width:110px">{name}</span>'
                f'<span>{_kv(data)}</span></div>'
            )
        if metadata.get("retrieval_s"):
            rows.append(
                f'<div style="color:#3f3f46;font-size:11px;text-align:right;margin-top:4px;'
                f'font-family:\'JetBrains Mono\',monospace;">⏱ {metadata["retrieval_s"]}s retrieval</div>'
            )
        st.markdown("".join(rows), unsafe_allow_html=True)


def _render_chart(metadata: dict[str, Any]) -> None:
    docs = metadata.get("citations", {}).get("documents", [])
    if len(docs) < 2:
        return
    with st.expander("📊 Document relevance scores", expanded=False):
        names  = [f"DOC {d['index']}: {d['source'][:24]}…" if len(d['source']) > 24
                  else f"DOC {d['index']}: {d['source']}" for d in docs]
        scores = [d.get("score", 0) for d in docs]
        colors = ["#34d399" if s > -3 else "#fbbf24" if s > -6.5 else "#f87171" for s in scores]
        fig = go.Figure(go.Bar(
            x=scores, y=names, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="%{y}<br>Score: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#52525b", size=11, family="JetBrains Mono"),
            xaxis=dict(showgrid=False, zeroline=True, zerolinecolor="#27272a"),
            yaxis=dict(showgrid=False),
            margin=dict(l=0, r=12, t=4, b=4),
            height=max(90, len(docs) * 34),
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

    # Confidence banner
    if conf_label:
        st.markdown(_conf_banner(conf_label, conf_score, model_used), unsafe_allow_html=True)

    # AI header
    model_html = "" if not model_used else f'<span class="msg-ai-model">{model_used}</span>'
    st.markdown(
        f'<div class="msg-ai-header">'
        f'<div class="msg-ai-avatar">🏛️</div>'
        f'<span class="msg-ai-name">HTE Assistant</span>'
        f'{model_html}</div>',
        unsafe_allow_html=True,
    )

    # Answer body — use st.markdown for proper rendering
    st.markdown('<div class="msg-ai-body">', unsafe_allow_html=True)
    st.markdown(content)
    st.markdown("</div>", unsafe_allow_html=True)

    # Sources
    src = _sources_html(metadata)
    if src:
        st.markdown(src, unsafe_allow_html=True)

    # Translation
    if metadata.get("translation_applied") and metadata.get("translated_answer"):
        lang = metadata.get("lang_used", st.session_state.get("translate_lang", "none"))
        if lang == "hindi":
            heading = "🔤 हिंदी अनुवाद"
        elif lang == "marathi":
            heading = "🔤 मराठी अनुवाद"
        else:
            heading = "🔤 Translation"
        st.markdown(
            f'<div class="trans-box">'
            f'<div class="trans-label">{heading}</div>'
            f'{metadata["translated_answer"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Pipeline trace
    if st.session_state.get("show_pipeline", False):
        _render_pipeline(metadata)
        _render_chart(metadata)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# CHAT AREA
# ══════════════════════════════════════════════════════
messages = st.session_state.get("messages", [])

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
        scholarships, circulars, regulations — grounded in official documents.
      </div>
      <div class="chips-label">Popular Questions</div>
    </div>""", unsafe_allow_html=True)
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
        "q", value=prefill,
        placeholder="Ask about admissions, fees, scholarships, circulars, regulations…",
        height=76, key="query_input", label_visibility="collapsed",
    )
with cb:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    ask_btn = st.button("Ask →", type="primary", use_container_width=True, key="ask_btn")


# ══════════════════════════════════════════════════════
# STREAMING  — FIX: STATUS tokens stripped before rendering
# ══════════════════════════════════════════════════════
META_PREFIX   = "<<<META>>>"
STATUS_PREFIX = "<<<STATUS>>>"

def _parse_stream(user_query: str, lang_out: str):
    """
    KEY FIX: STATUS tokens were leaking into the answer because
    the word-chunked IBM output included STATUS text in the token stream.
    Solution: check EVERY token for STATUS/META prefix before appending.
    """
    full_answer_parts: list[str] = []
    meta: dict[str, Any] = {}
    status_ph     = st.empty()
    answer_ph     = st.empty()
    started       = False
    buffer        = ""   # accumulates partial tokens to catch split STATUS/META markers

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

            # ── Accumulate into buffer for reliable prefix detection ──────
            buffer += raw_token

            # ── META token handling ───────────────────────────────────────
            if META_PREFIX in buffer:
                before, _, after = buffer.partition(META_PREFIX)
                before = before.strip()
                if before:
                    full_answer_parts.append(before)
                try:
                    meta = json.loads(after.strip())
                except Exception as exc:
                    logger.error("META parse error: %s | raw: %r", exc, after)
                buffer = ""
                continue

            # ── STATUS token handling ─────────────────────────────────────
            # Only process STATUS if buffer looks complete (ends with "}")
            if STATUS_PREFIX in buffer:
                # Extract STATUS portion
                idx = buffer.index(STATUS_PREFIX)
                # Keep anything before STATUS as answer text
                before_status = buffer[:idx].strip()
                if before_status:
                    full_answer_parts.append(before_status)
                    if not started:
                        started = True
                        status_ph.empty()
                    _show_streaming("".join(full_answer_parts))

                rest = buffer[idx + len(STATUS_PREFIX):]
                # Try to parse the JSON part
                try:
                    data = json.loads(rest.strip())
                    if not started:
                        _show_status(data.get("msg", "Working…"))
                except Exception:
                    pass  # incomplete JSON — will be handled next iteration
                buffer = ""
                continue

            # ── Regular answer token ──────────────────────────────────────
            # Only flush buffer to answer if it doesn't start with a control prefix
            # and doesn't look like it might be building toward one
            if not buffer.startswith("<") and len(buffer) > 50:
                if not started:
                    started = True
                    status_ph.empty()
                full_answer_parts.append(buffer)
                _show_streaming("".join(full_answer_parts))
                buffer = ""

    except Exception as exc:
        status_ph.empty()
        err_msg = f"⚠️ Pipeline error: {exc}"
        full_answer_parts = [err_msg]
        _show_streaming(err_msg, done=True)

    # Flush any remaining buffer
    if buffer.strip() and META_PREFIX not in buffer and STATUS_PREFIX not in buffer:
        full_answer_parts.append(buffer)

    answer = "".join(full_answer_parts).strip()

    # Final safety: strip any leaked STATUS/META text
    answer = re.sub(r'<<<STATUS>>>\{[^}]*\}', '', answer)
    answer = re.sub(r'<<<META>>>\{.*$', '', answer, flags=re.DOTALL)
    answer = answer.strip()

    _show_streaming(answer, done=True)
    answer_ph.empty()
    status_ph.empty()
    return answer, meta


if ask_btn and query.strip():
    user_query = query.strip()
    st.markdown(
        f'<div class="msg-user-wrap"><div class="msg-user">{user_query}</div></div>',
        unsafe_allow_html=True,
    )

    lang = st.session_state.get("translate_lang", "none")
    lang_out = {"marathi": "marathi", "hindi": "hindi"}.get(lang, "english")

    answer, meta = _parse_stream(user_query, lang_out)

    # Translation
    translated_answer   = ""
    translation_applied = False
    if answer and lang != "none":
        try:
            if lang == "marathi":
                translated_answer = translate_to_marathi(answer)
            elif lang == "hindi":
                translated_answer = translate_to_hindi(answer)
            translation_applied = bool(translated_answer and translated_answer != answer)
        except Exception:
            pass

    msg_meta: dict[str, Any] = {
        "confidence_label":    meta.get("confidence_label", ""),
        "confidence_score":    meta.get("confidence_score", 0.0),
        "model_used":          meta.get("model_used", ""),
        "translation_applied": translation_applied,
        "translated_answer":   translated_answer,
        "lang_used":           lang,
        "citations":           meta.get("citations", {"documents": [], "web": []}),
        "pipeline_trace":      meta.get("pipeline_trace", {}),
        "retrieval_s":         meta.get("retrieval_s"),
    }

    render_message("assistant", answer, msg_meta)

    st.session_state["messages"].append({"role": "user",      "content": user_query, "metadata": {}})
    st.session_state["messages"].append({"role": "assistant", "content": answer,     "metadata": msg_meta})

    sid = ensure_active_session(user.user_id)
    save_turn(session_id=sid, user_query=user_query, assistant_answer=answer, metadata=msg_meta)
    st.session_state["chat_sessions_cache"] = None

elif ask_btn and not query.strip():
    st.warning("Please enter a question before clicking Ask.")
