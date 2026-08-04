"""
app.py — HTE Knowledge Assistant — Streaming UI
Uses st.write_stream for smooth token-by-token display
"""
from __future__ import annotations
import logging, os, re, time, threading, json
from datetime import datetime, timezone, timedelta
from typing import Any, Generator

import streamlit as st

from auth import get_current_user, is_authenticated, render_auth_page, logout
from history_db import (
    ensure_active_session, get_sessions, save_turn,
    start_new_chat, switch_session, delete_session, ChatSession,
)
from crag import run_crag_pipeline, stream_crag_pipeline, CRAGResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Ask HTE — Maharashtra Education AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _init():
    defaults = {
        "messages": [], "active_session_id": None,
        "show_pipeline": False, "translate_lang": "English",
        "chat_sessions_cache": None, "pending_query": "",
        "crawler_started": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"]{
    background:#080808!important;color:#e8e8e8!important;
    font-family:'Inter',-apple-system,sans-serif!important;
}
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0c0c0c,#090909)!important;
    border-right:1px solid #181818!important;
}
[data-testid="stSidebar"]>div{padding:0!important;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important;}
.block-container{padding:0 1.75rem 7rem!important;max-width:800px!important;margin:0 auto!important;}

/* SIDEBAR */
.sb{padding:18px 14px 16px;height:100vh;overflow-y:auto;display:flex;flex-direction:column;}
.sb-user{display:flex;align-items:center;gap:10px;padding:11px 13px;margin-bottom:8px;
    background:#0f0f0f;border:1px solid #1c1c1c;border-radius:11px;}
.sb-av{width:33px;height:33px;border-radius:9px;flex-shrink:0;
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    display:flex;align-items:center;justify-content:center;
    font-size:14px;font-weight:800;color:#fff;}
.sb-un{font-size:12.5px;font-weight:600;color:#e0e0e0;line-height:1.2;}
.sb-em{font-size:10.5px;color:#383838;margin-top:1px;}
.sb-lbl{font-size:9px;text-transform:uppercase;letter-spacing:1.3px;
    color:#252525;font-weight:700;margin:14px 0 7px;}
.sb-stack{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:2px;}
.sbb{padding:2px 8px;border-radius:5px;font-size:9.5px;font-weight:600;
    letter-spacing:.2px;border:1px solid;}
.sbb-ibm{background:#01103a;color:#60a5fa;border-color:#1a3a7a;}
.sbb-gem{background:#011f0e;color:#34d399;border-color:#064e28;}
.sbb-groq{background:#1a1000;color:#fbbf24;border-color:#7a4a00;}
.sbb-crag{background:#150a2e;color:#a78bfa;border-color:#4c1d95;}
.sbb-ch{background:#1a0a00;color:#fb923c;border-color:#7c3400;}
.sb-hg{font-size:9px;text-transform:uppercase;letter-spacing:1px;
    color:#202020;font-weight:700;margin:10px 0 5px 2px;}
.sb-kb{font-size:11px;color:#2a2a2a;padding:7px 11px;
    background:#090909;border:1px solid #131313;border-radius:7px;margin-top:12px;}

/* TOP NAV */
.tnav{display:flex;align-items:center;justify-content:space-between;
    padding:15px 0 13px;border-bottom:1px solid #131313;
    position:sticky;top:0;background:#080808;z-index:200;}
.tn-l{display:flex;align-items:center;gap:11px;}
.tn-logo{width:33px;height:33px;border-radius:9px;flex-shrink:0;
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    display:flex;align-items:center;justify-content:center;font-size:16px;
    box-shadow:0 0 18px rgba(37,99,235,.25);}
.tn-name{font-size:15px;font-weight:800;color:#f0f0f0;letter-spacing:-.3px;}
.tn-sub{font-size:9.5px;color:#383838;margin-top:1px;}
.tn-tag{padding:2px 9px;border-radius:999px;font-size:9.5px;font-weight:600;
    border:1px solid #1c1c1c;background:#0f0f0f;color:#383838;margin-left:8px;}

/* WELCOME */
.wlc{text-align:center;padding:65px 0 38px;}
.wlc-glow{width:70px;height:70px;border-radius:20px;margin:0 auto 20px;
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    display:flex;align-items:center;justify-content:center;font-size:32px;
    box-shadow:0 0 40px rgba(37,99,235,.35),0 0 80px rgba(124,58,237,.15);}
.wlc-title{font-size:27px;font-weight:800;letter-spacing:-.5px;margin-bottom:10px;
    background:linear-gradient(135deg,#fff,#888);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.wlc-sub{font-size:13.5px;color:#484848;line-height:1.75;max-width:380px;margin:0 auto 34px;}
.wlc-cl{font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:#222;font-weight:700;margin-bottom:12px;}

/* CONFIDENCE */
.conf{display:flex;align-items:center;gap:10px;padding:9px 15px;
    border-radius:9px;font-size:12px;font-weight:500;margin-bottom:11px;}
.conf-h{background:linear-gradient(135deg,#021408,#031a0a);border:1px solid #0a4a1e;color:#34d399;}
.conf-m{background:linear-gradient(135deg,#140e00,#1a1200);border:1px solid #5a3d00;color:#fbbf24;}
.conf-l{background:linear-gradient(135deg,#140404,#1a0606);border:1px solid #6a1010;color:#f87171;}
.cdot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.cg{background:#10b981;box-shadow:0 0 7px rgba(16,185,129,.4);}
.cy{background:#f59e0b;box-shadow:0 0 7px rgba(245,158,11,.4);}
.cr{background:#ef4444;box-shadow:0 0 7px rgba(239,68,68,.4);}

/* MESSAGES */
.mu-wrap{display:flex;justify-content:flex-end;margin:18px 0 6px;}
.mu{background:linear-gradient(135deg,#101d3a,#141230);border:1px solid #1c2d5a;
    border-radius:15px 15px 4px 15px;padding:12px 17px;max-width:76%;
    color:#c4d4ef;font-size:14.5px;line-height:1.65;
    box-shadow:0 2px 18px rgba(37,99,235,.07);}
.ma-wrap{margin:6px 0 18px;}
.ma-hdr{display:flex;align-items:center;gap:8px;margin-bottom:8px;}
.ma-av{width:29px;height:29px;border-radius:8px;flex-shrink:0;
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    display:flex;align-items:center;justify-content:center;font-size:14px;
    box-shadow:0 0 10px rgba(37,99,235,.25);}
.ma-name{font-size:11.5px;color:#383838;font-weight:600;letter-spacing:.2px;}
.ma-model{font-size:9px;font-family:'JetBrains Mono',monospace;
    background:#0f0f0f;border:1px solid #1c1c1c;border-radius:4px;
    padding:2px 6px;color:#2a2a2a;}
.ma-body{background:linear-gradient(135deg,#0c0c0c,#090909);
    border:1px solid #161616;border-radius:4px 15px 15px 15px;
    padding:19px 23px;margin-left:37px;
    font-size:14.5px;line-height:1.85;color:#cccccc;
    box-shadow:0 2px 25px rgba(0,0,0,.25);}
.ma-body h1,.ma-body h2,.ma-body h3{color:#efefef;font-weight:700;margin:16px 0 8px;}
.ma-body h1{font-size:19px;border-bottom:1px solid #181818;padding-bottom:7px;}
.ma-body h2{font-size:16.5px;}.ma-body h3{font-size:14.5px;}
.ma-body ul,.ma-body ol{padding-left:21px;margin:9px 0;}
.ma-body li{margin-bottom:5px;color:#c0c0c0;}
.ma-body li::marker{color:#3b82f6;}
.ma-body strong{color:#efefef;font-weight:700;}
.ma-body a{color:#60a5fa;text-decoration:none;border-bottom:1px solid #1d3a6a;}
.ma-body code{background:#131313;border:1px solid #1c1c1c;border-radius:5px;
    padding:2px 6px;font-family:'JetBrains Mono',monospace;font-size:12.5px;color:#fb923c;}
.ma-body pre{background:#0c0c0c;border:1px solid #181818;border-radius:9px;
    padding:15px;overflow-x:auto;margin:13px 0;}
.ma-body pre code{background:none;border:none;padding:0;color:#ccc;font-size:12.5px;}
.ma-body table{width:100%;border-collapse:collapse;margin:15px 0;font-size:13px;
    border-radius:9px;overflow:hidden;}
.ma-body th{background:#111;border:1px solid #1c1c1c;padding:9px 13px;
    color:#ddd;font-weight:700;text-align:left;font-size:11.5px;
    text-transform:uppercase;letter-spacing:.4px;}
.ma-body td{border:1px solid #131313;padding:8px 13px;color:#b8b8b8;}
.ma-body tr:nth-child(even) td{background:#090909;}
.ma-body tr:hover td{background:#0e0e0e;}
.ma-body blockquote{border-left:3px solid #3b82f6;padding:11px 17px;
    background:linear-gradient(135deg,#091020,#0c1228);
    border-radius:0 9px 9px 0;margin:13px 0;color:#6a8fc4;font-style:italic;}
.ma-body hr{border:none;border-top:1px solid #181818;margin:14px 0;}
.ma-body p{margin-bottom:9px;}

/* Streaming container */
.stream-body{background:linear-gradient(135deg,#0c0c0c,#090909);
    border:1px solid #161616;border-radius:4px 15px 15px 15px;
    padding:19px 23px;margin-left:37px;
    font-size:14.5px;line-height:1.85;color:#cccccc;
    box-shadow:0 2px 25px rgba(0,0,0,.25);}
/* cursor blink */
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.cursor{display:inline-block;width:2px;height:1em;background:#3b82f6;
    margin-left:2px;vertical-align:text-bottom;animation:blink .8s infinite;}

/* SOURCES */
.src-wrap{margin:9px 0 0 37px;padding:13px 15px;
    background:#090909;border:1px solid #131313;border-radius:11px;}
.src-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.src-lbl{font-size:9px;text-transform:uppercase;letter-spacing:1.2px;color:#222;font-weight:700;}
.src-cnt{font-size:9.5px;color:#222;font-family:'JetBrains Mono',monospace;}
.src-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:6px;}
.src-card{display:flex;align-items:flex-start;gap:8px;padding:9px 11px;border-radius:8px;border:1px solid;}
.src-doc{background:linear-gradient(135deg,#081422,#0b1a2e);border-color:#182856;}
.src-doc:hover{border-color:#2a4a8a;}
.src-tbl{background:linear-gradient(135deg,#04100a,#071512);border-color:#094018;}
.src-tbl:hover{border-color:#125030;}
.src-web{background:#0c0c0c;border-color:#1c1c1c;text-decoration:none;display:flex;}
.src-web:hover{background:#101010;border-color:#282828;}
.src-ico{font-size:14px;flex-shrink:0;margin-top:1px;}
.src-info{min-width:0;flex:1;}
.src-nm{font-size:11px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.src-doc .src-nm{color:#5a9af0;}
.src-tbl .src-nm{color:#30c882;}
.src-web .src-nm{color:#666;}
.src-mt{font-size:9px;color:#222;margin-top:2px;font-family:'JetBrains Mono',monospace;}
.src-vfy{display:inline-block;padding:1px 5px;border-radius:3px;font-size:8.5px;
    background:#071512;color:#30c882;border:1px solid #094018;margin-top:2px;}

/* TRANSLATION */
.tr-wrap{margin:9px 0 0 37px;}
.tr-box{background:linear-gradient(135deg,#041008,#06140a);
    border:1px solid #094018;border-left:3px solid #10b981;
    border-radius:0 11px 11px 11px;padding:15px 19px;}
.tr-hdr{font-family:'Inter',sans-serif;font-size:9.5px;color:#10b981;
    font-weight:700;text-transform:uppercase;letter-spacing:1px;
    margin-bottom:9px;padding-bottom:7px;border-bottom:1px solid #092e18;}
.tr-txt{font-family:'Noto Serif Devanagari','Noto Serif',serif;
    font-size:15px;line-height:1.9;color:#4a9a68;}

/* INPUT */
.inp-hint{font-size:10.5px;color:#222;margin-bottom:7px;
    display:flex;align-items:center;gap:6px;}
.lang-pill{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;
    border-radius:999px;background:#0d0d0d;border:1px solid #181818;
    font-size:10.5px;color:#303030;}

[data-testid="stTextArea"] textarea{
    background:#0d0d0d!important;border:1px solid #1c1c1c!important;
    border-radius:13px!important;color:#e0e0e0!important;
    font-size:14.5px!important;font-family:'Inter',sans-serif!important;
    resize:none!important;line-height:1.65!important;}
[data-testid="stTextArea"] textarea:focus{
    border-color:#2563eb!important;
    box-shadow:0 0 0 3px rgba(37,99,235,.1),0 0 20px rgba(37,99,235,.04)!important;}
[data-testid="stTextArea"] textarea::placeholder{color:#252525!important;}

.stButton>button{border-radius:9px!important;font-family:'Inter',sans-serif!important;
    font-size:13px!important;font-weight:600!important;
    transition:all .18s ease!important;letter-spacing:.2px!important;}
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#2563eb,#7c3aed)!important;
    border:none!important;color:#fff!important;
    box-shadow:0 4px 18px rgba(37,99,235,.25)!important;}
.stButton>button[kind="primary"]:hover{
    box-shadow:0 6px 28px rgba(37,99,235,.45)!important;
    transform:translateY(-1px)!important;}
.stButton>button[kind="secondary"]{
    background:#0d0d0d!important;border:1px solid #1c1c1c!important;color:#484848!important;}
.stButton>button[kind="secondary"]:hover{
    background:#111!important;border-color:#282828!important;color:#787878!important;}

[data-testid="stSidebar"] .stButton>button{
    text-align:left!important;justify-content:flex-start!important;
    background:transparent!important;border:1px solid transparent!important;
    color:#383838!important;padding:8px 11px!important;border-radius:8px!important;
    font-size:12.5px!important;width:100%!important;
    overflow:hidden!important;text-overflow:ellipsis!important;white-space:nowrap!important;
    transform:none!important;box-shadow:none!important;}
[data-testid="stSidebar"] .stButton>button:hover{
    background:#0f0f0f!important;border-color:#1c1c1c!important;color:#686868!important;}

[data-testid="stToggle"] label,[data-testid="stToggle"] p{font-size:12px!important;color:#484848!important;}
[data-baseweb="select"] div{background:#0d0d0d!important;border-color:#1c1c1c!important;color:#787878!important;border-radius:8px!important;}
[data-testid="stExpander"]{border:1px solid #131313!important;border-radius:9px!important;
    background:#090909!important;margin:5px 0 5px 37px!important;}
[data-testid="stExpander"] summary{font-size:11.5px!important;color:#2a2a2a!important;padding:8px 13px!important;}
hr{border:none!important;border-top:1px solid #131313!important;margin:7px 0!important;}
[data-testid="stAlert"]{background:#14100a!important;border:1px solid #5a3d00!important;
    border-radius:9px!important;color:#fbbf24!important;font-size:12.5px!important;}
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:#1c1c1c;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

# ── Crawler ───────────────────────────────────────────────────────────────────
if not st.session_state["crawler_started"]:
    try:
        from crawler import start_scheduler
        start_scheduler()
    except Exception: pass
    st.session_state["crawler_started"] = True

# ── Auth ──────────────────────────────────────────────────────────────────────
if not is_authenticated():
    render_auth_page()
    st.stop()

user = get_current_user()
assert user is not None
ensure_active_session(user.user_id)

if st.session_state.get("chat_sessions_cache") is None:
    st.session_state["chat_sessions_cache"] = get_sessions(user.user_id, limit=60)
sessions: list[ChatSession] = st.session_state.get("chat_sessions_cache") or []

kb_count = 0
try:
    import chromadb as _cc
    _col = _cc.PersistentClient(path=os.getenv("CHROMA_PATH","./chroma_db")
                                ).get_collection(os.getenv("CHROMA_COLLECTION","hte_documents"))
    kb_count = _col.count()
except Exception: pass

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sb-user">
        <div class="sb-av">{user.email[0].upper()}</div>
        <div>
            <div class="sb-un">{user.email.split("@")[0][:16]}</div>
            <div class="sb-em">{user.email[:28]}{"…" if len(user.email)>28 else ""}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        if st.button("＋ New Chat", use_container_width=True, type="secondary", key="nb"):
            start_new_chat(user.user_id)
            st.session_state["chat_sessions_cache"]=None
            st.rerun()
    with c2:
        if st.button("⎋ Sign Out", use_container_width=True, type="secondary", key="so"):
            logout(); st.rerun()

    st.markdown("""<div class="sb-lbl">AI Stack</div>
    <div class="sb-stack">
        <span class="sbb sbb-ibm">IBM Granite 4.0</span>
        <span class="sbb sbb-gem">Gemini Embed</span>
        <span class="sbb sbb-groq">Groq Fallback</span>
        <span class="sbb sbb-crag">CRAG</span>
        <span class="sbb sbb-ch">ChromaDB</span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sb-lbl">Settings</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        mr = st.toggle("🔤 Marathi",
                       value=st.session_state.get("translate_lang")=="Marathi",key="mr_tog")
        st.session_state["translate_lang"] = "Marathi" if mr else "English"
    with c2:
        st.session_state["show_pipeline"] = st.toggle(
            "🔬 Trace", value=st.session_state.get("show_pipeline",False), key="pl_tog")

    try:
        from crawler import trigger_crawl_now
        if st.button("🌐 Crawl Sources", use_container_width=True, type="secondary", key="crawl"):
            with st.spinner("Crawling…"):
                s = trigger_crawl_now()
            st.success(f"✓ {s.get('ingested',0)} docs")
    except Exception: pass

    # History grouped by date
    st.markdown('<div class="sb-lbl">Chat History</div>', unsafe_allow_html=True)

    def _group(sl):
        now=datetime.now(timezone.utc); td=now.date()
        yd=(now-timedelta(days=1)).date(); wk=(now-timedelta(days=7)).date()
        g={"Today":[],"Yesterday":[],"Last 7 Days":[],"Older":[]}
        for s in sl:
            try: d=datetime.fromisoformat(s.updated_at.replace("Z","+00:00")).date()
            except: d=td
            if d==td: g["Today"].append(s)
            elif d==yd: g["Yesterday"].append(s)
            elif d>=wk: g["Last 7 Days"].append(s)
            else: g["Older"].append(s)
        return g

    for grp,grp_sess in _group(sessions).items():
        if not grp_sess: continue
        st.markdown(f'<div class="sb-hg">{grp}</div>', unsafe_allow_html=True)
        for sess in grp_sess:
            active = sess.id == st.session_state.get("active_session_id")
            td_ = sess.title[:40]+"…" if len(sess.title)>40 else sess.title
            hc1,hc2 = st.columns([8,1])
            with hc1:
                if st.button(f"{'▸ ' if active else ''}{td_}", key=f"h_{sess.id}", use_container_width=True):
                    switch_session(sess.id)
                    st.session_state["chat_sessions_cache"]=None; st.rerun()
            with hc2:
                if st.button("×", key=f"hd_{sess.id}", help="Delete"):
                    delete_session(sess.id,user.user_id)
                    if sess.id==st.session_state.get("active_session_id"):
                        start_new_chat(user.user_id)
                    st.session_state["chat_sessions_cache"]=None; st.rerun()

    if not sessions:
        st.markdown('<div style="font-size:11px;color:#1e1e1e;padding:8px 4px;">No conversations yet.</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sb-kb">📄 {kb_count:,} chunks &nbsp;·&nbsp; 🗄️ Dual-track CRAG</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TOP NAV
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="tnav">
  <div class="tn-l">
    <div class="tn-logo">🏛️</div>
    <div>
      <div class="tn-name">Ask HTE</div>
      <div class="tn-sub">Maharashtra Education AI</div>
    </div>
    <span class="tn-tag">Maharashtra Education</span>
  </div>
</div>
<div style="height:6px"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _clean(text):
    if not text: return text
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{0,3}(#{1,3}\s*)?(Sources?|References?|Citations?)\s*\n.*$',
                  '', text, flags=re.IGNORECASE|re.DOTALL)
    for pat in [r'^.*?(?:assistant\s*final|<\|assistant\|>)\s*',
                r'^assistantfinal\s*',r'^assistant\s*']:
        c=re.sub(pat,'',text,flags=re.IGNORECASE|re.DOTALL)
        if c!=text: text=c.strip(); break
    return re.sub(r'\n{3,}','\n\n',text).strip()

def _conf_banner(label,score,model):
    pct=int(score*100)
    mt=f'<span style="font-size:9.5px;color:#282828;font-family:\'JetBrains Mono\',monospace;margin-left:10px;">{model}</span>'
    if label=="CORRECT":
        return f'<div class="conf conf-h"><span class="cdot cg"></span>High confidence ({pct}%) — Grounded in official HTE documents{mt}</div>'
    elif label=="AMBIGUOUS":
        return f'<div class="conf conf-m"><span class="cdot cy"></span>Medium confidence ({pct}%) — Official documents + verified web sources{mt}</div>'
    else:
        return f'<div class="conf conf-l"><span class="cdot cr"></span>Low document match ({pct}%) — From verified government websites{mt}</div>'

def _sources_html(metadata):
    cits=metadata.get("citations",{})
    docs=cits.get("documents",[]); webs=cits.get("web",[])
    if not docs and not webs: return ""
    total=len(docs)+len(webs)
    p=[f'<div class="src-wrap"><div class="src-head">'
       f'<div class="src-lbl">Sources &amp; References</div>'
       f'<div class="src-cnt">{total} source{"s" if total!=1 else ""}</div>'
       f'</div><div class="src-grid">']
    for c in docs:
        src=c.get("source",""); pg=c.get("page","")
        et=c.get("element_type","text"); score=c.get("score",0)
        verified=c.get("verified",False)
        lang=c.get("language","en"); st_=c.get("source_type","pdf")
        disp=src[:24]+"…" if len(src)>24 else src
        icon="📊" if et=="table" else "📄"
        cls="src-card src-tbl" if et=="table" else "src-card src-doc"
        meta=f"p.{pg}"
        if lang!="en": meta+=f" · {lang.upper()}"
        if st_ not in ("pdf",""): meta+=f" · {st_}"
        vfy=f'<div class="src-vfy">✓ verified</div>' if verified else ""
        p.append(f'<div class="{cls}"><div class="src-ico">{icon}</div>'
                 f'<div class="src-info"><div class="src-nm" title="{src}">{disp}</div>'
                 f'<div class="src-mt">{meta} · {score:.2f}</div>{vfy}</div></div>')
    for w in webs:
        url=w.get("url","#"); title=w.get("title",url); score=w.get("score",0)
        try:
            from urllib.parse import urlparse as _up
            domain=_up(url).netloc.replace("www.","")[:22]
        except: domain=url[:22]
        disp=title[:26]+"…" if len(title)>26 else title
        p.append(f'<a href="{url}" target="_blank" class="src-card src-web">'
                 f'<div class="src-ico">🔗</div>'
                 f'<div class="src-info"><div class="src-nm" title="{title}">{disp}</div>'
                 f'<div class="src-mt">{domain} · {score:.2f}</div></div></a>')
    p.append('</div></div>')
    return "".join(p)

def _render_pipeline(metadata):
    trace=metadata.get("pipeline_trace",{})
    if not trace: return
    def _kv(d):
        bits=[]
        for k,v in d.items():
            if k in("original","rewritten","entities"): continue
            bits.append(f'<span style="color:#3b82f6">{v}</span><span style="color:#1c1c1c"> {k}</span>')
        return " · ".join(bits)
    with st.expander("🔬 Pipeline trace",expanded=False):
        stages=[("Query","🔤",trace.get("query",{})),
                ("Retrieval","🗄️",trace.get("retrieval",{})),
                ("CrossEncoder","⚖️",trace.get("crossencoder",{})),
                ("CRAG","🔀",trace.get("corrective_branch",{})),
                ("Context","📋",trace.get("context",{})),
                ("LLM","🤖",trace.get("llm_generation",{}))]
        rows=[]
        for name,icon,data in stages:
            if not data: continue
            rows.append(f'<div style="display:flex;align-items:flex-start;gap:9px;padding:7px 13px;'
                        f'background:#090909;border:1px solid #131313;border-radius:7px;margin-bottom:3px;'
                        f'font-family:\'JetBrains Mono\',monospace;font-size:10.5px;">'
                        f'<span style="color:#10b981">{icon}</span>'
                        f'<span style="color:#2a2a2a;min-width:90px">{name}</span>'
                        f'<span>{_kv(data)}</span></div>')
        t=trace.get("total_time_s","")
        if t: rows.append(f'<div style="color:#222;font-size:10px;text-align:right;margin-top:3px;font-family:\'JetBrains Mono\',monospace;">⏱ {t}s total</div>')
        st.markdown("".join(rows),unsafe_allow_html=True)

def render_message(role,content,metadata):
    if role=="user":
        st.markdown(f'<div class="mu-wrap"><div class="mu">{content}</div></div>',unsafe_allow_html=True)
        return
    conf_label=metadata.get("confidence_label","")
    conf_score=metadata.get("confidence_score",0.0)
    model_used=metadata.get("model_used","")
    if conf_label:
        st.markdown(_conf_banner(conf_label,conf_score,model_used),unsafe_allow_html=True)
    mt=f'<span class="ma-model">{model_used}</span>' if model_used else ""
    st.markdown(f'<div class="ma-hdr"><div class="ma-av">🏛️</div>'
                f'<span class="ma-name">HTE Assistant</span>{mt}</div>',unsafe_allow_html=True)
    cleaned=_clean(content)
    st.markdown('<div class="ma-body">',unsafe_allow_html=True)
    st.markdown(cleaned)
    st.markdown('</div>',unsafe_allow_html=True)
    # Actions
    a1,a2,_=st.columns([1,1,7])
    with a1:
        st.download_button("⬇ Save",data=cleaned,file_name="hte_answer.md",
                           mime="text/markdown",key=f"dl_{abs(hash(content))%99999}")
    with a2:
        if st.button("🔁 Retry",key=f"rt_{abs(hash(content))%99999}",type="secondary"):
            lu=next((m["content"] for m in reversed(st.session_state["messages"]) if m["role"]=="user"),None)
            if lu:
                st.session_state["pending_query"]=lu
                st.session_state["messages"]=st.session_state["messages"][:-2]
                st.rerun()
    src=_sources_html(metadata)
    if src: st.markdown(src,unsafe_allow_html=True)
    if metadata.get("translation_applied") and metadata.get("translated_answer"):
        lang=metadata.get("translate_lang","Marathi")
        lbl="🔤 मराठी अनुवाद" if lang=="Marathi" else "🔤 हिंदी अनुवाद"
        st.markdown(f'<div class="tr-wrap"><div class="tr-box">'
                    f'<div class="tr-hdr">{lbl} · {lang} Translation</div>'
                    f'<div class="tr-txt">{metadata["translated_answer"]}</div>'
                    f'</div></div>',unsafe_allow_html=True)
    if st.session_state.get("show_pipeline",False):
        _render_pipeline(metadata)
    st.markdown("<div style='height:2px'></div>",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════
messages=st.session_state.get("messages",[])

CHIPS=[
    {"i":"💰","t":"What is the fee structure for SC/ST students in government engineering colleges?"},
    {"i":"🎓","t":"How do I apply for EBC freeship scholarship in Maharashtra?"},
    {"i":"📋","t":"What are the MHT-CET 2025 eligibility criteria?"},
    {"i":"📅","t":"When does CAP Round 1 registration close for 2025-26?"},
    {"i":"🏫","t":"What hostel facilities are available for girl students?"},
    {"i":"📜","t":"Explain the college affiliation renewal process"},
    {"i":"🔢","t":"Show me the engineering seat matrix for CAP 2025"},
    {"i":"📊","t":"What government resolutions exist for higher education fees?"},
]

if messages:
    for msg in messages:
        render_message(msg["role"],msg["content"],msg.get("metadata",{}))
else:
    st.markdown("""
    <div class="wlc">
      <div class="wlc-glow">🏛️</div>
      <div class="wlc-title">Ask anything about HTE Maharashtra</div>
      <div class="wlc-sub">AI assistant for engineering &amp; diploma admissions, fees,
      scholarships, government resolutions, circulars — grounded in 3,001 official documents.</div>
      <div class="wlc-cl">Popular Questions</div>
    </div>""", unsafe_allow_html=True)
    cols=st.columns(2)
    for idx,chip in enumerate(CHIPS):
        with cols[idx%2]:
            txt=chip["t"]
            lbl=f"{chip['i']}  {txt[:54]}{'…' if len(txt)>54 else ''}"
            if st.button(lbl,key=f"chip_{idx}",use_container_width=True,type="secondary"):
                st.session_state["pending_query"]=chip["t"]; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════
prefill=st.session_state.get("pending_query","") or ""
if prefill: st.session_state["pending_query"]=""

st.markdown("<div style='height:10px'></div>",unsafe_allow_html=True)
lang=st.session_state.get("translate_lang","English")
li={"English":"🇺🇸","Marathi":"🇮🇳","Hindi":"🇮🇳"}.get(lang,"🌐")
st.markdown(f'<div class="inp-hint"><span class="lang-pill">{li} {lang}</span>'
            f'<span>· Toggle language in sidebar</span></div>',unsafe_allow_html=True)

cq,cb=st.columns([8,1])
with cq:
    query=st.text_area("q",value=prefill,
        placeholder="Ask about fees, scholarships, admissions, GRs, circulars…",
        height=80,key="qi",label_visibility="collapsed")
with cb:
    st.markdown("<div style='height:18px'></div>",unsafe_allow_html=True)
    ask_btn=st.button("Ask →",type="primary",use_container_width=True,key="ab")


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING SUBMIT
# ══════════════════════════════════════════════════════════════════════════════
if ask_btn and query.strip():
    q = query.strip()

    # Show user message immediately
    st.markdown(f'<div class="mu-wrap"><div class="mu">{q}</div></div>',unsafe_allow_html=True)

    # Show assistant header
    st.markdown('<div class="ma-hdr"><div class="ma-av">🏛️</div>'
                '<span class="ma-name">HTE Assistant</span>'
                '<span class="ma-model">ibm-granite-4.0</span></div>',
                unsafe_allow_html=True)

    # Stream the response
    full_tokens: list[str] = []
    final_meta:  dict      = {}

    st.markdown('<div class="stream-body">',unsafe_allow_html=True)

    def _token_generator():
        """Wraps stream_crag_pipeline, extracts META, yields clean tokens."""
        for token in stream_crag_pipeline(q):
            if token.startswith("\n<<<META>>>"):
                meta_str = token.split("<<<META>>>", 1)[1].strip()
                try:
                    final_meta.update(json.loads(meta_str))
                except Exception: pass
                return   # stop iteration
            full_tokens.append(token)
            yield token

    # st.write_stream renders tokens progressively with proper markdown
    streamed_text = st.write_stream(_token_generator())

    st.markdown('</div>',unsafe_allow_html=True)

    # Full answer
    full_answer = _clean("".join(full_tokens) if full_tokens else (streamed_text or ""))

    # If streaming not available, fall back
    if not full_answer:
        result: CRAGResult = run_crag_pipeline(q, translate_to_marathi=False)
        full_answer = result.answer
        final_meta = {
            "confidence_label": result.confidence_label,
            "confidence_score": result.confidence_score,
            "model_used":       result.model_used,
            "citations":        {"documents":[
                {"index":i+1,"source":c.source,"page":c.page,"category":c.category,
                 "score":round(c.score,3),"element_type":getattr(c,"element_type","text"),
                 "language":getattr(c,"language","en"),"source_type":getattr(c,"source_type","pdf"),
                 "verified":getattr(c,"verified",False)}
                for i,c in enumerate(result.doc_sources)
            ],"web":[
                {"index":i+1,"title":w.title,"url":w.url,"score":round(w.score,3)}
                for i,w in enumerate(result.web_sources)
            ]},
            "pipeline_trace": result.pipeline_trace,
        }

    # Translation
    translated_answer=""; translation_applied=False
    needs_tr=st.session_state.get("translate_lang","English")!="English"
    if needs_tr and full_answer:
        try:
            from translator import translate_to_marathi as _tr
            translated_answer=_tr(full_answer)
            translation_applied=bool(translated_answer and translated_answer!=full_answer)
        except Exception: pass

    msg_meta={
        "confidence_label":    final_meta.get("confidence_label","AMBIGUOUS"),
        "confidence_score":    final_meta.get("confidence_score",0.5),
        "model_used":          final_meta.get("model_used","ibm-granite-4.0"),
        "translation_applied": translation_applied,
        "translated_answer":   translated_answer,
        "translate_lang":      lang,
        "selected_agents":     [],
        "citations":           final_meta.get("citations",{"documents":[],"web":[]}),
        "pipeline_trace":      final_meta.get("pipeline_trace",{}),
    }

    # Show sources + translation + pipeline below streamed text
    src=_sources_html(msg_meta)
    if src: st.markdown(src,unsafe_allow_html=True)
    if translation_applied and translated_answer:
        lbl="🔤 मराठी अनुवाद" if lang=="Marathi" else "🔤 हिंदी अनुवाद"
        st.markdown(f'<div class="tr-wrap"><div class="tr-box">'
                    f'<div class="tr-hdr">{lbl} · {lang} Translation</div>'
                    f'<div class="tr-txt">{translated_answer}</div>'
                    f'</div></div>',unsafe_allow_html=True)
    if st.session_state.get("show_pipeline",False):
        _render_pipeline(msg_meta)

    # Save to session + DB
    st.session_state["messages"].append({"role":"user","content":q,"metadata":{}})
    st.session_state["messages"].append({"role":"assistant","content":full_answer,"metadata":msg_meta})

    sid=ensure_active_session(user.user_id)
    save_turn(session_id=sid,user_query=q,assistant_answer=full_answer,metadata=msg_meta)
    st.session_state["chat_sessions_cache"]=None
    st.rerun()

elif ask_btn and not query.strip():
    st.warning("Please enter a question before clicking Ask.")
