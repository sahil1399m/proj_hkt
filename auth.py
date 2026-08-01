"""
auth.py — HTE Knowledge Assistant Authentication
Clean Sign In / Sign Up / Logout — no feedback section
"""
from __future__ import annotations
import os, time, logging
from dataclasses import dataclass, field
from typing import Optional
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
from gotrue.errors import AuthApiError

load_dotenv()
logger = logging.getLogger(__name__)

MAX_FAILED_ATTEMPTS   = 5
LOCKOUT_SECONDS       = 300
SESSION_TIMEOUT_HOURS = 8


@st.cache_resource(show_spinner=False)
def _get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
    key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")
    if "/rest" in url:
        url = url.split("/rest")[0]
    return create_client(url, key)

def get_supabase() -> Client:
    return _get_supabase()

@dataclass
class UserSession:
    user_id: str
    email: str
    access_token: str
    refresh_token: str
    logged_in_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.logged_in_at) / 3600 >= SESSION_TIMEOUT_HOURS

    @property
    def display_name(self) -> str:
        return self.email.split("@")[0]

    @property
    def initials(self) -> str:
        return self.email[0].upper()


def _init_auth_state() -> None:
    defaults = {
        "user_session": None,
        "auth_failed_attempts": 0,
        "auth_lockout_until": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def get_current_user() -> Optional[UserSession]:
    _init_auth_state()
    s = st.session_state.get("user_session")
    if s and s.is_expired:
        logout(silent=True)
        return None
    return s

def is_authenticated() -> bool:
    return get_current_user() is not None

def _is_locked_out() -> bool:
    return time.time() < st.session_state.get("auth_lockout_until", 0.0)

def _record_fail() -> None:
    st.session_state["auth_failed_attempts"] += 1
    if st.session_state["auth_failed_attempts"] >= MAX_FAILED_ATTEMPTS:
        st.session_state["auth_lockout_until"] = time.time() + LOCKOUT_SECONDS
        st.session_state["auth_failed_attempts"] = 0

def _reset_fails() -> None:
    st.session_state["auth_failed_attempts"] = 0
    st.session_state["auth_lockout_until"] = 0.0

def _validate_password(pw: str) -> list[str]:
    errs = []
    if len(pw) < 8:                      errs.append("at least 8 characters")
    if not any(c.isupper() for c in pw): errs.append("one uppercase letter")
    if not any(c.islower() for c in pw): errs.append("one lowercase letter")
    if not any(c.isdigit() for c in pw): errs.append("one number")
    return errs

def register(email: str, password: str) -> tuple[bool, str]:
    _init_auth_state()
    if _is_locked_out():
        return False, f"Too many attempts. Try again in {int(st.session_state['auth_lockout_until']-time.time())}s."
    email = email.strip().lower()
    if not email or "@" not in email:
        return False, "Enter a valid email address."
    issues = _validate_password(password)
    if issues:
        return False, "Password must include: " + ", ".join(issues) + "."
    try:
        res = get_supabase().auth.sign_up({"email": email, "password": password})
        if res.user:
            return login(email, password)
        return False, "Registration failed. Please try again."
    except AuthApiError as e:
        if "already" in str(e).lower():
            return False, "Account already exists. Please sign in."
        return False, f"Error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"

def login(email: str, password: str) -> tuple[bool, str]:
    _init_auth_state()
    if _is_locked_out():
        return False, f"Locked out. Try again in {int(st.session_state['auth_lockout_until']-time.time())}s."
    email = email.strip().lower()
    if not email or not password:
        return False, "Email and password are required."
    try:
        res = get_supabase().auth.sign_in_with_password({"email": email, "password": password})
        if res.user and res.session:
            _reset_fails()
            st.session_state["user_session"] = UserSession(
                user_id=res.user.id, email=res.user.email,
                access_token=res.session.access_token,
                refresh_token=res.session.refresh_token,
            )
            return True, "Welcome!"
        _record_fail()
        return False, "Invalid email or password."
    except AuthApiError as e:
        _record_fail()
        if "email not confirmed" in str(e).lower():
            return False, "Please confirm your email first."
        return False, "Invalid email or password."
    except Exception as e:
        return False, f"Unexpected error: {e}"

def logout(silent: bool = False) -> None:
    _init_auth_state()
    if not silent:
        try: get_supabase().auth.sign_out()
        except Exception: pass
    st.session_state["user_session"] = None
    for k in ["active_session_id", "messages", "chat_sessions_cache",
              "panel", "show_pipeline", "translate_lang", "dark_mode"]:
        st.session_state.pop(k, None)


# ── AUTH PAGE ─────────────────────────────────────────────────────────────────

def render_auth_page() -> None:
    _init_auth_state()

    # Try to load news
    news_items = []
    try:
        from hte_feed import get_hte_news
        news_items = get_hte_news(max_articles=4)
    except Exception:
        pass

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewBlockContainer"]{
    background:#09090b!important;color:#e4e4e7!important;
    font-family:'Inter',-apple-system,sans-serif!important;
}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],[data-testid="stSidebar"]{display:none!important;}
.block-container{padding:0!important;max-width:100%!important;}

/* Layout */
.auth-grid{display:grid;grid-template-columns:1fr 420px;min-height:100vh;max-width:1100px;margin:0 auto;}

/* Left panel */
.auth-left{padding:48px 40px;display:flex;flex-direction:column;justify-content:space-between;}
.brand{display:flex;align-items:center;gap:12px;margin-bottom:48px;}
.brand-icon{width:40px;height:40px;border-radius:10px;
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    display:flex;align-items:center;justify-content:center;font-size:20px;}
.brand-name{font-size:17px;font-weight:600;color:#f4f4f5;}
.brand-sub{font-size:11px;color:#52525b;margin-top:1px;}
.left-headline{font-size:32px;font-weight:700;color:#f4f4f5;line-height:1.3;
    letter-spacing:-0.5px;margin-bottom:16px;}
.left-sub{font-size:14px;color:#71717a;line-height:1.7;margin-bottom:36px;}
.feature-list{display:flex;flex-direction:column;gap:12px;margin-bottom:40px;}
.feature-item{display:flex;align-items:center;gap:12px;font-size:13.5px;color:#a1a1aa;}
.feature-icon{width:32px;height:32px;border-radius:8px;
    background:#18181b;border:1px solid #27272a;
    display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0;}

/* News feed */
.news-label{font-size:10px;text-transform:uppercase;letter-spacing:1px;
    color:#3f3f46;font-weight:600;margin-bottom:12px;}
.news-card{background:#111115;border:1px solid #18181b;border-radius:10px;
    padding:14px 16px;margin-bottom:8px;cursor:pointer;text-decoration:none;
    display:block;transition:border-color .15s;}
.news-card:hover{border-color:#27272a;}
.news-src{font-size:10px;color:#3f3f46;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px;}
.news-title{font-size:13px;font-weight:500;color:#d4d4d8;line-height:1.45;}
.news-desc{font-size:11.5px;color:#52525b;margin-top:4px;line-height:1.5;}

/* Right panel */
.auth-right{background:#0c0c0f;border-left:1px solid #18181b;
    display:flex;flex-direction:column;justify-content:center;padding:48px 44px;}
.auth-title{font-size:24px;font-weight:700;color:#f4f4f5;letter-spacing:-.3px;margin-bottom:6px;}
.auth-sub{font-size:13px;color:#71717a;margin-bottom:28px;line-height:1.5;}
.auth-footer{font-size:11.5px;color:#3f3f46;text-align:center;margin-top:24px;line-height:1.6;}

/* Inputs */
[data-testid="stTextInput"] input{
    background:#111115!important;border:1px solid #27272a!important;
    border-radius:8px!important;color:#e4e4e7!important;
    font-size:14px!important;padding:11px 14px!important;
    font-family:'Inter',sans-serif!important;}
[data-testid="stTextInput"] input:focus{border-color:#3b82f6!important;
    box-shadow:0 0 0 3px rgba(59,130,246,.1)!important;}
[data-testid="stTextInput"] label{color:#71717a!important;font-size:12px!important;font-weight:500!important;}

/* Buttons */
.stButton>button{border-radius:8px!important;font-family:'Inter',sans-serif!important;font-weight:500!important;}
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#2563eb,#7c3aed)!important;
    border:none!important;color:white!important;font-size:14px!important;padding:11px!important;}
.stButton>button[kind="primary"]:hover{opacity:.9!important;}
.stButton>button[kind="secondary"]{background:transparent!important;
    border:1px solid #27272a!important;color:#71717a!important;font-size:13px!important;}
.stButton>button[kind="secondary"]:hover{background:#111115!important;color:#d4d4d8!important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:#111115!important;border-radius:8px!important;
    border:1px solid #18181b!important;padding:3px!important;}
.stTabs [data-baseweb="tab"]{color:#71717a!important;font-size:13px!important;
    font-weight:500!important;border-radius:6px!important;padding:8px 20px!important;}
.stTabs [aria-selected="true"]{background:#1c1c1f!important;color:#f4f4f5!important;}
.stTabs [data-baseweb="tab-panel"]{padding:20px 0 0!important;}

[data-testid="stAlert"]{border-radius:8px!important;font-size:13px!important;}
</style>
""", unsafe_allow_html=True)

    # Build news HTML
    news_html = ""
    if news_items:
        cards = []
        for a in news_items[:4]:
            t = a.get("title","")[:70]
            d = a.get("description","")[:100]
            u = a.get("url","#")
            s = a.get("source","")
            cards.append(f'''<a href="{u}" target="_blank" class="news-card">
                <div class="news-src">{s}</div>
                <div class="news-title">{t}</div>
                {"" if not d else f'<div class="news-desc">{d}</div>'}
            </a>''')
        news_html = f'''<div class="news-label">📰 Latest HTE Updates</div>{"".join(cards)}'''

    features = [
        ("🔍", "Semantic search across 2,000+ official HTE documents"),
        ("📊", "Table-aware retrieval for fees, scholarships & seat matrices"),
        ("🌐", "Live web search from official Maharashtra government sites"),
        ("🗣️", "Answers in English, Marathi and Hindi"),
        ("📄", "Source citations with page numbers for every answer"),
    ]
    feat_html = "".join(
        f'<div class="feature-item"><div class="feature-icon">{icon}</div>{text}</div>'
        for icon, text in features
    )

    col_left, col_right = st.columns([1.1, 1])

    with col_left:
        st.markdown(f"""
        <div class="auth-left">
            <div>
                <div class="brand">
                    <div class="brand-icon">🏛️</div>
                    <div>
                        <div class="brand-name">HTE Knowledge Assistant</div>
                        <div class="brand-sub">Maharashtra Higher &amp; Technical Education</div>
                    </div>
                </div>
                <div class="left-headline">Instant answers from<br>official HTE documents</div>
                <div class="left-sub">
                    AI-powered document intelligence for government officers,
                    students and administrators — grounded in official GRs,
                    circulars and brochures.
                </div>
                <div class="feature-list">{feat_html}</div>
                {news_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div style="padding:40px 8px 8px;">
            <div class="auth-title">Welcome back 👋</div>
            <div class="auth-sub">Sign in to access the HTE Knowledge Assistant</div>
        </div>
        """, unsafe_allow_html=True)

        tab_in, tab_reg = st.tabs(["Sign In", "Create Account"])

        with tab_in:
            with st.form("lf", clear_on_submit=False):
                em = st.text_input("Email address", placeholder="you@example.com", key="l_em")
                pw = st.text_input("Password", type="password", placeholder="••••••••", key="l_pw")
                sb = st.form_submit_button("Sign In →", type="primary", use_container_width=True)
            if sb:
                if _is_locked_out():
                    st.error(f"Locked out. Try in {int(st.session_state['auth_lockout_until']-time.time())}s.")
                else:
                    with st.spinner("Signing in…"):
                        ok, msg = login(em, pw)
                    if ok: st.rerun()
                    else: st.error(msg)

        with tab_reg:
            with st.form("rf", clear_on_submit=True):
                r_em  = st.text_input("Email address",   placeholder="you@example.com",                    key="r_em")
                r_pw  = st.text_input("Password",         type="password", placeholder="Min 8 chars · upper · lower · number", key="r_pw")
                r_pw2 = st.text_input("Confirm password", type="password", placeholder="Repeat password",   key="r_pw2")
                r_sb  = st.form_submit_button("Create Account →", type="primary", use_container_width=True)
            if r_sb:
                if r_pw != r_pw2: st.error("Passwords do not match.")
                else:
                    with st.spinner("Creating account…"):
                        ok, msg = register(r_em, r_pw)
                    if ok: st.rerun()
                    else: st.error(msg)

        st.markdown('<div class="auth-footer">Official government knowledge base · No hallucinations<br>Grounded in verified HTE documents</div>', unsafe_allow_html=True)