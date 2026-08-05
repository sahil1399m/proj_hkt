"""
auth.py — HTE Knowledge Assistant Authentication
Enhanced UI: password strength meter, animated states, polished sign-in/sign-up panels
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


# ── Auth state helpers ─────────────────────────────────────────────────────────

def _init_auth_state() -> None:
    defaults = {
        "user_session":        None,
        "auth_failed_attempts": 0,
        "auth_lockout_until":   0.0,
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
    st.session_state["auth_lockout_until"]   = 0.0

def _validate_password(pw: str) -> list[str]:
    errs = []
    if len(pw) < 8:                      errs.append("at least 8 characters")
    if not any(c.isupper() for c in pw): errs.append("one uppercase letter")
    if not any(c.islower() for c in pw): errs.append("one lowercase letter")
    if not any(c.isdigit() for c in pw): errs.append("one number")
    return errs

def _password_strength(pw: str) -> tuple[int, str, str]:
    """Returns (score 0-4, label, color)."""
    score = 0
    if len(pw) >= 8:                      score += 1
    if any(c.isupper() for c in pw):      score += 1
    if any(c.islower() for c in pw):      score += 1
    if any(c.isdigit() for c in pw):      score += 1
    labels = ["", "Weak", "Fair", "Good", "Strong"]
    colors = ["", "#ef4444", "#f59e0b", "#3b82f6", "#22c55e"]
    return score, labels[score] if score else "", colors[score] if score else "#27272a"


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


# ── AUTH PAGE ──────────────────────────────────────────────────────────────────

def render_auth_page() -> None:
    _init_auth_state()

    news_items = []
    try:
        from hte_feed import get_hte_news
        news_items = get_hte_news(max_articles=4)
    except Exception:
        pass

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"] {
    background: #07070a !important;
    color: #e4e4e7 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"] { display: none !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #27272a; border-radius: 4px; }

/* ── LEFT PANEL ── */
.auth-left {
    padding: 48px 44px 40px;
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    position: relative;
    overflow: hidden;
}

/* Ambient gradient blob behind left panel */
.auth-left::before {
    content: '';
    position: absolute;
    top: -120px; left: -80px;
    width: 480px; height: 480px;
    background: radial-gradient(circle, rgba(37,99,235,.13) 0%, transparent 70%);
    pointer-events: none;
}
.auth-left::after {
    content: '';
    position: absolute;
    bottom: -60px; right: 40px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(124,58,237,.10) 0%, transparent 70%);
    pointer-events: none;
}

/* ── BRAND ── */
.brand {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 56px;
    position: relative;
    z-index: 1;
}
.brand-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 0 1px rgba(255,255,255,.08), 0 8px 24px rgba(37,99,235,.3);
}
.brand-name {
    font-size: 16px; font-weight: 700; color: #f4f4f5;
    letter-spacing: -.2px;
}
.brand-sub {
    font-size: 11px; color: #52525b; margin-top: 2px;
    font-weight: 400; letter-spacing: .2px;
}

/* ── HEADLINE ── */
.left-content { position: relative; z-index: 1; flex: 1; }

.left-eyebrow {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 10.5px; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #3b82f6;
    background: rgba(59,130,246,.08);
    border: 1px solid rgba(59,130,246,.18);
    border-radius: 20px; padding: 4px 12px;
    margin-bottom: 20px;
}
.left-eyebrow-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #3b82f6;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .5; transform: scale(.7); }
}

.left-headline {
    font-size: 36px; font-weight: 800; color: #f4f4f5;
    line-height: 1.22; letter-spacing: -.8px; margin-bottom: 16px;
}
.left-headline span {
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.left-sub {
    font-size: 14px; color: #71717a; line-height: 1.75;
    margin-bottom: 40px; max-width: 400px;
}

/* ── FEATURE LIST ── */
.feature-list { display: flex; flex-direction: column; gap: 10px; margin-bottom: 44px; }
.feature-item {
    display: flex; align-items: flex-start; gap: 14px;
    padding: 12px 14px;
    background: rgba(255,255,255,.02);
    border: 1px solid #1c1c1f;
    border-radius: 10px;
    transition: border-color .2s, background .2s;
}
.feature-item:hover {
    border-color: #27272a;
    background: rgba(255,255,255,.035);
}
.feature-icon {
    width: 30px; height: 30px; border-radius: 7px;
    background: #111115; border: 1px solid #27272a;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; flex-shrink: 0; margin-top: 1px;
}
.feature-text { flex: 1; }
.feature-title {
    font-size: 13px; font-weight: 500; color: #d4d4d8;
    line-height: 1.4;
}

/* ── NEWS CARDS ── */
.news-section { position: relative; z-index: 1; }
.news-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 1.2px;
    color: #3f3f46; font-weight: 600; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
}
.news-label::after {
    content: ''; flex: 1; height: 1px; background: #18181b;
}
.news-card {
    background: #0e0e12;
    border: 1px solid #1c1c1f;
    border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;
    cursor: pointer; text-decoration: none; display: block;
    transition: border-color .18s, transform .18s, background .18s;
}
.news-card:hover {
    border-color: #2563eb;
    background: rgba(37,99,235,.04);
    transform: translateX(3px);
    text-decoration: none;
}
.news-card-inner { display: flex; gap: 10px; align-items: flex-start; }
.news-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #2563eb; flex-shrink: 0; margin-top: 5px;
    opacity: .6;
}
.news-src {
    font-size: 9.5px; color: #3f3f46; text-transform: uppercase;
    letter-spacing: .8px; margin-bottom: 3px; font-weight: 600;
}
.news-title { font-size: 12.5px; font-weight: 500; color: #d4d4d8; line-height: 1.45; }
.news-desc  { font-size: 11px; color: #52525b; margin-top: 3px; line-height: 1.5; }

/* ── RIGHT PANEL ── */
.auth-right-wrap {
    background: #0c0c0f;
    border-left: 1px solid #18181b;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 48px 44px;
    position: relative;
    overflow: hidden;
}
.auth-right-wrap::before {
    content: '';
    position: absolute;
    top: -200px; right: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,.07) 0%, transparent 70%);
    pointer-events: none;
}

.auth-header { margin-bottom: 28px; }
.auth-title {
    font-size: 26px; font-weight: 800; color: #f4f4f5;
    letter-spacing: -.4px; margin-bottom: 6px;
}
.auth-sub { font-size: 13.5px; color: #71717a; line-height: 1.55; }

/* ── TRUST BADGES ── */
.trust-row {
    display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 28px;
}
.trust-badge {
    display: flex; align-items: center; gap: 5px;
    font-size: 10.5px; color: #52525b; font-weight: 500;
    background: #111115; border: 1px solid #1c1c1f;
    border-radius: 20px; padding: 4px 10px;
}

/* ── STREAMLIT TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #111115 !important;
    border-radius: 10px !important;
    border: 1px solid #1c1c1f !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #71717a !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-radius: 7px !important;
    padding: 8px 22px !important;
    transition: color .15s !important;
}
.stTabs [aria-selected="true"] {
    background: #1c1c21 !important;
    color: #f4f4f5 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,.4) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 22px 0 0 !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ── INPUTS ── */
[data-testid="stTextInput"] label {
    color: #71717a !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
    letter-spacing: .4px !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}
[data-testid="stTextInput"] input {
    background: #111115 !important;
    border: 1px solid #27272a !important;
    border-radius: 9px !important;
    color: #e4e4e7 !important;
    font-size: 14px !important;
    padding: 12px 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color .15s, box-shadow .15s !important;
}
[data-testid="stTextInput"] input::placeholder { color: #3f3f46 !important; }
[data-testid="stTextInput"] input:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.12) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input:hover:not(:focus) {
    border-color: #3f3f46 !important;
}

/* ── BUTTONS ── */
.stButton > button {
    border-radius: 9px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: .1px !important;
    transition: all .18s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    border: none !important;
    color: white !important;
    font-size: 14px !important;
    padding: 12px 20px !important;
    box-shadow: 0 4px 14px rgba(37,99,235,.25) !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: .92 !important;
    box-shadow: 0 6px 20px rgba(37,99,235,.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(37,99,235,.2) !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #27272a !important;
    color: #71717a !important;
    font-size: 13px !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #111115 !important;
    color: #d4d4d8 !important;
    border-color: #3f3f46 !important;
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 9px !important;
    font-size: 13px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAlert"][data-baseweb="notification"] {
    border: none !important;
}

/* ── PASSWORD STRENGTH ── */
.pw-strength-wrap { margin-top: -8px; margin-bottom: 12px; }
.pw-strength-bar-bg {
    height: 3px; background: #1c1c1f; border-radius: 2px;
    margin-bottom: 5px; overflow: hidden;
}
.pw-strength-bar { height: 100%; border-radius: 2px; transition: width .3s, background .3s; }
.pw-strength-label { font-size: 11px; font-weight: 500; }

/* ── DIVIDER ── */
.form-divider {
    display: flex; align-items: center; gap: 12px;
    margin: 20px 0; color: #3f3f46; font-size: 11px;
}
.form-divider::before, .form-divider::after {
    content: ''; flex: 1; height: 1px; background: #1c1c1f;
}

/* ── FOOTER ── */
.auth-footer {
    font-size: 11px; color: #3f3f46; text-align: center;
    margin-top: 28px; line-height: 1.8;
    border-top: 1px solid #111115; padding-top: 20px;
}
.auth-footer a { color: #52525b; text-decoration: none; }
.auth-footer a:hover { color: #71717a; }

/* ── STAT ROW ── */
.stat-row {
    display: flex; gap: 0; margin-bottom: 44px;
    border: 1px solid #1c1c1f; border-radius: 12px; overflow: hidden;
}
.stat-item {
    flex: 1; padding: 16px 18px; position: relative;
}
.stat-item + .stat-item { border-left: 1px solid #1c1c1f; }
.stat-num {
    font-size: 22px; font-weight: 800; color: #f4f4f5;
    letter-spacing: -.5px; line-height: 1;
}
.stat-num span {
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.stat-label { font-size: 11px; color: #52525b; margin-top: 4px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

    # ── Build news HTML ──────────────────────────────────────────────────────
    news_html = ""
    if news_items:
        cards = []
        for a in news_items[:4]:
            t = (a.get("title", "")[:72] + "…") if len(a.get("title", "")) > 72 else a.get("title", "")
            d = (a.get("description", "")[:100] + "…") if len(a.get("description", "")) > 100 else a.get("description", "")
            u = a.get("url", "#")
            s = a.get("source", "Source")
            cards.append(f'''
<a href="{u}" target="_blank" class="news-card">
    <div class="news-card-inner">
        <div class="news-dot"></div>
        <div>
            <div class="news-src">{s}</div>
            <div class="news-title">{t}</div>
            {"" if not d else f'<div class="news-desc">{d}</div>'}
        </div>
    </div>
</a>''')
        news_html = f'''
<div class="news-section">
    <div class="news-label">📰 Latest HTE Updates</div>
    {"".join(cards)}
</div>'''

    features = [
        ("🔍", "Semantic search across 2,000+ official HTE documents"),
        ("📊", "Table-aware retrieval for fees, scholarships & seat matrices"),
        ("🌐", "Live web search from official Maharashtra government sites"),
        ("🗣️",  "Answers in English, Marathi and Hindi"),
        ("📄", "Source citations with page numbers for every answer"),
    ]
    feat_html = "".join(
        f'''<div class="feature-item">
            <div class="feature-icon">{icon}</div>
            <div class="feature-text">
                <div class="feature-title">{text}</div>
            </div>
        </div>'''
        for icon, text in features
    )

    col_left, col_right = st.columns([1.15, 1])

    # ── LEFT COLUMN ─────────────────────────────────────────────────────────
    with col_left:
        st.markdown(f"""
<div class="auth-left">
    <div class="brand">
        <div class="brand-icon">🏛️</div>
        <div>
            <div class="brand-name">HTE Knowledge Assistant</div>
            <div class="brand-sub">Maharashtra Higher &amp; Technical Education</div>
        </div>
    </div>

    <div class="left-content">
        <div class="left-eyebrow">
            <div class="left-eyebrow-dot"></div>
            AI-Powered · Official Sources Only
        </div>

        <div class="left-headline">
            Instant answers from<br><span>official HTE documents</span>
        </div>

        <div class="left-sub">
            AI document intelligence for government officers, students and
            administrators — grounded in verified GRs, circulars and brochures.
            No hallucinations. Always cited.
        </div>

        <div class="stat-row">
            <div class="stat-item">
                <div class="stat-num"><span>2,000+</span></div>
                <div class="stat-label">Official documents</div>
            </div>
            <div class="stat-item">
                <div class="stat-num"><span>3</span></div>
                <div class="stat-label">Languages supported</div>
            </div>
            <div class="stat-item">
                <div class="stat-num"><span>100%</span></div>
                <div class="stat-label">Source-cited answers</div>
            </div>
        </div>

        <div class="feature-list">{feat_html}</div>

        {news_html}
    </div>
</div>
""", unsafe_allow_html=True)

    # ── RIGHT COLUMN ─────────────────────────────────────────────────────────
    with col_right:
        st.markdown("""
<div class="auth-right-wrap">
    <div class="auth-header">
        <div class="auth-title">Welcome back 👋</div>
        <div class="auth-sub">Sign in to access the HTE Knowledge Assistant</div>
    </div>
    <div class="trust-row">
        <span class="trust-badge">🔒 Secure · Supabase Auth</span>
        <span class="trust-badge">✅ Verified government data</span>
        <span class="trust-badge">🏛️ Maharashtra Govt</span>
    </div>
</div>
""", unsafe_allow_html=True)

        tab_in, tab_reg = st.tabs(["  Sign In  ", "  Create Account  "])

        # ── SIGN IN TAB ──────────────────────────────────────────────────
        with tab_in:
            with st.form("lf", clear_on_submit=False):
                em = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                    key="l_em",
                )
                pw = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    key="l_pw",
                )
                sb = st.form_submit_button(
                    "Sign In →",
                    type="primary",
                    use_container_width=True,
                )

            if sb:
                if _is_locked_out():
                    remaining = int(st.session_state["auth_lockout_until"] - time.time())
                    st.error(f"🔒 Account locked. Try again in {remaining}s.")
                else:
                    with st.spinner("Verifying credentials…"):
                        ok, msg = login(em, pw)
                    if ok:
                        st.rerun()
                    else:
                        attempts_left = MAX_FAILED_ATTEMPTS - st.session_state["auth_failed_attempts"]
                        st.error(f"⚠️ {msg}" + (f"  ·  {attempts_left} attempt(s) remaining" if attempts_left < MAX_FAILED_ATTEMPTS else ""))

        # ── CREATE ACCOUNT TAB ───────────────────────────────────────────
        with tab_reg:
            r_pw_live = st.text_input(
                "Password",
                type="password",
                placeholder="Min 8 chars · uppercase · number",
                key="r_pw_preview",
                label_visibility="collapsed",
            )

            # Live password strength indicator
            if r_pw_live:
                score, label, color = _password_strength(r_pw_live)
                bar_pct = score * 25
                st.markdown(f"""
<div class="pw-strength-wrap">
    <div class="pw-strength-bar-bg">
        <div class="pw-strength-bar" style="width:{bar_pct}%;background:{color};"></div>
    </div>
    <div class="pw-strength-label" style="color:{color};">{label}</div>
</div>
""", unsafe_allow_html=True)

            with st.form("rf", clear_on_submit=True):
                r_em  = st.text_input(
                    "Email address",
                    placeholder="you@example.com",
                    key="r_em",
                )
                r_pw  = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Min 8 chars · uppercase · number",
                    key="r_pw",
                )
                r_pw2 = st.text_input(
                    "Confirm password",
                    type="password",
                    placeholder="Repeat your password",
                    key="r_pw2",
                )
                r_sb  = st.form_submit_button(
                    "Create Account →",
                    type="primary",
                    use_container_width=True,
                )

            if r_sb:
                if r_pw != r_pw2:
                    st.error("⚠️ Passwords do not match.")
                else:
                    with st.spinner("Creating your account…"):
                        ok, msg = register(r_em, r_pw)
                    if ok:
                        st.rerun()
                    else:
                        st.error(f"⚠️ {msg}")

        st.markdown("""
<div class="auth-footer">
    🔐 Your data is encrypted and stored securely<br>
    Official government knowledge base · No hallucinations<br>
    Grounded in verified HTE documents only
</div>
""", unsafe_allow_html=True)
