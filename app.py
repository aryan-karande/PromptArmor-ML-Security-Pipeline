"""
PromptArmor — Enterprise AI Security Gateway & Compute Cost Optimizer
======================================================================
Production-grade enterprise AI firewall & LLM risk intelligence platform.
High-end dark mode executive aesthetic: sleek #0a0a0a background, subtle grid lines,
crisp typography, custom dark armor logo branding, and landing-page first flow.

Run with:  streamlit run app.py
"""

import os
import base64
import warnings
import streamlit as st

# ── Suppress low-level logging ──────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PromptArmor • Enterprise AI Security Gateway",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load Logo Asset as Base64 ───────────────────────────────────────────────
def get_logo_base64() -> tuple[str, str]:
    candidates = [
        (os.path.join(os.path.dirname(__file__), "assets", "logo_transparent.png"), "image/png"),
        (os.path.join(os.path.dirname(__file__), "assets", "logo.png"), "image/png"),
        (os.path.join(os.path.dirname(__file__), "assets", "logo.jpg"), "image/jpeg"),
    ]
    for path, mime in candidates:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8"), mime
            except Exception:
                pass
    return "", "image/png"

LOGO_B64, LOGO_MIME = get_logo_base64()

# ── Custom High-End Dark Enterprise Cybersecurity Styling ───────────────────
st.markdown("""
<style>
    /* ── Hide Streamlit Default Headers, Footers & Watermark ─────── */
    #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    
    /* ── Lift Layout & Maximize Screen Coverage ─────────────────── */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 96% !important;
    }
    
    /* ── Main App Canvas (Deep Dark #0a0a0a with subtle grid) ───── */
    html, body, [class*="css"], .stApp {
        background-color: #0a0a0a !important;
        background-image: 
            linear-gradient(to right, rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255, 255, 255, 0.02) 1px, transparent 1px) !important;
        background-size: 40px 40px !important;
        color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* ── Typography & Headings ───────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em;
    }
    
    p, span, label {
        color: #94a3b8;
    }

    /* ── Top-Level Navigation Tabs (Dark Minimalist Architecture) ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111111 !important;
        padding: 6px 8px;
        border-radius: 12px;
        border: 1px solid #222222;
        margin-bottom: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94a3b8 !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border: 1px solid transparent !important;
        transition: all 0.15s ease-in-out;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background-color: #1a1a1a !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #ffffff !important;
        color: #0a0a0a !important;
        border: 1px solid #ffffff !important;
        box-shadow: 0 4px 18px rgba(255, 255, 255, 0.15) !important;
    }

    /* ── Metric Cards & Surfaces ─────────────────────────────────── */
    div[data-testid="stMetric"] {
        background-color: #111111 !important;
        border: 1px solid #222222 !important;
        border-radius: 12px !important;
        padding: 18px 22px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
    }
    
    div[data-testid="stMetricLabel"] p {
        color: #64748b !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }
    
    div[data-testid="stMetricValue"] div {
        color: #ffffff !important;
        font-size: 1.65rem !important;
        font-weight: 800 !important;
    }

    /* ── Input Fields & Selectors ────────────────────────────────── */
    textarea, input, .stSelectbox [data-baseweb="select"] {
        background-color: #111111 !important;
        color: #f8fafc !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }
    
    textarea:focus, input:focus {
        border-color: #ffffff !important;
        box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.2) !important;
    }

    /* ── Buttons (High-End White Glow & Outline) ─────────────────── */
    .stButton > button {
        border-radius: 9px !important;
        font-weight: 800 !important;
        padding: 12px 24px !important;
        transition: all 0.15s ease-in-out !important;
    }
    
    .stButton > button[kind="primary"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ffffff !important;
        box-shadow: 0 4px 14px rgba(255, 255, 255, 0.2) !important;
    }

    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div {
        color: #000000 !important;
        font-weight: 800 !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #f1f5f9 !important;
        color: #000000 !important;
        box-shadow: 0 6px 20px rgba(255, 255, 255, 0.35) !important;
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"]:hover p,
    .stButton > button[kind="primary"]:hover span,
    .stButton > button[kind="primary"]:hover div {
        color: #000000 !important;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: #141414 !important;
        color: #e2e8f0 !important;
        border: 1px solid #2a2a2a !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        border-color: #ffffff !important;
        color: #ffffff !important;
        background-color: #1f1f1f !important;
    }

    /* ── DataFrames & Tables ─────────────────────────────────────── */
    div[data-testid="stDataFrame"] {
        border: 1px solid #222222 !important;
        border-radius: 10px !important;
        overflow: hidden;
        background-color: #111111 !important;
    }

    /* ── Executive Top Header ────────────────────────────────────── */
    .enterprise-header {
        background: #0d0d0d;
        border: 1px solid #222222;
        border-radius: 14px;
        padding: 18px 28px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.6);
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 18px;
    }
    
    .header-logo-container {
        width: 48px;
        height: 48px;
        border-radius: 10px;
        background: transparent;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }
    
    .header-logo-img {
        width: 44px;
        height: 44px;
        object-fit: contain;
        background: transparent !important;
    }
    
    .header-branding h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 900;
        color: #ffffff !important;
        letter-spacing: -0.035em;
        line-height: 1.1;
    }
    
    .header-branding p {
        margin: 3px 0 0 0;
        color: #64748b;
        font-size: 0.92rem;
        font-weight: 500;
    }
    
    .status-pill {
        font-size: 0.82rem;
        font-weight: 700;
        color: #e2e8f0;
        background: #161616;
        border: 1px solid #2a2a2a;
        padding: 8px 18px;
        border-radius: 20px;
        letter-spacing: 0.04em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }
</style>
""", unsafe_allow_html=True)

# ── Executive Top Header with Armor Logo ────────────────────────────────────
logo_html = f'<div class="header-logo-container"><img src="data:{LOGO_MIME};base64,{LOGO_B64}" class="header-logo-img" alt="PromptArmor Logo" /></div>' if LOGO_B64 else ''

st.markdown(f"""
<div class="enterprise-header">
    <div class="header-left">
        {logo_html}
        <div class="header-branding">
            <h1>PromptArmor</h1>
            <p>Enterprise AI Security Gateway & Continuous LLM Risk Intelligence</p>
        </div>
    </div>
    <div class="status-pill">
        <span class="status-dot"></span>
        System Status: Active | V-2.4
    </div>
</div>
""", unsafe_allow_html=True)

# ── Navigation State Management ─────────────────────────────────────────────
# Default active view is View 1: Executive Landing & About Page
if "page" not in st.session_state:
    st.session_state["page"] = "landing"

def nav_to_sandbox():
    st.session_state["page"] = "sandbox"
    st.rerun()

def nav_to_landing():
    st.session_state["page"] = "landing"
    st.rerun()

# ── Module Imports ──────────────────────────────────────────────────────────
from tabs.tab3_about import render as render_about
from tabs.tab1_sandbox import render as render_sandbox
from tabs.tab2_report import render as render_report

# ── Stateful Routing ────────────────────────────────────────────────────────
if st.session_state["page"] == "landing":
    # View 1: Executive Landing & About Page
    render_about(on_launch_callback=nav_to_sandbox)
else:
    # View 2: Live Security Firewall Sandbox & PDF Report Generator
    col_back, col_spacer = st.columns([2.5, 7.5])
    with col_back:
        if st.button("← Back to Home / Overview", key="btn_back_home", type="secondary", use_container_width=True):
            nav_to_landing()
            
    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)
    
    tab_sandbox, tab_audit = st.tabs([
        "Live Firewall Sandbox",
        "Executive Audit & PDF Report",
    ])
    
    with tab_sandbox:
        render_sandbox()
        
    with tab_audit:
        render_report()

# ── Minimalist Clean Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#475569; font-size:0.82rem; letter-spacing:0.04em; padding: 18px 0;'>"
    "PromptArmor Enterprise Defense • Continuous Risk Intelligence • SOC-2 Type II &amp; ISO/IEC 27001 Certified"
    "</center>",
    unsafe_allow_html=True,
)
