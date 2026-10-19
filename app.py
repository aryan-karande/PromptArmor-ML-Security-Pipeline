"""
PromptArmor — Enterprise AI Security Gateway & Compute Cost Optimizer
======================================================================
Production-grade commercial AI firewall and telemetry defense platform.

Run with:  streamlit run app.py
"""

import os
import warnings
import streamlit as st

# ── Suppress low-level logging ──────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PromptArmor • Enterprise AI Security Gateway",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom Enterprise Dark-Mode CSS Theme ───────────────────────────────────
st.markdown("""
<style>
    /* ── Base Enterprise Dark Theme ──────────────────────────────── */
    html, body, [class*="css"], .stApp {
        background-color: #0b0f19 !important;
        color: #f9fafb !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    /* ── Typography & Headers ────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        color: #f9fafb !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    p, span, label {
        color: #d1d5db;
    }

    /* ── Top-Level Tabs ──────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #111827 !important;
        padding: 6px 10px;
        border-radius: 12px;
        border: 1px solid #1f2937;
        margin-bottom: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1f2937 !important;
        color: #9ca3af !important;
        border-radius: 8px !important;
        padding: 10px 18px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        border: 1px solid transparent !important;
        transition: all 0.2s ease-in-out;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f9fafb !important;
        background-color: #374151 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: 1px solid #60a5fa !important;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.4) !important;
    }

    /* ── Metric Cards & Surfaces ─────────────────────────────────── */
    div[data-testid="stMetric"] {
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
    }
    
    div[data-testid="stMetricLabel"] p {
        color: #9ca3af !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    
    div[data-testid="stMetricValue"] div {
        color: #f9fafb !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* ── Containers, Expanders & Cards ───────────────────────────── */
    .streamlit-expanderHeader {
        background-color: #1f2937 !important;
        color: #f9fafb !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stExpander"] {
        border: 1px solid #374151 !important;
        border-radius: 10px !important;
        background-color: #111827 !important;
        margin-bottom: 12px;
    }

    /* ── Input Fields & Selectors ────────────────────────────────── */
    textarea, input, .stSelectbox [data-baseweb="select"] {
        background-color: #111827 !important;
        color: #f9fafb !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }
    
    textarea:focus, input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────── */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        box-shadow: 0 6px 18px rgba(59, 130, 246, 0.5) !important;
        transform: translateY(-1px);
    }
    
    .stButton > button[kind="secondary"] {
        background-color: #1f2937 !important;
        color: #d1d5db !important;
        border: 1px solid #374151 !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    /* ── DataFrames & Tables ─────────────────────────────────────── */
    div[data-testid="stDataFrame"] {
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
        overflow: hidden;
    }

    /* ── Alert Overrides ─────────────────────────────────────────── */
    .stAlert {
        border-radius: 8px !important;
        background-color: #1f2937 !important;
        border: 1px solid #374151 !important;
    }

    /* ── Executive Hero Header ───────────────────────────────────── */
    .enterprise-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0b0f19 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .header-branding {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    
    .header-logo {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        width: 52px;
        height: 52px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
    }
    
    .header-title-text h1 {
        margin: 0;
        font-size: 1.85rem;
        color: #f9fafb !important;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .header-title-text p {
        margin: 3px 0 0 0;
        color: #9ca3af;
        font-size: 0.92rem;
    }
    
    .status-pill-row {
        display: flex;
        gap: 10px;
    }
    
    .status-pill {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10b981;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }
</style>
""", unsafe_allow_html=True)

# ── Executive Hero Header ───────────────────────────────────────────────────
st.markdown("""
<div class="enterprise-header">
    <div class="header-branding">
        <div class="header-logo">🛡️</div>
        <div class="header-title-text">
            <h1>PromptArmor <span style="font-size:0.75rem; font-weight:700; color:#3b82f6; background:rgba(59,130,246,0.15); padding:2px 8px; border-radius:10px; border:1px solid rgba(59,130,246,0.3); vertical-align:middle;">ENTERPRISE SAAS</span></h1>
            <p>AI Security Gateway & Compute Optimization Platform • Executive Security Operations Center</p>
        </div>
    </div>
    <div class="status-pill-row">
        <div class="status-pill">
            <span class="status-dot"></span> GATEWAY: ONLINE
        </div>
        <div class="status-pill" style="background:rgba(59,130,246,0.12); border-color:rgba(59,130,246,0.4); color:#60a5fa;">
            <span class="status-dot" style="background:#60a5fa;"></span> THREAT SHIELD: ACTIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top-Level Tab Navigation ────────────────────────────────────────────────
from tabs.tab1_sandbox import render as render_sandbox
from tabs.tab2_threat_intel import render as render_threat_intel
from tabs.tab3_sanitization import render as render_sanitization
from tabs.tab4_guardrails import render as render_guardrails
from tabs.tab5_forecasting import render as render_forecasting

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚀 Live Security Sandbox",
    "🧠 Threat Intelligence Arena",
    "🧹 Data Sanitization & Ingestion Engine",
    "📋 Policy Guardrails Engine",
    "📊 Compute Cost & Latency Forecasting",
])

with tab1:
    render_sandbox()

with tab2:
    render_threat_intel()

with tab3:
    render_sanitization()

with tab4:
    render_guardrails()

with tab5:
    render_forecasting()

# ── Enterprise Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:#6b7280; font-size:0.82rem; letter-spacing:0.02em; padding: 12px 0;'>"
    "PromptArmor Enterprise AI Security Gateway v2.4.0 • Zero-Trust LLM Defense Architecture • "
    "SOC-2 Type II &amp; ISO/IEC 27001 Compliant Invariant Engine"
    "</center>",
    unsafe_allow_html=True,
)
