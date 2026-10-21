"""
Tab 3 — Executive Landing & About Page
=======================================
High-end dark mode executive landing page explaining continuous risk
intelligence, LLM firewall infrastructure, with a prominent Launch CTA button.
"""

import streamlit as st


def render(on_launch_callback=None):
    # ── Hero Section with Minimalist Luxury Aesthetic ────────────────────────
    st.markdown("""
    <div style="text-align: center; padding: 48px 20px 32px 20px; max-width: 900px; margin: 0 auto;">
        <div style="display: inline-block; padding: 6px 16px; border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 24px; background: rgba(255, 255, 255, 0.03); color: #94a3b8; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 20px;">
            Continuous Risk Intelligence & LLM Security
        </div>
        <h1 style="font-size: 3.2rem; font-weight: 800; color: #ffffff; letter-spacing: -0.03em; margin: 0 0 16px 0; line-height: 1.15;">
            Enterprise AI Defense Infrastructure
        </h1>
        <p style="font-size: 1.15rem; color: #94a3b8; max-width: 720px; margin: 0 auto 36px auto; line-height: 1.6; font-weight: 400;">
            PromptArmor inspects prompt payloads in real-time, blocking adversarial jailbreaks, enforcing corporate access invariants, and forecasting compute cost before model invocation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Prominent Centered Launch CTA Button ─────────────────────────────────
    c_left, c_center, c_right = st.columns([1.2, 2.2, 1.2])
    with c_center:
        launch_clicked = st.button(
            "Launch Live Prompt Tester",
            type="primary",
            use_container_width=True,
            key="btn_launch_prompt_tester",
        )
        if launch_clicked and on_launch_callback:
            on_launch_callback()

    st.markdown("<div style='margin-bottom: 50px;'></div>", unsafe_allow_html=True)

    # ── Core Architecture Pillars Grid ───────────────────────────────────────
    st.markdown("<p style='font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; margin-bottom:14px; text-align:center;'>Core Defense Architecture</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #111111; border: 1px solid #222222; border-radius: 14px; padding: 28px; height: 100%; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">01 / Telemetry</div>
            <div style="font-weight: 700; font-size: 1.2rem; color: #ffffff; margin-bottom: 10px;">
                Structural Text Invariant Engine
            </div>
            <p style="margin: 0; font-size: 0.92rem; color: #94a3b8; line-height: 1.6;">
                Analyzes 7 mathematical text metrics in under 2 milliseconds: override keyword density, symbol distribution ratios, Shannon character entropy, uppercase shouting bursts, token count, word complexity, and imperative verbs.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #111111; border: 1px solid #222222; border-radius: 14px; padding: 28px; height: 100%; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">02 / Neural Defense</div>
            <div style="font-weight: 700; font-size: 1.2rem; color: #ffffff; margin-bottom: 10px;">
                Zero-Day Jailbreak Interception
            </div>
            <p style="margin: 0; font-size: 0.92rem; color: #94a3b8; line-height: 1.6;">
                Multi-layer deep neural networks backed by deterministic firewall rule paths classify prompts into Safe, Complex Query, or Adversarial Jailbreak. Hostile injections are neutralized before reaching foundation models.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 18px;'></div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div style="background: #111111; border: 1px solid #222222; border-radius: 14px; padding: 28px; height: 100%; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">03 / Financial Optimization</div>
            <div style="font-weight: 700; font-size: 1.2rem; color: #ffffff; margin-bottom: 10px;">
                Compute Cost & Latency Forecasting
            </div>
            <p style="margin: 0; font-size: 0.92rem; color: #94a3b8; line-height: 1.6;">
                Regularized regression models evaluate token consumption, expected execution latency, and GPU operational expenditures prior to invocation, enabling automated smart routing to cost-effective accelerator pools.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div style="background: #111111; border: 1px solid #222222; border-radius: 14px; padding: 28px; height: 100%; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;">04 / Governance</div>
            <div style="font-weight: 700; font-size: 1.2rem; color: #ffffff; margin-bottom: 10px;">
                Deterministic Compliance & PDF Audit
            </div>
            <p style="margin: 0; font-size: 0.92rem; color: #94a3b8; line-height: 1.6;">
                Provides automated cryptographic audit logging, continuous invariant verification, and instant export of formal SOC-2 Type II and ISO/IEC 27001 executive PDF compliance certificates.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Enterprise Operational Benchmarks ────────────────────────────────────
    st.markdown("<div style='margin-bottom: 36px;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: #111111; border: 1px solid #222222; border-radius: 14px; padding: 26px 32px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 24px; text-align: center;">
            <div>
                <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Inspection Latency</div>
                <div style="font-size: 1.6rem; color: #ffffff; font-weight: 800; margin-top: 4px;">&lt; 2.1 ms</div>
            </div>
            <div>
                <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Telemetry Buffer</div>
                <div style="font-size: 1.6rem; color: #ffffff; font-weight: 800; margin-top: 4px;">1,500 Prompts</div>
            </div>
            <div>
                <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Threat Accuracy</div>
                <div style="font-size: 1.6rem; color: #ffffff; font-weight: 800; margin-top: 4px;">93.3%</div>
            </div>
            <div>
                <div style="font-size: 0.78rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;">Compliance Standard</div>
                <div style="font-size: 1.6rem; color: #ffffff; font-weight: 800; margin-top: 4px;">SOC-2 / ISO</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
