"""
Tab 4 — Policy Guardrails Engine
=================================
Symbolic access rule version space boundary engine (Most Specific S and
Most General G boundaries), contradictory policy collapse detection, and
interactive enterprise security tier compliance verification.
"""

import streamlit as st
import pandas as pd
from modules.concept_learning import (
    generate_policy_examples,
    find_s,
    candidate_elimination,
    format_hypothesis,
    check_version_space_collapse,
)


def render():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0; color: #f9fafb; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em;">
            📋 Policy Guardrails Engine
        </h2>
        <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 0.95rem;">
            Symbolic Version Space Boundaries • Contradictory Policy Collapse • Enterprise Compliance Verification
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1: Version Space Dual-Boundary Engine ────────────────────────
    st.markdown("""
    <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:16px; margin-bottom:16px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-weight:700; color:#f9fafb; font-size:1.05rem;">Dual-Boundary Symbolic Guardrails (S & G Boundaries)</span>
                <p style="margin:4px 0 0 0; color:#9ca3af; font-size:0.85rem;">
                    Maintains the exact Version Space of all policy hypotheses consistent with positive and negative security audit logs.
                </p>
            </div>
            <div style="background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.3); border-radius:20px; padding:4px 14px; color:#60a5fa; font-size:0.8rem; font-weight:600;">
                SYMBOLIC REASONING: CANDIDATE ELIMINATION
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ctrl_col1, ctrl_col2 = st.columns([1, 2])
    with ctrl_col1:
        inject_noise = st.toggle(
            "💥 Inject Contradictory Security Policy (Noise)",
            value=False,
            help="Simulates an admin error where identical attributes are labeled both allowed and denied.",
            key="guardrails_noise_toggle"
        )
    with ctrl_col2:
        if inject_noise:
            st.markdown("<p style='color:#ef4444; font-size:0.85rem; margin-top:6px;'>⚠️ Contradictory policy rule injected: demonstrates Version Space Collapse requiring human SecOps intervention.</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#10b981; font-size:0.85rem; margin-top:6px;'>✅ Clean audit log: consistent Version Space convergence guaranteed.</p>", unsafe_allow_html=True)

    # Compute boundaries
    examples = generate_policy_examples(inject_noise=inject_noise)
    final_S, final_G, S_hist, G_hist = candidate_elimination(examples)
    is_collapsed = check_version_space_collapse(final_S, final_G)

    # Version Space Collapse Alert
    if is_collapsed:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.16) 0%, rgba(185, 28, 28, 0.08) 100%);
                    border: 1px solid rgba(239, 68, 68, 0.6);
                    border-radius: 10px; padding: 16px 20px; margin-bottom: 18px;
                    box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);">
            <div style="color: #ef4444; font-weight: 700; font-size: 1.05rem;">
                🚨 CRITICAL WARNING: VERSION SPACE COLLAPSED
            </div>
            <div style="color: #fca5a5; font-size: 0.9rem; margin-top: 4px;">
                The Most Specific (S) and Most General (G) boundaries have crossed: <strong>S ∩ G = ∅</strong>.<br>
                Contradictory policy entries detected in access audit logs. No single deterministic symbolic policy can satisfy all historical rules.<br>
                <strong>SecOps Action Required:</strong> Human governance audit triggered. Traffic falls back to Deep Neural Threat Filter.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.08) 100%);
                    border: 1px solid rgba(16, 185, 129, 0.5);
                    border-radius: 10px; padding: 14px 20px; margin-bottom: 18px;">
            <div style="color: #10b981; font-weight: 700; font-size: 1rem;">
                ✅ VERSION SPACE CONSISTENT & CONVERGED
            </div>
            <div style="color: #d1fae5; font-size: 0.85rem; margin-top: 3px;">
                Deterministic policy boundaries successfully established. Invariants active for edge enforcement.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Boundary Cards
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px;">
            <div style="color:#60a5fa; font-weight:700; font-size:0.95rem;">🎯 Most Specific Boundary (S) — Strictest Whitelist</div>
            <div style="color:#9ca3af; font-size:0.8rem; margin:3px 0 10px 0;">Only allows requests matching the learned minimum general invariant:</div>
        """, unsafe_allow_html=True)
        s_display = format_hypothesis(final_S) if final_S else "∅ (Collapsed)"
        st.code(s_display, language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    with b_col2:
        st.markdown("""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px;">
            <div style="color:#10b981; font-weight:700; font-size:0.95rem;">🌐 Most General Boundary (G) — Permissive Outer Envelope</div>
            <div style="color:#9ca3af; font-size:0.8rem; margin:3px 0 10px 0;">Maximal permissible hypothesis space consistent with negative logs:</div>
        """, unsafe_allow_html=True)
        if final_G:
            g_lines = "\n".join([format_hypothesis(g) for g in final_G])
            st.code(g_lines, language="text")
        else:
            st.code("∅ (Collapsed)", language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    # Step-by-Step Version Space Evolution
    with st.expander("📈 Step-by-Step Policy Boundary Convergence Evolution"):
        st.markdown("<p style='color:#9ca3af; font-size:0.85rem;'>Displays how the S boundary generalizes and the G boundary specializes upon ingesting each audit record.</p>", unsafe_allow_html=True)
        evo_data = []
        max_steps = max(len(S_hist), len(G_hist))
        for step in range(max_steps):
            s_str = format_hypothesis(S_hist[step]) if step < len(S_hist) else "—"
            g_str = " | ".join([format_hypothesis(g) for g in G_hist[step]]) if step < len(G_hist) and G_hist[step] else "∅"
            evo_data.append({
                "Step": step + 1,
                "S Boundary (Specific)": s_str,
                "G Boundary (General)": g_str,
            })
        st.dataframe(pd.DataFrame(evo_data), use_container_width=True, hide_index=True)

    # ── Section 2: Interactive Compliance Status Monitor ─────────────────────
    st.markdown("""
    <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:16px; margin-top:24px; margin-bottom:16px;">
        <span style="font-weight:700; color:#f9fafb; font-size:1.05rem;">Interactive Compliance Status Monitor</span>
        <p style="margin:4px 0 0 0; color:#9ca3af; font-size:0.85rem;">
            Test arbitrary client configurations against active enterprise security guardrails.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tier_presets = {
        "Enterprise Platinum (Tier-1)": {
            "tier": "Enterprise",
            "intent": "Analysis",
            "code": "Yes",
            "auth": "Authenticated",
            "desc": "Full code execution privileges for verified enterprise accounts.",
        },
        "FinTech High-Security (PCI/GLBA)": {
            "tier": "Enterprise",
            "intent": "Question",
            "code": "No",
            "auth": "Authenticated",
            "desc": "Strict financial compliance: no arbitrary code execution.",
        },
        "Healthcare HIPAA Strict": {
            "tier": "Pro",
            "intent": "Analysis",
            "code": "No",
            "auth": "Authenticated",
            "desc": "Read-only medical telemetry: code execution blocked.",
        },
        "Developer API Pro": {
            "tier": "Pro",
            "intent": "Instruction",
            "code": "Yes",
            "auth": "Authenticated",
            "desc": "Standard developer sandbox with authenticated API access.",
        },
        "Public Sandbox (Free API)": {
            "tier": "Free",
            "intent": "Question",
            "code": "No",
            "auth": "Anonymous",
            "desc": "Unauthenticated public questions: strict sandbox guardrails.",
        },
    }

    preset_name = st.selectbox("Load Enterprise Compliance Profile Preset:", list(tier_presets.keys()), key="compliance_preset_select")
    current_cfg = tier_presets[preset_name]
    st.caption(f"Profile Description: {current_cfg['desc']}")

    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
    with c_col1:
        test_tier = st.selectbox("Client Tier", ["Free", "Pro", "Enterprise"], index=["Free", "Pro", "Enterprise"].index(current_cfg["tier"]), key="c_tier")
    with c_col2:
        test_intent = st.selectbox("Intent Type", ["Question", "Instruction", "Creative", "Analysis"], index=["Question", "Instruction", "Creative", "Analysis"].index(current_cfg["intent"]), key="c_intent")
    with c_col3:
        test_code = st.selectbox("Code Execution", ["No", "Yes"], index=["No", "Yes"].index(current_cfg["code"]), key="c_code")
    with c_col4:
        test_auth = st.selectbox("Auth State", ["Authenticated", "Anonymous"], index=["Authenticated", "Anonymous"].index(current_cfg["auth"]), key="c_auth")

    # Evaluate against boundaries
    def _matches_hyp(hyp, instance):
        if hyp is None or len(hyp) != 4:
            return False
        for h_val, i_val in zip(hyp, instance):
            if h_val != "?" and h_val != i_val:
                return False
        return True

    instance = [test_tier, test_intent, test_code, test_auth]

    s_match = _matches_hyp(final_S, instance) if final_S else False
    g_match = any(_matches_hyp(g, instance) for g in final_G) if final_G else False

    st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#9ca3af; margin-top:14px;'>Compliance Evaluation Verdict</p>", unsafe_allow_html=True)
    
    if is_collapsed:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.4); border-radius:8px; padding:14px;">
            <span style="color:#ef4444; font-weight:700;">STATUS: GUARDRAIL DISPUTED</span> • Version Space collapsed due to conflicting administrative rules. Manual SecOps approval required.
        </div>
        """, unsafe_allow_html=True)
    elif s_match and g_match:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.12); border:1px solid rgba(16,185,129,0.4); border-radius:8px; padding:14px;">
            <span style="color:#10b981; font-weight:700;">STATUS: FULLY COMPLIANT (PASSED)</span> • Satisfies both Most Specific (S) and Most General (G) boundaries. Permitted without restriction.
        </div>
        """, unsafe_allow_html=True)
    elif g_match and not s_match:
        st.markdown("""
        <div style="background:rgba(59,130,246,0.12); border:1px solid rgba(59,130,246,0.4); border-radius:8px; padding:14px;">
            <span style="color:#60a5fa; font-weight:700;">STATUS: CONDITIONALLY PERMITTED</span> • Falls within general security envelope G, but outside strict whitelist S. Monitored execution enforced.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(239,68,68,0.12); border:1px solid rgba(239,68,68,0.4); border-radius:8px; padding:14px;">
            <span style="color:#ef4444; font-weight:700;">STATUS: NON-COMPLIANT (BLOCKED)</span> • Violates general security boundary G. Inbound request denied by policy guardrail.
        </div>
        """, unsafe_allow_html=True)

    # Invariant evaluation breakdown
    st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#9ca3af; margin-top:14px;'>Boundary Invariant Verification Audit</p>", unsafe_allow_html=True)
    audit_rows = [
        {"Attribute": "Client Tier", "Tested Value": test_tier, "S Constraint": final_S[0] if final_S else "∅", "Verdict": "MATCH" if (final_S and (final_S[0] == "?" or final_S[0] == test_tier)) else "RESTRICTED"},
        {"Attribute": "Intent Type", "Tested Value": test_intent, "S Constraint": final_S[1] if final_S else "∅", "Verdict": "MATCH" if (final_S and (final_S[1] == "?" or final_S[1] == test_intent)) else "RESTRICTED"},
        {"Attribute": "Code Execution", "Tested Value": test_code, "S Constraint": final_S[2] if final_S else "∅", "Verdict": "MATCH" if (final_S and (final_S[2] == "?" or final_S[2] == test_code)) else "RESTRICTED"},
        {"Attribute": "Auth State", "Tested Value": test_auth, "S Constraint": final_S[3] if final_S else "∅", "Verdict": "MATCH" if (final_S and (final_S[3] == "?" or final_S[3] == test_auth)) else "RESTRICTED"},
    ]
    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
