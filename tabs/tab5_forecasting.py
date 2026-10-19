"""
Tab 5 — Compute Cost & Latency Forecasting
===========================================
Regularized regression performance (OLS, Ridge L2, Lasso L1), live R² and error
metrics, Lasso coefficient shrinkage bar chart, actual vs predicted scatter plots,
and interactive workload forecasting simulator.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modules.model_service import get_security_engines
from modules.feature_extraction import FEATURE_NAMES
from modules.regression import (
    get_lasso_feature_importance,
    get_ridge_feature_importance,
)

plt.style.use("dark_background")


def _actual_vs_predicted_plot(y_true, y_pred, title, unit):
    """Render scatter plot of actual vs predicted values with ideal y=x reference."""
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color="#3b82f6", edgecolors="none")
    mn = min(float(np.min(y_true)), float(np.min(y_pred)))
    mx = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Ideal (y=x)")
    ax.set_xlabel(f"Actual Ground Truth ({unit})", color="#9ca3af", fontsize=8)
    ax.set_ylabel(f"Model Predicted ({unit})", color="#9ca3af", fontsize=8)
    ax.set_title(title, color="#f9fafb", fontsize=9, fontweight="bold")
    ax.tick_params(colors="#9ca3af", labelsize=7)
    ax.spines["bottom"].set_color("#374151")
    ax.spines["left"].set_color("#374151")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#1f2937", edgecolor="#374151", fontsize=8)
    fig.tight_layout()
    return fig


def render():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0; color: #f9fafb; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em;">
            📊 Compute Cost & Latency Forecasting
        </h2>
        <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 0.95rem;">
            Regularized Regression Performance • Lasso L1 Sparsity Shrinkage • Workload Simulator
        </p>
    </div>
    """, unsafe_allow_html=True)

    engines = get_security_engines()

    # Target Selector
    st.markdown("""
    <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px; margin-bottom:18px;">
        <span style="font-weight:700; color:#f9fafb; font-size:0.95rem;">Select Workload Prediction Target:</span>
    </div>
    """, unsafe_allow_html=True)

    target_choice = st.radio(
        "Prediction Target:",
        ["Response Latency (ms)", "Token Compute Cost ($)"],
        horizontal=True,
        label_visibility="collapsed",
        key="forecasting_target_radio"
    )

    is_latency = "Latency" in target_choice
    reg_data = engines["latency_reg"] if is_latency else engines["cost_reg"]
    unit = "ms" if is_latency else "$"
    y_test = engines["test_df"]["response_latency_ms"].values if is_latency else engines["test_df"]["token_cost"].values

    # ── Section 1: Regularized Regression Benchmark ──────────────────────────
    st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#9ca3af;'>Regression Performance & Hyperparameter Tuning (5-Fold CV)</p>", unsafe_allow_html=True)

    comp_df = reg_data["comparison"].copy()
    st.dataframe(
        comp_df.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R2": "{:.4f}"
        }).highlight_min(subset=["MAE", "RMSE"], color="#10b981")
         .highlight_max(subset=["R2"], color="#10b981"),
        use_container_width=True,
        hide_index=True,
    )

    # Metric Cards
    c1, c2, c3 = st.columns(3)
    with c1:
        ols_m = reg_data["ols"]["metrics"]
        st.markdown(f"""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#9ca3af; font-size:0.75rem; text-transform:uppercase; font-weight:600;">OLS Baseline (Unregularized)</div>
            <div style="color:#f9fafb; font-size:1.15rem; font-weight:700; margin-top:4px;">R² = {ols_m['R2']:.4f}</div>
            <div style="color:#9ca3af; font-size:0.8rem; margin-top:2px;">MAE: {ols_m['MAE']:.4f} {unit} | RMSE: {ols_m['RMSE']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        ridge_m = reg_data["ridge"]["metrics"]
        ridge_alpha = reg_data["ridge"]["best_alpha"]
        st.markdown(f"""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#60a5fa; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Ridge (L2 Regularized)</div>
            <div style="color:#60a5fa; font-size:1.15rem; font-weight:700; margin-top:4px;">R² = {ridge_m['R2']:.4f}</div>
            <div style="color:#9ca3af; font-size:0.8rem; margin-top:2px;">Best α = {ridge_alpha:.4f} | MAE: {ridge_m['MAE']:.4f} {unit}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        lasso_m = reg_data["lasso"]["metrics"]
        lasso_alpha = reg_data["lasso"]["best_alpha"]
        st.markdown(f"""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#10b981; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Lasso (L1 Feature Selected)</div>
            <div style="color:#10b981; font-size:1.15rem; font-weight:700; margin-top:4px;">R² = {lasso_m['R2']:.4f}</div>
            <div style="color:#9ca3af; font-size:0.8rem; margin-top:2px;">Best α = {lasso_alpha:.4f} | MAE: {lasso_m['MAE']:.4f} {unit}</div>
        </div>
        """, unsafe_allow_html=True)

    # Actual vs Predicted Plots
    st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#9ca3af; margin-top:16px;'>Actual vs. Predicted Correlation Plots</p>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        fig_ols = _actual_vs_predicted_plot(y_test, reg_data["ols"]["predictions"], "OLS Baseline Fit", unit)
        st.pyplot(fig_ols, use_container_width=True)
        plt.close(fig_ols)
    with p_col2:
        fig_ridge = _actual_vs_predicted_plot(y_test, reg_data["ridge"]["predictions"], f"Ridge (L2, α={ridge_alpha})", unit)
        st.pyplot(fig_ridge, use_container_width=True)
        plt.close(fig_ridge)
    with p_col3:
        fig_lasso = _actual_vs_predicted_plot(y_test, reg_data["lasso"]["predictions"], f"Lasso (L1, α={lasso_alpha})", unit)
        st.pyplot(fig_lasso, use_container_width=True)
        plt.close(fig_lasso)

    # ── Section 2: Lasso (L1) Feature Shrinkage Demonstration ────────────────
    st.markdown("""
    <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:16px; margin-top:20px; margin-bottom:14px;">
        <span style="font-weight:700; color:#f9fafb; font-size:1rem;">Feature Coefficient Shrinkage Analysis (Lasso L1 vs. Ridge L2)</span>
        <p style="margin:4px 0 0 0; color:#9ca3af; font-size:0.85rem;">
            Demonstrates L1 penalization: uninformative or redundant telemetry features are shrunk to <strong>exactly zero</strong>, achieving automated sparse feature selection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    lasso_fi = get_lasso_feature_importance(reg_data["lasso"]["model"], FEATURE_NAMES)
    ridge_fi = get_ridge_feature_importance(reg_data["ridge"]["model"], FEATURE_NAMES)

    shrunk_zero_feats = lasso_fi[abs(lasso_fi["Coefficient"]) < 1e-6]["Feature"].tolist()

    bar_col1, bar_col2 = st.columns(2)
    with bar_col1:
        st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#ef4444;'>Lasso (L1) Coefficients (Red = Shrunk to Exactly 0)</p>", unsafe_allow_html=True)
        fig_l, ax_l = plt.subplots(figsize=(6, 3.2))
        l_colors = ["#ef4444" if abs(c) < 1e-6 else "#10b981" for c in lasso_fi["Coefficient"]]
        ax_l.barh(lasso_fi["Feature"], lasso_fi["Coefficient"], color=l_colors, alpha=0.9)
        ax_l.set_xlabel("Lasso Feature Weight", color="#9ca3af")
        ax_l.tick_params(colors="#9ca3af")
        ax_l.spines["bottom"].set_color("#374151")
        ax_l.spines["left"].set_color("#374151")
        ax_l.spines["top"].set_visible(False)
        ax_l.spines["right"].set_visible(False)
        fig_l.tight_layout()
        st.pyplot(fig_l, use_container_width=True)
        plt.close(fig_l)

        if shrunk_zero_feats:
            st.warning(f"🚫 Features eliminated by Lasso (weight = 0.0000): **{', '.join(shrunk_zero_feats)}**")
        else:
            st.info("All features retained with non-zero weights.")

    with bar_col2:
        st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#3b82f6;'>Ridge (L2) Coefficients (Continuous Smooth Shrinkage)</p>", unsafe_allow_html=True)
        fig_r, ax_r = plt.subplots(figsize=(6, 3.2))
        ax_r.barh(ridge_fi["Feature"], ridge_fi["Coefficient"], color="#3b82f6", alpha=0.9)
        ax_r.set_xlabel("Ridge Feature Weight", color="#9ca3af")
        ax_r.tick_params(colors="#9ca3af")
        ax_r.spines["bottom"].set_color("#374151")
        ax_r.spines["left"].set_color("#374151")
        ax_r.spines["top"].set_visible(False)
        ax_r.spines["right"].set_visible(False)
        fig_r.tight_layout()
        st.pyplot(fig_r, use_container_width=True)
        plt.close(fig_r)

        st.info("💡 Ridge penalizes large weights via Euclidean L2 norm but preserves all features without hard sparsity.")

    # ── Section 3: Interactive Workload Scenario Simulator ───────────────────
    st.markdown("""
    <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:16px; margin-top:24px; margin-bottom:14px;">
        <span style="font-weight:700; color:#f9fafb; font-size:1rem;">Interactive Cost & Latency Forecasting Simulator</span>
        <p style="margin:4px 0 0 0; color:#9ca3af; font-size:0.85rem;">
            Simulate hypothetical prompt workloads and predict real-time GPU compute expenditure and edge turnaround.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sim1, sim2, sim3 = st.columns(3)
    with sim1:
        sim_chars = st.slider("Simulated Prompt Length (Chars)", 20, 2000, 350, key="sim_chars")
        sim_words = max(5, int(sim_chars / 6.0))
        sim_entropy = st.slider("Shannon Entropy", 2.0, 6.0, 4.2, 0.1, key="sim_ent")

    with sim2:
        sim_override = st.slider("Override Density", 0.0, 0.5, 0.02, 0.01, key="sim_over")
        sim_symbol = st.slider("Symbol Ratio", 0.0, 0.4, 0.08, 0.01, key="sim_sym")

    with sim3:
        sim_burst = st.slider("Uppercase Burst", 0.0, 0.6, 0.05, 0.01, key="sim_burst")
        sim_complexity = st.slider("Avg Word Complexity", 3.0, 10.0, 5.2, 0.1, key="sim_comp")

    # Build simulated vector
    sim_vector = np.array([
        sim_override,
        sim_symbol,
        sim_entropy,
        sim_burst,
        min(sim_chars / 1000.0, 10.0),
        sim_complexity,
        0.05,  # imperative score
    ])

    scaler = engines["scaler"]
    sim_sc = scaler.transform(sim_vector.reshape(1, -1))

    pred_lat_ols = max(5.0, float(engines["latency_reg"]["ols"]["model"].predict(sim_sc)[0]))
    pred_lat_ridge = max(5.0, float(engines["latency_reg"]["ridge"]["model"].predict(sim_sc)[0]))
    pred_lat_lasso = max(5.0, float(engines["latency_reg"]["lasso"]["model"].predict(sim_sc)[0]))

    pred_cost_ols = max(0.0001, float(engines["cost_reg"]["ols"]["model"].predict(sim_sc)[0]))
    pred_cost_ridge = max(0.0001, float(engines["cost_reg"]["ridge"]["model"].predict(sim_sc)[0]))
    pred_cost_lasso = max(0.0001, float(engines["cost_reg"]["lasso"]["model"].predict(sim_sc)[0]))

    st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#9ca3af; margin-top:14px;'>Simulated Workload Forecast</p>", unsafe_allow_html=True)
    res1, res2, res3 = st.columns(3)
    with res1:
        st.markdown(f"""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#9ca3af; font-size:0.75rem; text-transform:uppercase; font-weight:600;">OLS Baseline Estimate</div>
            <div style="color:#f9fafb; font-size:1.2rem; font-weight:700; margin-top:4px;">{pred_lat_ols:.1f} ms</div>
            <div style="color:#10b981; font-size:0.9rem; margin-top:2px;">${pred_cost_ols:.5f} / req</div>
        </div>
        """, unsafe_allow_html=True)

    with res2:
        st.markdown(f"""
        <div style="background:#1f2937; border:1px solid #3b82f6; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#60a5fa; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Ridge L2 Production Estimate</div>
            <div style="color:#60a5fa; font-size:1.2rem; font-weight:700; margin-top:4px;">{pred_lat_ridge:.1f} ms</div>
            <div style="color:#10b981; font-size:0.9rem; margin-top:2px;">${pred_cost_ridge:.5f} / req</div>
        </div>
        """, unsafe_allow_html=True)

    with res3:
        st.markdown(f"""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#9ca3af; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Lasso L1 Sparse Estimate</div>
            <div style="color:#f9fafb; font-size:1.2rem; font-weight:700; margin-top:4px;">{pred_lat_lasso:.1f} ms</div>
            <div style="color:#10b981; font-size:0.9rem; margin-top:2px;">${pred_cost_lasso:.5f} / req</div>
        </div>
        """, unsafe_allow_html=True)

    # 100k batch calculation
    cost_per_100k = pred_cost_ridge * 100000.0
    st.info(f"💡 **At Scale Economic Analysis:** 100,000 requests of this payload profile are forecasted to cost **${cost_per_100k:.2f}** in GPU tokens with an aggregate compute duration of **{(pred_lat_ridge * 100000 / 1000 / 60):.1f} compute minutes**.")
