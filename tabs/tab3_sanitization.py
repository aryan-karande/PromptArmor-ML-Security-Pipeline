"""
Tab 3 — Data Sanitization & Ingestion Engine
============================================
Telemetry ingestion audit, missing-value imputation (median/mode),
IQR-driven statistical anomaly scrubbing, and correlation telemetry distribution.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules.model_service import get_dataset
from modules.preprocessing import (
    impute_missing_values,
    encode_categoricals,
    filter_outliers,
    detect_outliers_iqr,
    get_correlation_matrix,
)

plt.style.use("dark_background")


def render():
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="margin: 0; color: #f9fafb; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em;">
            🧹 Data Sanitization & Ingestion Engine
        </h2>
        <p style="margin: 4px 0 0 0; color: #9ca3af; font-size: 0.95rem;">
            Production Telemetry Ingestion • Missing Value Scrubbing • Statistical Anomaly Quarantine
        </p>
    </div>
    """, unsafe_allow_html=True)

    df_raw = get_dataset()

    # ── Executive KPI Row ────────────────────────────────────────────────────
    total_records = len(df_raw)
    raw_missing = int(df_raw.isnull().sum().sum())
    total_cells = df_raw.shape[0] * df_raw.shape[1]
    hygiene_score = (1.0 - (raw_missing / total_cells)) * 100.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#9ca3af; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Ingested Logs</div>
            <div style="color:#f9fafb; font-size:1.25rem; font-weight:700; margin-top:4px;">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#9ca3af; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Raw Incomplete Fields</div>
            <div style="color:#ef4444; font-size:1.25rem; font-weight:700; margin-top:4px;">{raw_missing:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#9ca3af; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Data Hygiene Score</div>
            <div style="color:#10b981; font-size:1.25rem; font-weight:700; margin-top:4px;">{hygiene_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:14px; text-align:center;">
            <div style="color:#9ca3af; font-size:0.75rem; text-transform:uppercase; font-weight:600;">Telemetry Schema</div>
            <div style="color:#60a5fa; font-size:1.25rem; font-weight:700; margin-top:4px;">{df_raw.shape[1]} Attributes</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

    # ── Section 1: Raw Ingestion Buffer Explorer ─────────────────────────────
    with st.expander("📋 Raw Telemetry Buffer (1,500 Ingested Enterprise Records)", expanded=False):
        st.dataframe(df_raw.head(25), use_container_width=True, height=280)

    # ── Section 2: Missing Value Imputation Engine ────────────────────────────
    st.markdown("""
    <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:16px; margin-top:14px; margin-bottom:14px;">
        <span style="font-weight:700; color:#f9fafb; font-size:1rem;">1. Enterprise Missing Value Imputation Engine</span>
        <p style="margin:4px 0 0 0; color:#9ca3af; font-size:0.85rem;">
            Preserves continuous distributions via <strong>Median Imputation</strong> for latency/cost telemetry, and <strong>Mode Imputation</strong> for categorical security metadata.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df_imputed, impute_stats = impute_missing_values(df_raw.copy())

    imp_col1, imp_col2 = st.columns([1, 1])

    with imp_col1:
        st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#9ca3af;'>Raw Missing Values Per Telemetry Metric</p>", unsafe_allow_html=True)
        missing_counts = df_raw.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        if len(missing_counts) > 0:
            fig_m, ax_m = plt.subplots(figsize=(5.5, 2.8))
            ax_m.barh(missing_counts.index, missing_counts.values, color="#ef4444", alpha=0.85)
            ax_m.set_xlabel("Missing Values Detected", color="#9ca3af")
            ax_m.tick_params(colors="#9ca3af")
            ax_m.spines["bottom"].set_color("#374151")
            ax_m.spines["left"].set_color("#374151")
            ax_m.spines["top"].set_visible(False)
            ax_m.spines["right"].set_visible(False)
            fig_m.tight_layout()
            st.pyplot(fig_m, use_container_width=True)
            plt.close(fig_m)

    with imp_col2:
        st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#9ca3af;'>Imputation Resolution Audit Log</p>", unsafe_allow_html=True)
        if impute_stats:
            imp_summary = []
            for col, details in impute_stats.items():
                imp_summary.append({
                    "Telemetry Attribute": col,
                    "Values Scrubbed": details["count"],
                    "Imputation Policy": details["strategy"].upper(),
                    "Computed Value": f"{details['fill_value']:.4f}" if isinstance(details["fill_value"], float) else str(details["fill_value"])
                })
            st.dataframe(pd.DataFrame(imp_summary), use_container_width=True, hide_index=True)
        st.success("✅ Telemetry buffer 100% sanitized: all missing values imputed.")

    # ── Section 3: IQR Outlier Detection & Anomaly Scrubbing ──────────────────
    st.markdown("""
    <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:16px; margin-top:20px; margin-bottom:14px;">
        <span style="font-weight:700; color:#f9fafb; font-size:1rem;">2. Statistical Anomaly & Outlier Isolation (IQR Fences)</span>
        <p style="margin:4px 0 0 0; color:#9ca3af; font-size:0.85rem;">
            Calculates Interquartile Range (IQR = Q3 - Q1) and establishes dynamic fences [Q1 - k×IQR, Q3 + k×IQR] to quarantine extreme payload telemetry anomalies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    numerical_cols = ["prompt_length", "token_estimate", "response_latency_ms", "token_cost"]
    numerical_cols = [c for c in numerical_cols if c in df_imputed.columns]

    slider_col1, slider_col2 = st.columns([1, 2])
    with slider_col1:
        iqr_k = st.slider("IQR Fence Multiplier (k)", 1.0, 3.0, 1.5, 0.1, key="sanit_iqr_slider")
    with slider_col2:
        st.markdown(f"<p style='color:#9ca3af; font-size:0.85rem; margin-top:12px;'>Currently evaluating with <strong>k = {iqr_k:.1f}</strong> (Standard Enterprise Isolation: 1.5 for moderate anomalies, 3.0 for extreme outliers).</p>", unsafe_allow_html=True)

    df_filtered, outlier_stats = filter_outliers(df_imputed.copy(), numerical_cols, multiplier=iqr_k)

    # Boxplot Before vs After
    box_b, box_a = st.columns(2)
    with box_b:
        st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#ef4444;'>Before Anomaly Scrubbing (Raw Distribution)</p>", unsafe_allow_html=True)
        fig_b, axes_b = plt.subplots(1, len(numerical_cols), figsize=(12, 3.2))
        for i, col_name in enumerate(numerical_cols):
            axes_b[i].boxplot(df_imputed[col_name].dropna(), vert=True, patch_artist=True,
                              boxprops=dict(facecolor="#ef4444", alpha=0.7, color="#ef4444"),
                              medianprops=dict(color="#ffffff"))
            axes_b[i].set_title(col_name, fontsize=8, color="#9ca3af")
            axes_b[i].tick_params(colors="#9ca3af", labelsize=7)
        fig_b.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    with box_a:
        st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#10b981;'>After IQR Anomaly Quarantine (Sanitized)</p>", unsafe_allow_html=True)
        fig_a, axes_a = plt.subplots(1, len(numerical_cols), figsize=(12, 3.2))
        for i, col_name in enumerate(numerical_cols):
            axes_a[i].boxplot(df_filtered[col_name].dropna(), vert=True, patch_artist=True,
                              boxprops=dict(facecolor="#10b981", alpha=0.7, color="#10b981"),
                              medianprops=dict(color="#ffffff"))
            axes_a[i].set_title(col_name, fontsize=8, color="#9ca3af")
            axes_a[i].tick_params(colors="#9ca3af", labelsize=7)
        fig_a.tight_layout()
        st.pyplot(fig_a, use_container_width=True)
        plt.close(fig_a)

    # Outlier Breakdown Table
    st.markdown("<p style='font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; color:#9ca3af; margin-top:12px;'>IQR Fence & Quarantine Audit</p>", unsafe_allow_html=True)
    out_table = []
    for col, stats in outlier_stats.items():
        out_table.append({
            "Telemetry Feature": col,
            "Q1 (25th %)": f"{stats['Q1']:.2f}",
            "Q3 (75th %)": f"{stats['Q3']:.2f}",
            "IQR": f"{stats['IQR']:.2f}",
            "Lower Fence": f"{stats['lower_bound']:.2f}",
            "Upper Fence": f"{stats['upper_bound']:.2f}",
            "Anomalies Quarantined": stats["num_outliers"],
            "Quarantine %": f"{stats['outlier_pct']:.1f}%",
        })
    st.dataframe(pd.DataFrame(out_table), use_container_width=True, hide_index=True)

    total_purged = len(df_imputed) - len(df_filtered)
    st.info(f"🛡️ Isolation Engine quarantined **{total_purged:,}** anomalous payload records ({(total_purged / len(df_imputed) * 100):.1f}% of total). Retained **{len(df_filtered):,}** clean production records.")

    # ── Section 4: Interactive Telemetry Distributions & Pearson Correlation ─
    st.markdown("""
    <div style="background:#1f2937; border:1px solid #374151; border-radius:10px; padding:16px; margin-top:20px; margin-bottom:14px;">
        <span style="font-weight:700; color:#f9fafb; font-size:1rem;">3. Telemetry Correlation & Signal Relationships</span>
        <p style="margin:4px 0 0 0; color:#9ca3af; font-size:0.85rem;">
            Evaluates Pearson correlation coefficients across prompt structural dimensions, response latency, and compute cost.
        </p>
    </div>
    """, unsafe_allow_html=True)

    corr_matrix = get_correlation_matrix(df_filtered, numerical_cols)
    fig_corr, ax_corr = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax_corr,
        linewidths=1,
        linecolor="#1f2937",
        cbar=True,
    )
    ax_corr.set_title("Pearson Correlation Matrix (Sanitized Telemetry)", color="#f9fafb", fontsize=10, fontweight="bold")
    ax_corr.tick_params(colors="#9ca3af", labelsize=8)
    fig_corr.tight_layout()
    st.pyplot(fig_corr, use_container_width=True)
    plt.close(fig_corr)
