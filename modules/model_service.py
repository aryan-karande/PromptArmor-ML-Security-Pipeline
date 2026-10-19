"""
PromptArmor — Central Model & Security Pipeline Service
======================================================
Provides high-performance, cached initialization of enterprise machine learning
classifiers, regularized regression predictors, and real-time security inspection.
"""

import os
import time
import hashlib
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.dataset_generator import generate_dataset
from modules.feature_extraction import extract_features, extract_features_batch, FEATURE_NAMES
from modules.classifiers import (
    build_mlp,
    train_mlp,
    get_mlp_results,
    train_decision_tree,
    get_tree_rules,
    get_tree_results,
    train_knn,
    get_knn_results,
    find_nearest_neighbors,
    DEFAULT_CLASS_NAMES,
)
from modules.regression import (
    train_ols,
    train_ridge,
    train_lasso,
    evaluate_regression,
    get_lasso_feature_importance,
    get_ridge_feature_importance,
    run_regression_pipeline,
)
from modules.concept_learning import (
    generate_policy_examples,
    find_s,
    candidate_elimination,
    format_hypothesis,
    check_version_space_collapse,
)

CLASS_NAMES = ["Safe", "Complex Query", "Malicious Jailbreak"]


@st.cache_data(show_spinner="🔄 Loading Enterprise Telemetry Buffer (1,500 records)...")
def get_dataset(seed: int = 42) -> pd.DataFrame:
    """Generate and cache the 1,500 production prompt telemetry dataset."""
    return generate_dataset(seed=seed)


@st.cache_data(show_spinner="🔬 Extracting 7 Structural Telemetry Vectors...")
def get_extracted_features(prompts_tuple: tuple) -> pd.DataFrame:
    """Extract and cache the 7 numerical structural security metrics."""
    return extract_features_batch(pd.Series(prompts_tuple))


@st.cache_resource(show_spinner="🛡️ Initializing Enterprise Security Pipeline & ML Engines...")
def get_security_engines() -> Dict[str, Any]:
    """
    Train and cache all core machine learning models:
      - Deep Neural Network (MLP)
      - Interpretable Decision Tree Policy Engine
      - Nearest-Neighbor Similarity Matcher (KNN)
      - Regularized Regression Models (OLS, Ridge L2, Lasso L1)
    """
    df = get_dataset()
    df_clean = df.dropna(subset=["prompt_length", "token_estimate", "response_latency_ms", "token_cost"]).copy()
    
    # ── Feature Extraction ───────────────────────────────────────────────────
    prompts_tuple = tuple(df_clean["prompt_text"].tolist())
    features_df = get_extracted_features(prompts_tuple)
    X = features_df.values
    y = df_clean["label"].values
    
    # Train/Test Split with stratification
    indices = np.arange(len(df_clean))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_df = df_clean.iloc[train_idx].reset_index(drop=True)
    test_df = df_clean.iloc[test_idx].reset_index(drop=True)
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # ── 1. Deep Neural Network (MLP) ────────────────────────────────────────
    mlp_model = build_mlp(input_dim=7, num_classes=3, single_layer=False)
    mlp_history = train_mlp(mlp_model, X_train_sc, y_train, X_test_sc, y_test, epochs=20, batch_size=32)
    mlp_results = get_mlp_results(mlp_model, X_test_sc, y_test)
    
    # ── 2. Interpretable Decision Tree ──────────────────────────────────────
    dt_model = train_decision_tree(X_train_sc, y_train, max_depth=3, criterion="entropy")
    dt_results = get_tree_results(dt_model, X_test_sc, y_test)
    dt_rules = get_tree_rules(dt_model, FEATURE_NAMES, CLASS_NAMES)
    
    # ── 3. Nearest-Neighbor Matcher (KNN) ────────────────────────────────────
    knn_model = train_knn(X_train_sc, y_train, k=5)
    knn_results = get_knn_results(knn_model, X_test_sc, y_test)
    
    # ── 4. Compute Cost & Latency Regression ─────────────────────────────────
    y_latency_tr = train_df["response_latency_ms"].values
    y_latency_te = test_df["response_latency_ms"].values
    y_cost_tr = train_df["token_cost"].values
    y_cost_te = test_df["token_cost"].values
    
    latency_reg = run_regression_pipeline(X_train_sc, X_test_sc, y_latency_tr, y_latency_te, FEATURE_NAMES)
    cost_reg = run_regression_pipeline(X_train_sc, X_test_sc, y_cost_tr, y_cost_te, FEATURE_NAMES)
    
    # ── 5. Policy Guardrails Baseline ────────────────────────────────────────
    policy_df_clean = generate_policy_examples(inject_noise=False)
    final_S, final_G, S_hist, G_hist = candidate_elimination(policy_df_clean)
    
    return {
        "df_clean": df_clean,
        "features_df": features_df,
        "train_df": train_df,
        "test_df": test_df,
        "scaler": scaler,
        "X_train_sc": X_train_sc,
        "X_test_sc": X_test_sc,
        "y_train": y_train,
        "y_test": y_test,
        "mlp_model": mlp_model,
        "mlp_history": mlp_history,
        "mlp_results": mlp_results,
        "dt_model": dt_model,
        "dt_results": dt_results,
        "dt_rules": dt_rules,
        "knn_model": knn_model,
        "knn_results": knn_results,
        "latency_reg": latency_reg,
        "cost_reg": cost_reg,
        "final_S": final_S,
        "final_G": final_G,
        "policy_df_clean": policy_df_clean,
    }


def inspect_security_payload(
    prompt_text: str,
    client_tier: str = "Enterprise Platinum",
    intent: str = "Information Retrieval",
    auth_state: str = "Authenticated (MFA/SSO)",
    code_execution: str = "No",
) -> Dict[str, Any]:
    """
    Run an end-to-end security inspection on an inbound prompt payload:
      1. Structural telemetry feature extraction (7 metrics).
      2. Multi-model classification & softmax confidence.
      3. Historical threat vector similarity search.
      4. Latency and compute cost forecasting.
      5. Policy guardrail compliance verification.
    """
    engines = get_security_engines()
    scaler = engines["scaler"]
    
    # 1. Feature extraction
    raw_features = extract_features(prompt_text)
    scaled_features = scaler.transform(raw_features.reshape(1, -1))
    
    # 2. Neural Network Classification
    mlp = engines["mlp_model"]
    probs = mlp.predict(scaled_features, verbose=0)[0]
    predicted_class = int(np.argmax(probs))
    
    # Decision Tree confirmation
    dt = engines["dt_model"]
    dt_pred = int(dt.predict(scaled_features)[0])
    
    # 3. Nearest Threat Vectors via KNN
    knn = engines["knn_model"]
    train_df = engines["train_df"]
    distances, neighbor_indices = knn.kneighbors(scaled_features, n_neighbors=min(5, len(train_df)))
    
    matched_vectors = []
    for dist, idx in zip(distances[0], neighbor_indices[0]):
        row = train_df.iloc[idx]
        matched_vectors.append({
            "Historical Prompt": str(row.get("prompt_text", ""))[:120] + "...",
            "Threat Category": CLASS_NAMES[int(row.get("label", 0))],
            "Client Tier": row.get("user_tier", "Unknown"),
            "Euclidean Distance": float(dist),
            "Similarity Score": float(max(0.0, 1.0 - (dist / 10.0))),
        })
    
    # 4. Latency and Cost Forecast
    ridge_lat_model = engines["latency_reg"]["ridge"]["model"]
    ridge_cost_model = engines["cost_reg"]["ridge"]["model"]
    
    pred_latency_ms = max(5.0, float(ridge_lat_model.predict(scaled_features)[0]))
    pred_token_cost = max(0.0001, float(ridge_cost_model.predict(scaled_features)[0]))
    
    # Word & Token estimation
    word_count = len(prompt_text.split())
    estimated_tokens = max(1, int(len(prompt_text) / 3.8))
    
    # 5. Policy Guardrails check
    final_S = engines["final_S"]
    # Check compliance against S boundary:
    # Attributes: user_tier, intent_type, code_execution, auth_state
    norm_tier = "Enterprise" if "Enterprise" in client_tier else ("Pro" if "Pro" in client_tier else "Free")
    norm_intent = "Instruction" if "Code" in intent or "Directive" in intent else ("Question" if "Info" in intent else "Analysis")
    norm_auth = "Authenticated" if "Authenticated" in auth_state else "Anonymous"
    norm_code = "Yes" if "Yes" in code_execution or "Code Generation" in intent else "No"
    
    compliant = True
    rule_violations = []
    
    # If S has specific constraints, evaluate them
    if final_S is not None and len(final_S) == 4:
        s_tier, s_intent, s_code, s_auth = final_S
        if s_tier != "?" and s_tier != norm_tier:
            rule_violations.append(f"Tier mismatch: policy requires {s_tier}, got {norm_tier}")
        if s_auth != "?" and s_auth != norm_auth:
            rule_violations.append(f"Auth state violation: policy requires {s_auth}, got {norm_auth}")
        if s_code != "?" and s_code == "No" and norm_code == "Yes":
            rule_violations.append("Code execution is strictly prohibited under current active guardrails")
    
    # Disallow jailbreaks regardless of auth
    if predicted_class == 2:
        compliant = False
        rule_violations.append("Adversarial jailbreak signature detected by neural threat filter")
    elif norm_auth == "Anonymous" and norm_code == "Yes":
        compliant = False
        rule_violations.append("Anonymous code execution privilege rejected")
        
    # Transaction Hash
    sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16].upper()
    receipt_id = f"PA-{int(time.time()) % 1000000:06d}-{sha256[:6]}"
    
    return {
        "receipt_id": receipt_id,
        "sha256": sha256,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "predicted_class": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence_probs": {
            "Safe": float(probs[0]),
            "Complex Query": float(probs[1]),
            "Malicious Jailbreak": float(probs[2]),
        },
        "features": dict(zip(FEATURE_NAMES, raw_features)),
        "matched_vectors": matched_vectors,
        "predicted_latency_ms": pred_latency_ms,
        "predicted_token_cost": pred_token_cost,
        "estimated_tokens": estimated_tokens,
        "word_count": word_count,
        "character_count": len(prompt_text),
        "compliant": compliant and (predicted_class != 2),
        "rule_violations": rule_violations,
    }
