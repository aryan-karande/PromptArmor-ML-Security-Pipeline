# 🛡️ PromptArmor — Enterprise AI Security Gateway & Compute Cost Optimizer

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Enterprise%20UI-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-Neural%20Networks-ff6f00)
![Security](https://img.shields.io/badge/Security-Zero--Trust%20LLM%20Defense-10b981)

**PromptArmor** is a production-grade Enterprise AI Security SaaS platform and threat interception gateway designed to safeguard Large Language Model (LLM) infrastructures against zero-day adversarial prompt injection attacks, jailbreaks, and unauthorized privilege escalation while forecasting execution latency and GPU compute expenditure in real time.

Built for live executive demonstration, PromptArmor operates as an intelligent firewall layer sitting between client applications and foundation model clusters, providing deep neural inspection, deterministic symbolic policy guardrails, and automated workload routing.

---

## 🏛️ Enterprise Architecture & Top-Level Modules

PromptArmor is organized into five top-level operational consoles:

### 1. 🚀 Live Security Sandbox (The Main Showcase)
* **Real-Time Prompt Ingestion Terminal:** Interactive prompt submission console with quick presets for Safe Queries, Complex Technical Code, and Critical Adversarial Jailbreaks.
* **Client Identity & Metadata Context:** Configurable client tiers (Enterprise Platinum, FinTech, HIPAA, Pro Developer), intent categories, authentication contexts, and execution privilege flags.
* **Dynamic Warning Banners:** Live color-coded security enforcement banners (Green for Cleared Traffic, Blue for Specialized Accelerator Routing, Red for Adversarial Jailbreak Blocked).
* **Structural Telemetry Gauges:** Instantaneous extraction of 7 numerical structural security vectors and expected latency/token cost calculations.
* **Cryptographic Audit Receipts:** SHA-256 payload digest, transaction receipt ID, and routing telemetry.

### 2. 🧠 Threat Intelligence Arena
* **Deep Neural Classifier (Keras MLP):** Multi-Layer Perceptron architecture (`Input(7) -> Dense(24, ReLU) -> Dense(12, ReLU) -> Dense(3, Softmax)`) delivering multi-class confidence distributions (Safe vs. Complex Query vs. Malicious Jailbreak) with training convergence curves.
* **Interpretable Decision Tree Policy Engine:** Entropy and Information Gain-based transparent firewall rule paths rendered in clean, human-readable IF-THEN logic with feature importance rankings.
* **Nearest-Neighbor Threat Vector Store:** Euclidean vector indexing across historical adversarial prompt families with cross-validation hyperparameter tuning.
* **Multi-Model Performance Benchmarking:** Comprehensive comparative audit matrix evaluating accuracy, inference latency, and interpretability.

### 3. 🧹 Data Sanitization & Ingestion Engine
* **Production Telemetry Buffer:** Ingestion monitoring and hygiene analysis across 1,500 enterprise prompt records.
* **Missing Value Imputation:** Statistical imputation preserving continuous distributions (Median for latency/cost telemetry, Mode for categorical security metadata).
* **IQR Anomaly Quarantine:** Dynamic Interquartile Range ($Q_1 - k \times \text{IQR}$ to $Q_3 + k \times \text{IQR}$) statistical filtering to isolate and quarantine anomalous telemetry spikes.
* **Interactive Telemetry Distributions:** Distribution histograms and Pearson correlation heatmaps across structural security metrics and computational cost.

### 4. 📋 Policy Guardrails Engine
* **Symbolic Version Space Boundaries:** Tracks the Most Specific boundary ($S$, strictest acceptable whitelist) and Most General boundary ($G$, outer permissive envelope) using Candidate Elimination.
* **Contradictory Policy Injection & Collapse:** Simulates policy contradictions to demonstrate Version Space collapse, triggering automated SecOps escalation alerts.
* **Interactive Compliance Status Monitor:** Validates client requests against active enterprise compliance profiles (Enterprise Platinum, FinTech PCI/GLBA, Healthcare HIPAA, Developer API, Public Sandbox).

### 5. 📊 Compute Cost & Latency Forecasting
* **Regularized Regression Performance:** Comparative evaluation of Ordinary Least Squares (OLS), Ridge ($L_2$), and Lasso ($L_1$) regression models with live $R^2$, MAE, RMSE, and cross-validated $\alpha$ tuning.
* **Lasso $L_1$ Sparsity Shrinkage:** Visual horizontal bar chart proving Lasso's ability to drive uninformative feature coefficients to exactly zero.
* **Workload Simulator:** Interactive slider-driven forecast predicting token compute expenditures and latency for hypothetical prompt workloads at enterprise scale (e.g., 100K batch analysis).

---

## 🛠️ Technology Stack

* **UI & Dashboard Framework:** Streamlit (Custom enterprise dark-mode CSS: `#0b0f19` background, `#1f2937` surfaces, `#3b82f6` blue accents)
* **Core Machine Learning:** Scikit-Learn, NumPy, Pandas
* **Deep Learning Framework:** TensorFlow / Keras (Multi-Layer Perceptron with Categorical Cross-Entropy)
* **Statistical Visualizations:** Matplotlib, Seaborn

---

## 📦 Local Setup & Deployment

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/aryan-karande/PromptArmor-ML-Security-Pipeline.git
   cd PromptArmor-ML-Security-Pipeline
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Pipeline Integrity:**
   ```bash
   python verify_pipeline.py
   ```

4. **Launch PromptArmor Security Gateway:**
   ```bash
   streamlit run app.py
   ```

---

## 🔒 Enterprise Compliance Standards

PromptArmor is designed to adhere to zero-trust architecture principles, providing deterministic policy invariants, cryptographic audit receipts, and automated containment for enterprise LLM deployments.
