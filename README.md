# PromptArmor: AI Prompt Security & Compute Cost Optimizer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-Neural%20Networks-ff6f00)

PromptArmor is an end-to-end machine learning security firewall and compute cost predictor designed to safeguard enterprise LLM applications against zero-day prompt injection attacks (jailbreaks) and unauthorized access, while predicting operational token latency and costs. 

Built as an advanced machine learning demonstration, this project integrates foundational data preprocessing, concept learning, multi-class classification, and regularized regression pipelines into a unified interactive dashboard.

---

## 🚀 Key Features & Pipeline Architecture

The application is structured into four interactive modules representing a comprehensive machine learning workflow:

### 1. Data Preprocessing & Sanitization Pipeline (Lab 1 & Lab 2)
* **Missing Value Imputation:** Automatically handles incomplete prompt logs using median imputation for continuous features and mode imputation for categorical attributes.
* **Categorical Encoding:** Applies One-Hot Encoding with `drop='first'` to prevent the Dummy Variable Trap.
* **Feature Scaling:** Utilizes `StandardScaler` fitted strictly on training data partitions to prevent data leakage.
* **Outlier Filtering:** Incorporates the **Interquartile Range (IQR) method** ($Q_1 - 1.5 \times \text{IQR}$ to $Q_3 + 1.5 \times \text{IQR}$) to isolate and remove extreme length anomalies and noise.

### 2. Concept Learning & Policy Guardrails (Lab 3)
* **Version Space Boundaries:** Implements **Find-S** and **Candidate Elimination** algorithms to track and update the Most Specific ($S$) and Most General ($G$) hypothesis spaces governing valid enterprise query rules.
* **Noise Vulnerability Analysis:** Features an interactive noise injection toggle to demonstrate Version Space collapse under contradictory training instances.

### 3. Multi-Model Threat Classification Arena (Labs 4, 5 & 6)
* **Feature Extraction Engine:** Dynamically extracts 7 structural metrics from any input text (Override Density, Symbol Ratio, Shannon Entropy, Uppercase Burst, Prompt Length, Word Complexity, and Imperative Score).
* **Multi-Layer Perceptron (MLP):** Implements a deep neural network architecture (`Input(shape=(7,)) -> Dense(24, ReLU) -> Dense(12, ReLU) -> Dense(3, Softmax)`) with `EarlyStopping` callbacks to classify inputs into **Safe**, **Complex Query**, or **Malicious Jailbreak**, complete with training/validation loss curves and Multiclass Confusion Matrices.
* **Decision Tree Explainer:** Trains an entropy-based Decision Tree classifier with an adjustable pruning depth (`max_depth=3` vs. unconstrained `19`) to visualize transparent firewall decision thresholds.
* **K-Nearest Neighbors (KNN):** Utilizes Euclidean distance metrics with an interactive $K$-slider ($K=1$ to $30$) and Cross-Validation accuracy tuning to retrieve similar historical attack vectors.

### 4. Regression & Cost Forecaster (Lab 2)
* **Regularized Regression Models:** Implements Multiple Linear Regression (OLS), Ridge (L2), and Lasso (L1) models optimized via `GridSearchCV` ($cv=5$) to predict continuous response latency (ms) and operational cost.
* **Feature Selection:** Demonstrates Lasso regression automatically shrinking uninformative text coefficients to absolute zero.

---

## 🛠️ Tech Stack

* **Core Programming Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning & Preprocessing:** Scikit-Learn, NumPy, Pandas
* **Deep Learning:** TensorFlow / Keras
* **Data Visualization:** Matplotlib, Seaborn

---

## 📦 Installation & Local Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/aryan-karande/PromptArmor-ML-Security-Pipeline.git](https://github.com/aryan-karande/PromptArmor-ML-Security-Pipeline.git)
   cd PromptArmor-ML-Security-Pipeline
