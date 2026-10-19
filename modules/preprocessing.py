"""
PromptArmor — Preprocessing & Outlier Detection Module
======================================================
Production-grade data preprocessing, sanitization, and outlier detection
pipeline for the PromptArmor security framework.

Implements:
  Data Ingestion & Preprocessing:
    1. impute_missing_values: Median imputation for continuous metrics and
       mode imputation for categorical attributes.
    2. encode_categoricals: One-Hot Encoding with drop='first' to prevent the
       Dummy Variable Trap, supporting stateful fit/transform.
    3. scale_features: StandardScaler fitted strictly on the training partition
       to prevent data leakage, transforming train and test sets.
    4. run_full_preprocessing: End-to-end orchestration pipeline
       (impute -> encode -> split -> scale).

  Outlier Detection & Correlation Analysis:
    5. detect_outliers_iqr: Interquartile Range (IQR) thresholding to compute
       bounds, outlier masks, counts, and percentages.
    6. filter_outliers: Multi-column IQR filtering to isolate and purge extreme
       anomalies and noisy samples.
    7. get_correlation_matrix: Pearson correlation computation for numerical
       prompt security and cost features.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Module Logger Configuration ──────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ── Default Column Constants ──────────────────────────────────────────────────
DEFAULT_NUMERICAL_COLS: List[str] = [
    "prompt_length",
    "token_estimate",
    "response_latency_ms",
    "token_cost",
]

DEFAULT_CATEGORICAL_COLS: List[str] = [
    "user_tier",
    "intent_type",
    "code_execution",
    "auth_state",
]

DEFAULT_IQR_MULTIPLIER: float = 1.5

# ── Module-Level Encoder Cache ────────────────────────────────────────────────
_STORED_ENCODER: Optional[OneHotEncoder] = None
_STORED_CATEGORICAL_COLS: Optional[List[str]] = None
_STORED_ENCODING_MAP: Optional[Dict[str, List[str]]] = None
_STORED_FEATURE_NAMES: Optional[List[str]] = None


# ── Metadata Helper Class ────────────────────────────────────────────────────
class ImputationStat(int):
    """
    Numeric integer count of imputed values that also supports dictionary-style
    metadata access (strategy and fill_value) for Streamlit UI display.

    Inherits from `int` so expressions like `impute_stats['col'] > 0` and
    `impute_stats['col'] == 5` work identically to an integer, while
    `v['count']`, `v['strategy']`, and `v['fill_value']` satisfy dashboard views.
    """

    count: int
    strategy: str
    fill_value: Any

    def __new__(
        cls,
        count: int,
        strategy: str = "median",
        fill_value: Any = 0.0,
    ) -> "ImputationStat":
        instance = super().__new__(cls, count)
        instance.count = int(count)
        instance.strategy = str(strategy)
        instance.fill_value = fill_value
        return instance

    def __getitem__(self, key: str) -> Any:
        if key == "count":
            return self.count
        elif key == "strategy":
            return self.strategy
        elif key == "fill_value":
            return self.fill_value
        raise KeyError(f"Key '{key}' not found in ImputationStat metadata.")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> List[str]:
        return ["count", "strategy", "fill_value"]

    def values(self) -> List[Any]:
        return [self.count, self.strategy, self.fill_value]

    def items(self) -> List[Tuple[str, Any]]:
        return [
            ("count", self.count),
            ("strategy", self.strategy),
            ("fill_value", self.fill_value),
        ]

    def __repr__(self) -> str:
        return f"{self.count}"


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def _create_one_hot_encoder() -> OneHotEncoder:
    """
    Instantiate a scikit-learn OneHotEncoder with drop='first' compatible with
    both older and modern scikit-learn versions.
    """
    configs = [
        {"drop": "first", "sparse_output": False, "handle_unknown": "ignore"},
        {"drop": "first", "sparse_output": False},
        {"drop": "first", "sparse": False, "handle_unknown": "ignore"},
        {"drop": "first", "sparse": False},
    ]
    for kwargs in configs:
        try:
            return OneHotEncoder(**kwargs)
        except (TypeError, ValueError):
            continue
    return OneHotEncoder(drop="first")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Missing Value Imputation
# ─────────────────────────────────────────────────────────────────────────────
def impute_missing_values(
    df: pd.DataFrame,
    numerical_cols: Optional[Sequence[str]] = None,
    categorical_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, ImputationStat]]:
    """
    Impute missing values in the dataset using statistical strategies:
      - Continuous numerical features imputed via median.
      - Discrete categorical features imputed via mode.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw prompt security and compute records.
    numerical_cols : Optional[Sequence[str]], default=None
        Specific numerical columns to impute. If None, targets columns matching
        DEFAULT_NUMERICAL_COLS plus any other numerical columns in df.
    categorical_cols : Optional[Sequence[str]], default=None
        Specific categorical columns to impute. If None, targets columns matching
        DEFAULT_CATEGORICAL_COLS plus any object/category columns in df.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, ImputationStat]]
        A 2-tuple containing:
          - Imputed copy of the original DataFrame.
          - Imputation summary dict mapping column names to `ImputationStat` instances
            (providing integer count, 'strategy', and 'fill_value').

    Raises
    ------
    ValueError
        If df is None or not a pd.DataFrame.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a valid pandas DataFrame.")

    df_imputed = df.copy()
    imputation_stats: Dict[str, ImputationStat] = {}

    # Determine candidate numerical columns
    if numerical_cols is not None:
        target_num_cols = [c for c in numerical_cols if c in df_imputed.columns]
    else:
        preset_num = [c for c in DEFAULT_NUMERICAL_COLS if c in df_imputed.columns]
        other_num = [
            c for c in df_imputed.select_dtypes(include=[np.number]).columns
            if c not in preset_num
        ]
        target_num_cols = preset_num + other_num

    # Determine candidate categorical columns
    if categorical_cols is not None:
        target_cat_cols = [c for c in categorical_cols if c in df_imputed.columns]
    else:
        preset_cat = [c for c in DEFAULT_CATEGORICAL_COLS if c in df_imputed.columns]
        other_cat = [
            c for c in df_imputed.select_dtypes(include=["object", "category", "bool"]).columns
            if c not in preset_cat and c not in target_num_cols and c not in ["prompt_text", "prompt"]
        ]
        target_cat_cols = preset_cat + other_cat

    # 1. Impute Numerical Features with Median
    for col in target_num_cols:
        num_missing = int(df_imputed[col].isna().sum())
        if num_missing > 0:
            median_val = df_imputed[col].median()
            fill_val = 0.0 if pd.isna(median_val) else float(median_val)
            df_imputed[col] = df_imputed[col].fillna(fill_val)
            imputation_stats[col] = ImputationStat(
                count=num_missing,
                strategy="median",
                fill_value=round(fill_val, 4),
            )
            logger.info("Imputed %d missing values in '%s' using median: %.4f", num_missing, col, fill_val)

    # 2. Impute Categorical Features with Mode
    for col in target_cat_cols:
        num_missing = int(df_imputed[col].isna().sum())
        if num_missing > 0:
            modes = df_imputed[col].mode(dropna=True)
            mode_val = modes.iloc[0] if len(modes) > 0 else "Unknown"
            df_imputed[col] = df_imputed[col].fillna(mode_val)
            imputation_stats[col] = ImputationStat(
                count=num_missing,
                strategy="mode",
                fill_value=mode_val,
            )
            logger.info("Imputed %d missing values in '%s' using mode: %s", num_missing, col, mode_val)

    # 3. Fallback for any remaining unhandled columns with missing values
    remaining_missing = df_imputed.columns[df_imputed.isna().any()].tolist()
    for col in remaining_missing:
        num_missing = int(df_imputed[col].isna().sum())
        if pd.api.types.is_numeric_dtype(df_imputed[col]):
            med = df_imputed[col].median()
            fill_val = 0.0 if pd.isna(med) else float(med)
            df_imputed[col] = df_imputed[col].fillna(fill_val)
            imputation_stats[col] = ImputationStat(
                count=num_missing,
                strategy="median",
                fill_value=round(fill_val, 4),
            )
        else:
            modes = df_imputed[col].mode(dropna=True)
            mode_val = modes.iloc[0] if len(modes) > 0 else "Unknown"
            df_imputed[col] = df_imputed[col].fillna(mode_val)
            imputation_stats[col] = ImputationStat(
                count=num_missing,
                strategy="mode",
                fill_value=mode_val,
            )

    return df_imputed, imputation_stats


# ─────────────────────────────────────────────────────────────────────────────
# 2. Categorical Encoding (One-Hot Encoding with drop='first')
# ─────────────────────────────────────────────────────────────────────────────
def encode_categoricals(
    df: pd.DataFrame,
    fit: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    categorical_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Encode categorical features using One-Hot Encoding with drop='first' to avoid
    the Dummy Variable Trap (multicollinearity).

    Supports stateful persistence: fits and stores the encoder when fit=True,
    and applies the stored encoder when fit=False for test partitions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical attributes.
    fit : bool, default=True
        If True, fits a new OneHotEncoder and stores it in module cache.
        If False, uses the previously fitted/stored encoder to transform df.
    encoder : Optional[OneHotEncoder], default=None
        Explicit OneHotEncoder instance to use. If None and fit=False, retrieves
        the globally cached encoder.
    categorical_cols : Optional[Sequence[str]], default=None
        Explicit list of categorical columns to encode. If None, defaults to
        DEFAULT_CATEGORICAL_COLS present in df.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, List[str]]]
        A 2-tuple containing:
          - Encoded DataFrame with original categorical columns replaced by binary
            dummy indicators, preserving other columns and index.
          - Encoding map dictionary mapping each original categorical feature to
            its newly generated dummy column names.

    Raises
    ------
    ValueError
        If fit=False but no encoder has been fitted or provided, or if required
        categorical columns are absent from df.
    """
    global _STORED_ENCODER, _STORED_CATEGORICAL_COLS, _STORED_ENCODING_MAP, _STORED_FEATURE_NAMES

    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a valid pandas DataFrame.")

    df_result = df.copy()

    # Identify categorical columns to encode
    if fit:
        if categorical_cols is not None:
            cols_to_encode = [c for c in categorical_cols if c in df_result.columns]
        else:
            cols_to_encode = [c for c in DEFAULT_CATEGORICAL_COLS if c in df_result.columns]
            if not cols_to_encode:
                # Fallback to any string/object columns (excluding text/target)
                cols_to_encode = [
                    c for c in df_result.select_dtypes(include=["object", "category"]).columns
                    if c not in ["prompt_text", "prompt", "label", "threat_level", "target"]
                ]

        if not cols_to_encode:
            logger.warning("No categorical columns identified for encoding in DataFrame.")
            return df_result, {}

        # Instantiate and fit encoder
        enc = encoder or _create_one_hot_encoder()
        cat_data = df_result[cols_to_encode].astype(str)
        enc.fit(cat_data)

        # Retrieve output feature names
        try:
            feat_names = enc.get_feature_names_out(cols_to_encode).tolist()
        except AttributeError:
            feat_names = enc.get_feature_names(cols_to_encode).tolist()

        # Build mapping from original column to new dummy column names
        encoding_map: Dict[str, List[str]] = {}
        for col in cols_to_encode:
            prefix = f"{col}_"
            encoding_map[col] = [fn for fn in feat_names if fn.startswith(prefix)]

        # Store state in module cache
        _STORED_ENCODER = enc
        _STORED_CATEGORICAL_COLS = list(cols_to_encode)
        _STORED_ENCODING_MAP = encoding_map
        _STORED_FEATURE_NAMES = feat_names

        # Transform data
        encoded_arr = enc.transform(cat_data)
        if hasattr(encoded_arr, "toarray"):
            encoded_arr = encoded_arr.toarray()

    else:
        # Transform-only mode using existing encoder
        enc = encoder or _STORED_ENCODER
        if enc is None:
            raise ValueError(
                "No fitted encoder found. Call encode_categoricals with fit=True first "
                "or pass a fitted OneHotEncoder via the 'encoder' parameter."
            )

        cols_to_encode = list(categorical_cols) if categorical_cols is not None else _STORED_CATEGORICAL_COLS
        if not cols_to_encode:
            raise ValueError("Categorical columns to encode could not be resolved for transform.")

        missing_cols = [c for c in cols_to_encode if c not in df_result.columns]
        if missing_cols:
            raise ValueError(f"Missing required categorical columns in test DataFrame: {missing_cols}")

        cat_data = df_result[cols_to_encode].astype(str)
        encoded_arr = enc.transform(cat_data)
        if hasattr(encoded_arr, "toarray"):
            encoded_arr = encoded_arr.toarray()

        feat_names = _STORED_FEATURE_NAMES
        if feat_names is None:
            try:
                feat_names = enc.get_feature_names_out(cols_to_encode).tolist()
            except AttributeError:
                feat_names = enc.get_feature_names(cols_to_encode).tolist()

        encoding_map = _STORED_ENCODING_MAP or {}
        if not encoding_map:
            for col in cols_to_encode:
                prefix = f"{col}_"
                encoding_map[col] = [fn for fn in feat_names if fn.startswith(prefix)]

    # Assemble output DataFrame preserving other columns and original index
    encoded_df = pd.DataFrame(
        encoded_arr,
        columns=feat_names,
        index=df_result.index,
        dtype=np.float64,
    )

    remaining_df = df_result.drop(columns=cols_to_encode)
    final_df = pd.concat([remaining_df, encoded_df], axis=1)

    logger.info(
        "Encoded %d categorical columns into %d one-hot features (drop='first').",
        len(cols_to_encode),
        len(feat_names),
    )
    return final_df, encoding_map


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature Scaling (StandardScaler fitted strictly on X_train)
# ─────────────────────────────────────────────────────────────────────────────
def scale_features(
    X_train: Union[np.ndarray, pd.DataFrame, Sequence],
    X_test: Union[np.ndarray, pd.DataFrame, Sequence],
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize numerical features by removing the mean and scaling to unit variance.

    The StandardScaler is fitted exclusively on X_train to prevent data leakage,
    and then used to transform both X_train and X_test partitions.

    Parameters
    ----------
    X_train : array-like or DataFrame of shape (n_train_samples, n_features)
        Training partition feature matrix.
    X_test : array-like or DataFrame of shape (n_test_samples, n_features)
        Testing partition feature matrix.
    feature_cols : Optional[Sequence[str]], default=None
        Subset of column names to select and scale if X_train/X_test are DataFrames.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, StandardScaler]
        A 3-tuple containing:
          - Scaled X_train numpy array (float64).
          - Scaled X_test numpy array (float64).
          - Fitted StandardScaler instance.

    Raises
    ------
    ValueError
        If inputs are empty, None, contain NaNs, or have mismatched feature counts.
    """
    if X_train is None or X_test is None:
        raise ValueError("X_train and X_test feature matrices cannot be None.")

    # Extract columns if DataFrames and feature_cols are provided
    if isinstance(X_train, pd.DataFrame):
        if feature_cols is not None:
            selected = [c for c in feature_cols if c in X_train.columns]
            X_tr = X_train[selected].to_numpy(dtype=np.float64)
        else:
            X_tr = X_train.to_numpy(dtype=np.float64)
    else:
        X_tr = np.asarray(X_train, dtype=np.float64)

    if isinstance(X_test, pd.DataFrame):
        if feature_cols is not None:
            selected = [c for c in feature_cols if c in X_test.columns]
            X_te = X_test[selected].to_numpy(dtype=np.float64)
        else:
            X_te = X_test.to_numpy(dtype=np.float64)
    else:
        X_te = np.asarray(X_test, dtype=np.float64)

    # Ensure 2D shape
    if X_tr.ndim == 1:
        X_tr = X_tr.reshape(-1, 1)
    if X_te.ndim == 1:
        X_te = X_te.reshape(-1, 1)

    if X_tr.shape[0] == 0 or X_te.shape[0] == 0:
        raise ValueError("X_train and X_test must contain at least one sample.")

    if X_tr.shape[1] != X_te.shape[1]:
        raise ValueError(
            f"Feature count mismatch: X_train has {X_tr.shape[1]} features, "
            f"while X_test has {X_te.shape[1]} features."
        )

    if np.isnan(X_tr).any():
        raise ValueError("X_train contains NaN values. Please impute missing values before scaling.")

    if np.isnan(X_te).any():
        raise ValueError("X_test contains NaN values. Please impute missing values before scaling.")

    # Fit StandardScaler strictly on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_tr)
    X_test_scaled = scaler.transform(X_te)

    logger.info(
        "Fitted StandardScaler on %d training samples across %d features.",
        X_tr.shape[0],
        X_tr.shape[1],
    )
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-End Preprocessing Pipeline Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run_full_preprocessing(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: Optional[str] = None,
    feature_cols: Optional[Sequence[str]] = None,
    stratify: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate the complete data preprocessing pipeline:
      1. Impute missing values (median for numerical, mode for categorical).
      2. Extract and protect target label from feature encoding.
      3. One-hot encode categorical features with drop='first'.
      4. Split into train and test sets (with optional stratification).
      5. Scale numerical features with StandardScaler fitted exclusively on train.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset containing prompt attributes, security metrics, and targets.
    test_size : float, default=0.2
        Fraction of the dataset to allocate to the test partition.
    random_state : int, default=42
        Seed for reproducible train/test splitting.
    target_col : Optional[str], default=None
        Target column name. If None, automatically searches candidates:
        ['label', 'threat_level', 'target', 'is_jailbreak', 'attack_type', 'class', 'y'].
    feature_cols : Optional[Sequence[str]], default=None
        Explicit list of feature columns to retain for modeling. If None, all
        numeric columns resulting after one-hot encoding are used.
    stratify : bool, default=True
        Whether to perform stratified sampling on discrete target classes.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing pipeline outputs:
          - 'X_train': np.ndarray (scaled training features)
          - 'X_test': np.ndarray (scaled test features)
          - 'y_train': np.ndarray (training targets)
          - 'y_test': np.ndarray (test targets)
          - 'scaler': fitted StandardScaler instance
          - 'imputation_stats': dict of imputation statistics per column
          - 'encoding_map': dict mapping original categoricals to one-hot columns
          - 'feature_names': list of feature column names (str)

    Raises
    ------
    ValueError
        If df is empty, None, or target cannot be resolved.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    logger.info("Initiating PromptArmor full preprocessing pipeline (N=%d rows)...", len(df))

    # Step 1: Impute missing values
    df_imputed, imputation_stats = impute_missing_values(df)

    # Step 2: Separate Target Column
    resolved_target = target_col
    if resolved_target is None:
        target_candidates = [
            "label",
            "threat_level",
            "target",
            "is_jailbreak",
            "attack_type",
            "class",
            "y",
            "response_latency_ms",
            "token_cost",
        ]
        for candidate in target_candidates:
            if candidate in df_imputed.columns:
                resolved_target = candidate
                break

    if resolved_target is not None and resolved_target in df_imputed.columns:
        y_raw = df_imputed[resolved_target].copy()
        df_features = df_imputed.drop(columns=[resolved_target]).copy()
        logger.info("Identified target column: '%s'", resolved_target)
    else:
        y_raw = None
        df_features = df_imputed.copy()
        logger.info("No target column identified; pipeline will execute in feature-only mode.")

    # Step 3: Categorical Encoding (fit=True)
    df_encoded, encoding_map = encode_categoricals(df_features, fit=True)

    # Step 4: Resolve Modeling Features (numeric only)
    if feature_cols is not None:
        selected_features = [c for c in feature_cols if c in df_encoded.columns]
    else:
        selected_features = [
            c for c in df_encoded.columns
            if pd.api.types.is_numeric_dtype(df_encoded[c])
        ]

    if not selected_features:
        raise ValueError("No numerical features available for modeling after preprocessing.")

    X_df = df_encoded[selected_features].copy()
    feature_names = list(selected_features)

    # Step 5: Train / Test Split
    if y_raw is not None:
        stratify_target = None
        if stratify and not pd.api.types.is_float_dtype(y_raw):
            try:
                class_counts = y_raw.value_counts()
                if (class_counts >= 2).all() and len(class_counts) > 1:
                    stratify_target = y_raw
            except Exception:
                stratify_target = None

        X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
            X_df,
            y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_target,
        )
        y_train_arr = y_train_series.to_numpy()
        y_test_arr = y_test_series.to_numpy()
    else:
        X_train_df, X_test_df = train_test_split(
            X_df,
            test_size=test_size,
            random_state=random_state,
        )
        y_train_arr = None
        y_test_arr = None

    # Step 6: Feature Scaling (fit on train only)
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_df,
        X_test_df,
        feature_cols=feature_names,
    )

    logger.info(
        "Full preprocessing complete. Train shape: %s, Test shape: %s",
        X_train_scaled.shape,
        X_test_scaled.shape,
    )

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train_arr,
        "y_test": y_test_arr,
        "scaler": scaler,
        "imputation_stats": imputation_stats,
        "encoding_map": encoding_map,
        "feature_names": feature_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. IQR Outlier Detection
# ─────────────────────────────────────────────────────────────────────────────
def detect_outliers_iqr(
    series: Union[pd.Series, Sequence[float], np.ndarray],
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> Dict[str, Any]:
    """
    Calculate Interquartile Range (IQR) outlier boundaries and boolean detection
    mask for a numerical series.

    Formula:
      - Q1 = 25th percentile
      - Q3 = 75th percentile
      - IQR = Q3 - Q1
      - lower_bound = Q1 - multiplier * IQR
      - upper_bound = Q3 + multiplier * IQR
      - Outlier if: value < lower_bound OR value > upper_bound

    Parameters
    ----------
    series : pd.Series or array-like
        Numerical series to analyze.
    multiplier : float, default=1.5
        IQR fence multiplier (standard Tukey's fence is 1.5; extreme fence is 3.0).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing outlier diagnostics:
          - 'Q1': float (25th percentile)
          - 'Q3': float (75th percentile)
          - 'IQR': float (interquartile range)
          - 'lower_bound': float (lower outlier threshold)
          - 'upper_bound': float (upper outlier threshold)
          - 'outlier_mask': pd.Series of booleans (True for outliers)
          - 'num_outliers': int (total count of identified outliers)
          - 'outlier_pct': float (percentage of valid observations that are outliers)

    Raises
    ------
    ValueError
        If series is None or multiplier is negative.
    """
    if series is None:
        raise ValueError("Input series cannot be None.")

    if multiplier < 0:
        raise ValueError(f"IQR multiplier must be non-negative. Got {multiplier}.")

    s = pd.Series(series) if not isinstance(series, pd.Series) else series.copy()

    # Handle empty or all-NaN series
    valid_data = s.dropna()
    total_valid = len(valid_data)

    if total_valid == 0:
        empty_mask = pd.Series(False, index=s.index)
        return {
            "Q1": 0.0,
            "Q3": 0.0,
            "IQR": 0.0,
            "lower_bound": 0.0,
            "upper_bound": 0.0,
            "outlier_mask": empty_mask,
            "num_outliers": 0,
            "outlier_pct": 0.0,
        }

    # Calculate quartiles and bounds
    q1 = float(valid_data.quantile(0.25))
    q3 = float(valid_data.quantile(0.75))
    iqr = float(q3 - q1)

    lower_bound = float(q1 - multiplier * iqr)
    upper_bound = float(q3 + multiplier * iqr)

    # Boolean mask: True where value falls outside the IQR fence
    outlier_mask = (s < lower_bound) | (s > upper_bound)
    num_outliers = int(outlier_mask.sum())
    outlier_pct = float((num_outliers / total_valid * 100.0) if total_valid > 0 else 0.0)

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_mask": outlier_mask,
        "num_outliers": num_outliers,
        "outlier_pct": outlier_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Multi-Column Outlier Filtering
# ─────────────────────────────────────────────────────────────────────────────
def filter_outliers(
    df: pd.DataFrame,
    columns: Sequence[str],
    multiplier: float = DEFAULT_IQR_MULTIPLIER,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Apply Interquartile Range (IQR) outlier filtering across specified numerical columns.
    Purges any row that constitutes an outlier in at least one of the analyzed features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing numerical records.
    columns : Sequence[str]
        List of numerical column names on which to evaluate IQR boundaries.
    multiplier : float, default=1.5
        IQR fence multiplier (1.5 for moderate outliers, 3.0 for extreme).

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]
        A 2-tuple containing:
          - Filtered DataFrame with outlier rows removed.
          - Dictionary mapping each analyzed column name to its detailed IQR stats
            dictionary as returned by `detect_outliers_iqr`.

    Raises
    ------
    ValueError
        If df is None or not a pd.DataFrame.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a valid pandas DataFrame.")

    target_columns = [c for c in columns if c in df.columns]
    if not target_columns:
        logger.warning("None of the specified columns exist in DataFrame for outlier filtering.")
        return df.copy(), {}

    combined_outlier_mask = pd.Series(False, index=df.index)
    outlier_stats: Dict[str, Dict[str, Any]] = {}

    for col in target_columns:
        col_stats = detect_outliers_iqr(df[col], multiplier=multiplier)
        outlier_stats[col] = col_stats
        combined_outlier_mask = combined_outlier_mask | col_stats["outlier_mask"]

    # Filter out all detected outlier rows
    df_filtered = df.loc[~combined_outlier_mask].copy()

    total_removed = int(combined_outlier_mask.sum())
    logger.info(
        "IQR outlier filtering complete: removed %d/%d rows (%.1f%%) across columns %s.",
        total_removed,
        len(df),
        (total_removed / len(df) * 100.0) if len(df) > 0 else 0.0,
        target_columns,
    )

    return df_filtered, outlier_stats


# ─────────────────────────────────────────────────────────────────────────────
# 7. Correlation Matrix Analysis
# ─────────────────────────────────────────────────────────────────────────────
def get_correlation_matrix(
    df: pd.DataFrame,
    numerical_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for specified numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing continuous attributes (e.g., prompt_length,
        token_estimate, response_latency_ms, token_cost).
    numerical_cols : Optional[Sequence[str]], default=None
        Columns to include in correlation matrix. If None, automatically selects
        all numerical columns present in df.

    Returns
    -------
    pd.DataFrame
        Square DataFrame containing pairwise Pearson correlation coefficients
        ranging from -1.0 to 1.0. Returns an empty DataFrame if no valid numerical
        columns exist.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a valid pandas DataFrame.")

    if numerical_cols is not None:
        valid_cols = [
            c for c in numerical_cols
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        valid_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not valid_cols:
        logger.warning("No valid numerical columns available for correlation analysis.")
        return pd.DataFrame()

    corr_matrix = df[valid_cols].corr(method="pearson")
    return corr_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Cache Management & Utility Helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_stored_encoder() -> Optional[OneHotEncoder]:
    """Return the globally cached OneHotEncoder instance, or None if not yet fitted."""
    return _STORED_ENCODER


def reset_stored_encoder() -> None:
    """Clear the globally cached OneHotEncoder and its associated metadata."""
    global _STORED_ENCODER, _STORED_CATEGORICAL_COLS, _STORED_ENCODING_MAP, _STORED_FEATURE_NAMES
    _STORED_ENCODER = None
    _STORED_CATEGORICAL_COLS = None
    _STORED_ENCODING_MAP = None
    _STORED_FEATURE_NAMES = None
    logger.info("Reset stored categorical encoder cache.")


# ── Module Public API ─────────────────────────────────────────────────────────
__all__ = [
    "ImputationStat",
    "DEFAULT_NUMERICAL_COLS",
    "DEFAULT_CATEGORICAL_COLS",
    "DEFAULT_IQR_MULTIPLIER",
    "impute_missing_values",
    "encode_categoricals",
    "scale_features",
    "run_full_preprocessing",
    "detect_outliers_iqr",
    "filter_outliers",
    "get_correlation_matrix",
    "get_stored_encoder",
    "reset_stored_encoder",
]
