"""
PromptArmor — Compute Cost & Latency Forecasting Engine
======================================================
Latency and compute cost prediction pipeline for prompt security analysis.
Implements:
  1. Ordinary Least Squares (OLS) Multiple Linear Regression
  2. Ridge Regression (L2 regularization) with GridSearchCV hyperparameter tuning
  3. Lasso Regression (L1 regularization) with feature shrinkage and selection
  4. Model evaluation (MAE, RMSE, R2, residuals)
  5. Feature importance analysis & comparison benchmarking
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default regularization search space for Ridge & Lasso
DEFAULT_ALPHAS: List[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
RANDOM_STATE: int = 42


def _validate_inputs(
    X: Union[np.ndarray, pd.DataFrame, Sequence],
    y: Union[np.ndarray, pd.Series, Sequence],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and convert input training/testing data into clean NumPy arrays.

    Parameters
    ----------
    X : array-like or DataFrame of shape (n_samples, n_features)
        Feature matrix.
    y : array-like or Series of shape (n_samples,)
        Target values.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Converted X and y as 2D and 1D numpy arrays respectively.

    Raises
    ------
    ValueError
        If inputs are empty, None, mismatched lengths, or contain NaNs.
    """
    if X is None or y is None:
        raise ValueError("Feature matrix X and target y cannot be None.")

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64).ravel()

    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    if X_arr.shape[0] == 0:
        raise ValueError("Input feature matrix X has 0 samples.")

    if y_arr.shape[0] == 0:
        raise ValueError("Input target vector y has 0 samples.")

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Sample count mismatch: X has {X_arr.shape[0]} samples, "
            f"while y has {y_arr.shape[0]} samples."
        )

    if np.isnan(X_arr).any() or np.isinf(X_arr).any():
        raise ValueError("Feature matrix X contains NaN or infinite values.")

    if np.isnan(y_arr).any() or np.isinf(y_arr).any():
        raise ValueError("Target vector y contains NaN or infinite values.")

    return X_arr, y_arr


def _resolve_feature_names(
    X: Union[np.ndarray, pd.DataFrame],
    feature_names: Optional[Sequence[str]] = None,
    n_features: Optional[int] = None,
) -> List[str]:
    """
    Extract or generate consistent feature names from DataFrame columns or index.

    Parameters
    ----------
    X : array-like or DataFrame
        Feature matrix.
    feature_names : Optional[Sequence[str]]
        Explicit list of feature names.
    n_features : Optional[int]
        Expected feature count if X is not provided.

    Returns
    -------
    List[str]
        List of feature names matching the feature dimension.
    """
    if feature_names is not None:
        return [str(f) for f in feature_names]

    if isinstance(X, pd.DataFrame):
        return [str(col) for col in X.columns]

    if hasattr(X, "columns"):
        return [str(col) for col in getattr(X, "columns")]

    count = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else (n_features or 1)
    return [f"feature_{i}" for i in range(count)]


# ─────────────────────────────────────────────────────────────────────────────
# 1. OLS Multiple Linear Regression
# ─────────────────────────────────────────────────────────────────────────────

def train_ols(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
) -> LinearRegression:
    """
    Train standard Ordinary Least Squares (OLS) Multiple Linear Regression model.

    Parameters
    ----------
    X_train : array-like or DataFrame of shape (n_samples, n_features)
        Training feature matrix (e.g., prompt token length, complexity, embeddings).
    y_train : array-like or Series of shape (n_samples,)
        Target values (e.g., latency in ms or inference cost in USD).

    Returns
    -------
    LinearRegression
        Fitted scikit-learn LinearRegression model instance.

    Raises
    ------
    ValueError
        If inputs are invalid, empty, or have dimension mismatches.
    """
    X_clean, y_clean = _validate_inputs(X_train, y_train)

    logger.info(
        "Training OLS Linear Regression on %d samples with %d features...",
        X_clean.shape[0],
        X_clean.shape[1],
    )

    ols_model = LinearRegression(fit_intercept=True)
    ols_model.fit(X_clean, y_clean)

    # Attach feature names if available for downstream inspection
    feature_names = _resolve_feature_names(X_train, n_features=X_clean.shape[1])
    setattr(ols_model, "feature_names_in_", np.array(feature_names, dtype=object))

    logger.info(
        "OLS training completed. Intercept: %.4f, Mean coef magnitude: %.4f",
        ols_model.intercept_,
        float(np.mean(np.abs(ols_model.coef_))),
    )
    return ols_model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ridge Regression (L2 Regularization)
# ─────────────────────────────────────────────────────────────────────────────

def train_ridge(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    alphas: Optional[Sequence[float]] = None,
    cv: int = 5,
) -> Ridge:
    """
    Train Ridge (L2 regularized) Regression with GridSearchCV for optimal alpha tuning.

    Parameters
    ----------
    X_train : array-like or DataFrame of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like or Series of shape (n_samples,)
        Target vector.
    alphas : Optional[Sequence[float]], default=None
        Candidate L2 penalty strengths.
        If None, defaults to [0.001, 0.01, 0.1, 1, 10, 100].
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Ridge
        Best fitted Ridge model with tuned alpha attribute (.alpha or .best_alpha_).
    """
    X_clean, y_clean = _validate_inputs(X_train, y_train)
    candidate_alphas = list(alphas) if alphas is not None else list(DEFAULT_ALPHAS)

    # Ensure CV folds do not exceed sample count
    n_samples = X_clean.shape[0]
    effective_cv = max(2, min(cv, n_samples)) if n_samples >= 2 else None

    logger.info(
        "Tuning Ridge Regression across alphas=%s with cv=%s (samples=%d)...",
        candidate_alphas,
        effective_cv,
        n_samples,
    )

    base_ridge = Ridge(random_state=RANDOM_STATE)

    if effective_cv is not None and effective_cv >= 2:
        param_grid = {"alpha": candidate_alphas}
        grid = GridSearchCV(
            estimator=base_ridge,
            param_grid=param_grid,
            cv=effective_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_clean, y_clean)
        best_model: Ridge = grid.best_estimator_
        best_alpha = float(grid.best_params_["alpha"])
        best_score = float(-grid.best_score_)
        logger.info(
            "Ridge tuning complete. Best alpha: %s (Mean CV MSE: %.4f)",
            best_alpha,
            best_score,
        )
        setattr(best_model, "best_alpha_", best_alpha)
        setattr(best_model, "cv_results_", grid.cv_results_)
    else:
        # Fallback if too few samples for k-fold CV
        default_alpha = candidate_alphas[len(candidate_alphas) // 2]
        logger.warning(
            "Insufficient samples for CV (%d). Fitting single Ridge with alpha=%.4f",
            n_samples,
            default_alpha,
        )
        best_model = Ridge(alpha=default_alpha, random_state=RANDOM_STATE)
        best_model.fit(X_clean, y_clean)
        setattr(best_model, "best_alpha_", default_alpha)

    # Preserve feature names
    feature_names = _resolve_feature_names(X_train, n_features=X_clean.shape[1])
    setattr(best_model, "feature_names_in_", np.array(feature_names, dtype=object))

    return best_model


# ─────────────────────────────────────────────────────────────────────────────
# 3. Lasso Regression (L1 Regularization & Feature Shrinkage)
# ─────────────────────────────────────────────────────────────────────────────

def train_lasso(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    alphas: Optional[Sequence[float]] = None,
    cv: int = 5,
) -> Lasso:
    """
    Train Lasso (L1 regularized) Regression with GridSearchCV for alpha tuning.
    Evaluates sparsity and identifies which features are shrunk to zero.

    Parameters
    ----------
    X_train : array-like or DataFrame of shape (n_samples, n_features)
        Training feature matrix.
    y_train : array-like or Series of shape (n_samples,)
        Target vector.
    alphas : Optional[Sequence[float]], default=None
        Candidate L1 penalty strengths.
        If None, defaults to [0.001, 0.01, 0.1, 1, 10, 100].
    cv : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    Lasso
        Best fitted Lasso model with .best_alpha_ and .zero_features_ attributes.
    """
    X_clean, y_clean = _validate_inputs(X_train, y_train)
    candidate_alphas = list(alphas) if alphas is not None else list(DEFAULT_ALPHAS)

    n_samples = X_clean.shape[0]
    effective_cv = max(2, min(cv, n_samples)) if n_samples >= 2 else None

    logger.info(
        "Tuning Lasso Regression across alphas=%s with cv=%s (samples=%d)...",
        candidate_alphas,
        effective_cv,
        n_samples,
    )

    base_lasso = Lasso(random_state=RANDOM_STATE, max_iter=10000, tol=1e-4)

    if effective_cv is not None and effective_cv >= 2:
        param_grid = {"alpha": candidate_alphas}
        grid = GridSearchCV(
            estimator=base_lasso,
            param_grid=param_grid,
            cv=effective_cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_clean, y_clean)
        best_model: Lasso = grid.best_estimator_
        best_alpha = float(grid.best_params_["alpha"])
        best_score = float(-grid.best_score_)
        logger.info(
            "Lasso tuning complete. Best alpha: %s (Mean CV MSE: %.4f)",
            best_alpha,
            best_score,
        )
        setattr(best_model, "best_alpha_", best_alpha)
        setattr(best_model, "cv_results_", grid.cv_results_)
    else:
        default_alpha = candidate_alphas[len(candidate_alphas) // 2]
        logger.warning(
            "Insufficient samples for CV (%d). Fitting single Lasso with alpha=%.4f",
            n_samples,
            default_alpha,
        )
        best_model = Lasso(
            alpha=default_alpha,
            random_state=RANDOM_STATE,
            max_iter=10000,
            tol=1e-4,
        )
        best_model.fit(X_clean, y_clean)
        setattr(best_model, "best_alpha_", default_alpha)

    # Resolve feature names and find zero-shrunk features
    feature_names = _resolve_feature_names(X_train, n_features=X_clean.shape[1])
    setattr(best_model, "feature_names_in_", np.array(feature_names, dtype=object))

    coef = best_model.coef_
    zero_mask = np.isclose(coef, 0.0, atol=1e-5)
    zero_features = [feature_names[i] for i, is_z in enumerate(zero_mask) if is_z]
    non_zero_features = [feature_names[i] for i, is_z in enumerate(zero_mask) if not is_z]

    setattr(best_model, "zero_features_", zero_features)
    setattr(best_model, "non_zero_features_", non_zero_features)

    logger.info(
        "Lasso feature shrinkage report: %d/%d features shrunk to zero: %s",
        len(zero_features),
        len(feature_names),
        zero_features,
    )

    return best_model


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation Functions
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regression(
    model: Any,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics and diagnostics for a regression model.

    Calculates:
      - MAE (Mean Absolute Error)
      - RMSE (Root Mean Squared Error)
      - R2 score (Coefficient of Determination)
      - Predictions vector
      - Residuals vector (y_true - y_pred)

    Parameters
    ----------
    model : Any fitted scikit-learn regressor
        Model providing a `.predict(X)` method.
    X_test : array-like or DataFrame of shape (n_samples, n_features)
        Test feature matrix.
    y_test : array-like or Series of shape (n_samples,)
        Ground truth target values.

    Returns
    -------
    dict
        Dictionary containing:
          - 'mae': float
          - 'rmse': float
          - 'r2': float
          - 'r2_score': float (alias for 'r2')
          - 'predictions': np.ndarray (1D)
          - 'residuals': np.ndarray (1D)

    Raises
    ------
    ValueError
        If inputs are invalid or model lacks predict method.
    """
    if not hasattr(model, "predict"):
        raise ValueError(f"Provided model {type(model)} does not have a predict method.")

    X_clean, y_clean = _validate_inputs(X_test, y_test)

    # Generate predictions
    predictions = np.asarray(model.predict(X_clean), dtype=np.float64).ravel()
    residuals = y_clean - predictions

    # Calculate metrics
    mae = float(mean_absolute_error(y_clean, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_clean, predictions)))
    r2 = float(r2_score(y_clean, predictions))

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "r2_score": r2,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "predictions": predictions,
        "residuals": residuals,
    }


def compare_models(
    models_dict: Dict[str, Any],
    X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    y_test: Optional[Union[np.ndarray, pd.Series]] = None,
) -> pd.DataFrame:
    """
    Compare multiple regression models and return a standardized summary DataFrame.

    Supports passing either:
      1. A dictionary of fitted models: {'OLS': ols_model, 'Ridge': ridge_model, ...}
         (requires X_test and y_test).
      2. A dictionary of model run results: {'ols': {'model': ..., 'metrics': ...}, ...}
         (can reuse precomputed metrics if X_test/y_test are not provided).

    Parameters
    ----------
    models_dict : dict
        Mapping of model labels to model estimators or sub-result dicts.
    X_test : Optional[array-like or DataFrame]
        Test feature matrix.
    y_test : Optional[array-like or Series]
        Test target vector.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Model', 'MAE', 'RMSE', 'R2'], sorted by RMSE ascending.
    """
    rows = []

    for name, item in models_dict.items():
        metrics = None

        # Check if item is already a dictionary containing 'metrics'
        if isinstance(item, dict):
            if "metrics" in item and isinstance(item["metrics"], dict):
                m = item["metrics"]
                if "mae" in m and "rmse" in m and ("r2" in m or "r2_score" in m):
                    metrics = {
                        "MAE": float(m["mae"]),
                        "RMSE": float(m["rmse"]),
                        "R2": float(m.get("r2", m.get("r2_score"))),
                    }

            if metrics is None and "model" in item:
                model_obj = item["model"]
                if X_test is not None and y_test is not None:
                    eval_res = evaluate_regression(model_obj, X_test, y_test)
                    metrics = {
                        "MAE": eval_res["mae"],
                        "RMSE": eval_res["rmse"],
                        "R2": eval_res["r2"],
                    }

        # Check if item is an estimator directly
        elif hasattr(item, "predict"):
            if X_test is not None and y_test is not None:
                eval_res = evaluate_regression(item, X_test, y_test)
                metrics = {
                    "MAE": eval_res["mae"],
                    "RMSE": eval_res["rmse"],
                    "R2": eval_res["r2"],
                }

        if metrics is not None:
            rows.append({
                "Model": str(name).upper() if len(str(name)) <= 5 else str(name).title(),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
            })
        else:
            logger.warning(
                "Could not extract metrics for model '%s'. Ensure X_test and y_test are provided.",
                name,
            )

    df_comparison = pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"])
    if not df_comparison.empty:
        # Best model has lowest RMSE
        df_comparison = df_comparison.sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    return df_comparison


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature Importance Analysis
# ─────────────────────────────────────────────────────────────────────────────

def get_lasso_feature_importance(
    lasso_model: Lasso,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return DataFrame of feature names and their Lasso coefficients, sorted by absolute value.
    Highlights which features were shrunk to zero (sparsity selection).

    Parameters
    ----------
    lasso_model : Lasso
        Fitted scikit-learn Lasso model.
    feature_names : Optional[Sequence[str]]
        Names corresponding to the features. If None, inferred from model or generated.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
          - 'Feature': feature name
          - 'Coefficient': raw model coefficient
          - 'Abs_Coefficient': absolute coefficient value
          - 'Zeroed_Out': boolean flag indicating if feature was shrunk to zero
          - 'Shrunk_To_Zero': alias for Zeroed_Out
    """
    if not hasattr(lasso_model, "coef_"):
        raise ValueError("Lasso model must be fitted before extracting feature importance.")

    coef = np.asarray(lasso_model.coef_, dtype=np.float64).ravel()
    n_features = len(coef)

    # Determine names
    if feature_names is None:
        if hasattr(lasso_model, "feature_names_in_"):
            names = [str(f) for f in getattr(lasso_model, "feature_names_in_")]
        else:
            names = [f"feature_{i}" for i in range(n_features)]
    else:
        names = [str(f) for f in feature_names]

    if len(names) != n_features:
        logger.warning(
            "Feature names count (%d) does not match Lasso coefficient count (%d). Generating generic names.",
            len(names),
            n_features,
        )
        names = [f"feature_{i}" for i in range(n_features)]

    zero_mask = np.isclose(coef, 0.0, atol=1e-5)

    df = pd.DataFrame({
        "Feature": names,
        "Coefficient": coef,
        "Abs_Coefficient": np.abs(coef),
        "Zeroed_Out": zero_mask,
        "Shrunk_To_Zero": zero_mask,
    })

    return df.sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)


def get_ridge_feature_importance(
    ridge_model: Ridge,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return DataFrame of feature names and their Ridge coefficients, sorted by absolute value.

    Parameters
    ----------
    ridge_model : Ridge
        Fitted scikit-learn Ridge model.
    feature_names : Optional[Sequence[str]]
        Names corresponding to the features. If None, inferred from model or generated.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
          - 'Feature': feature name
          - 'Coefficient': raw model coefficient
          - 'Abs_Coefficient': absolute coefficient magnitude
    """
    if not hasattr(ridge_model, "coef_"):
        raise ValueError("Ridge model must be fitted before extracting feature importance.")

    coef = np.asarray(ridge_model.coef_, dtype=np.float64).ravel()
    n_features = len(coef)

    if feature_names is None:
        if hasattr(ridge_model, "feature_names_in_"):
            names = [str(f) for f in getattr(ridge_model, "feature_names_in_")]
        else:
            names = [f"feature_{i}" for i in range(n_features)]
    else:
        names = [str(f) for f in feature_names]

    if len(names) != n_features:
        logger.warning(
            "Feature names count (%d) does not match Ridge coefficient count (%d). Generating generic names.",
            len(names),
            n_features,
        )
        names = [f"feature_{i}" for i in range(n_features)]

    df = pd.DataFrame({
        "Feature": names,
        "Coefficient": coef,
        "Abs_Coefficient": np.abs(coef),
    })

    return df.sort_values(by="Abs_Coefficient", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_regression_pipeline(
    X_train: Union[np.ndarray, pd.DataFrame],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    y_test: Union[np.ndarray, pd.Series],
    feature_names: Optional[Sequence[str]] = None,
    alphas: Optional[Sequence[float]] = None,
    cv: int = 5,
) -> Dict[str, Any]:
    """
    Execute the complete compute cost and latency regression pipeline:
      1. Train OLS, Ridge (tuned), and Lasso (tuned) models.
      2. Evaluate predictions, residuals, MAE, RMSE, and R2 on test data.
      3. Extract best hyperparameters (alpha) and features shrunk to zero.
      4. Build a comprehensive model comparison benchmarking table.

    Parameters
    ----------
    X_train : array-like or DataFrame of shape (n_train_samples, n_features)
        Training feature set.
    X_test : array-like or DataFrame of shape (n_test_samples, n_features)
        Testing feature set.
    y_train : array-like or Series of shape (n_train_samples,)
        Training target values (latency/cost).
    y_test : array-like or Series of shape (n_test_samples,)
        Testing target values (latency/cost).
    feature_names : Optional[Sequence[str]]
        Names for the features. Inferred from DataFrame if omitted.
    alphas : Optional[Sequence[float]]
        Candidate alpha regularization parameters for Ridge & Lasso.
    cv : int, default=5
        Cross-validation folds for grid search.

    Returns
    -------
    dict
        Structured output dictionary:
        {
            'ols': {
                'model': LinearRegression,
                'metrics': {'mae': float, 'rmse': float, 'r2': float, ...},
                'predictions': np.ndarray,
            },
            'ridge': {
                'model': Ridge,
                'metrics': {'mae': float, 'rmse': float, 'r2': float, ...},
                'predictions': np.ndarray,
                'best_alpha': float,
                'feature_importance': pd.DataFrame,
            },
            'lasso': {
                'model': Lasso,
                'metrics': {'mae': float, 'rmse': float, 'r2': float, ...},
                'predictions': np.ndarray,
                'best_alpha': float,
                'zero_features': List[str],
                'feature_importance': pd.DataFrame,
            },
            'comparison': pd.DataFrame
        }
    """
    logger.info("Starting PromptArmor regression pipeline execution...")

    # Resolve feature names
    resolved_features = _resolve_feature_names(
        X_train,
        feature_names=feature_names,
        n_features=X_train.shape[1] if hasattr(X_train, "shape") and len(X_train.shape) > 1 else None,
    )

    # 1. Train OLS
    logger.info("1/3 Fitting Ordinary Least Squares (OLS)...")
    ols_model = train_ols(X_train, y_train)
    ols_eval = evaluate_regression(ols_model, X_test, y_test)

    # 2. Train Ridge
    logger.info("2/3 Fitting Ridge (L2) with alpha tuning...")
    ridge_model = train_ridge(X_train, y_train, alphas=alphas, cv=cv)
    ridge_eval = evaluate_regression(ridge_model, X_test, y_test)
    ridge_alpha = float(getattr(ridge_model, "best_alpha_", getattr(ridge_model, "alpha", 1.0)))
    ridge_fi = get_ridge_feature_importance(ridge_model, feature_names=resolved_features)

    # 3. Train Lasso
    logger.info("3/3 Fitting Lasso (L1) with alpha tuning & shrinkage analysis...")
    lasso_model = train_lasso(X_train, y_train, alphas=alphas, cv=cv)
    lasso_eval = evaluate_regression(lasso_model, X_test, y_test)
    lasso_alpha = float(getattr(lasso_model, "best_alpha_", getattr(lasso_model, "alpha", 1.0)))
    lasso_fi = get_lasso_feature_importance(lasso_model, feature_names=resolved_features)
    zero_features = lasso_fi.loc[lasso_fi["Zeroed_Out"], "Feature"].tolist()

    # 4. Benchmarking comparison table
    models_dict = {
        "OLS": {"model": ols_model, "metrics": ols_eval},
        "Ridge": {"model": ridge_model, "metrics": ridge_eval},
        "Lasso": {"model": lasso_model, "metrics": lasso_eval},
    }
    comparison_df = compare_models(models_dict, X_test=X_test, y_test=y_test)

    logger.info(
        "Regression pipeline successfully completed. Comparison summary:\n%s",
        comparison_df.to_string(index=False),
    )

    return {
        "ols": {
            "model": ols_model,
            "metrics": ols_eval,
            "predictions": ols_eval["predictions"],
        },
        "ridge": {
            "model": ridge_model,
            "metrics": ridge_eval,
            "predictions": ridge_eval["predictions"],
            "best_alpha": ridge_alpha,
            "feature_importance": ridge_fi,
        },
        "lasso": {
            "model": lasso_model,
            "metrics": lasso_eval,
            "predictions": lasso_eval["predictions"],
            "best_alpha": lasso_alpha,
            "zero_features": zero_features,
            "feature_importance": lasso_fi,
        },
        "comparison": comparison_df,
    }
