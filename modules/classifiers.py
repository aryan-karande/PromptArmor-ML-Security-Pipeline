"""
Threat Classification Module for PromptArmor.
=============================================
This module implements three machine learning classifiers for detecting and
categorizing prompt injection threats into three distinct classes:
    - 0: Safe (benign prompt)
    - 1: Complex Query (legitimate multi-step / long-context prompt)
    - 2: Malicious Jailbreak (adversarial override / injection attack)

All classifiers operate on 7 engineered numerical features:
    1. Override Density
    2. Symbol Ratio
    3. Shannon Entropy
    4. Uppercase Burst
    5. Prompt Length
    6. Word Complexity
    7. Imperative Score

Implemented Classifiers:
    - Deep Neural Network (Multi-Layer Perceptron via TensorFlow/Keras)
    - Interpretable Decision Tree Classifier (Entropy / Information Gain via scikit-learn)
    - K-Nearest Neighbors Classifier (Euclidean metric & CV tuning via scikit-learn)
"""

import os
import warnings

# Suppress TensorFlow C++ logging and oneDNN optimizations messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)

from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Canonical Class and Feature Definitions ──────────────────────────────────
DEFAULT_CLASS_NAMES: List[str] = [
    "Safe",
    "Complex Query",
    "Malicious Jailbreak",
]

DEFAULT_FEATURE_NAMES: List[str] = [
    "Override Density",
    "Symbol Ratio",
    "Shannon Entropy",
    "Uppercase Burst",
    "Prompt Length",
    "Word Complexity",
    "Imperative Score",
]


# ─────────────────────────────────────────────────────────────────────────────
# Deep Neural Network Architecture (TensorFlow/Keras Multi-Layer Perceptron)
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp(
    input_dim: int = 7,
    num_classes: int = 3,
    single_layer: bool = False,
    random_seed: int = 42,
) -> tf.keras.Model:
    """Build a Multi-Layer Perceptron.

    If single_layer=False (default):
        Input(shape=(7,)) -> Dense(24, relu) -> Dense(12, relu) -> Dense(3, softmax)
    If single_layer=True (baseline):
        Input(shape=(7,)) -> Dense(3, softmax)

    Compile with optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'].

    Parameters
    ----------
    input_dim : int, default=7
        Number of input features.
    num_classes : int, default=3
        Number of output target classes.
    single_layer : bool, default=False
        If True, builds a single-layer baseline model. If False, builds a 3-layer MLP.
    random_seed : int, default=42
        Random seed for reproducibility of weight initializers.

    Returns
    -------
    tf.keras.Model
        Compiled Keras model ready for training.
    """
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if num_classes <= 1:
        raise ValueError(f"num_classes must be greater than 1, got {num_classes}")

    tf.keras.utils.set_random_seed(random_seed)

    if single_layer:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer"),
        ], name="single_layer_perceptron")
    else:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(24, activation="relu", name="dense_24"),
            tf.keras.layers.Dense(12, activation="relu", name="dense_12"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer"),
        ], name="deep_mlp")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_mlp(
    model: tf.keras.Model,
    X_train: Union[np.ndarray, Sequence, Any],
    y_train: Union[np.ndarray, Sequence, Any],
    X_val: Union[np.ndarray, Sequence, Any],
    y_val: Union[np.ndarray, Sequence, Any],
    epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 0,
) -> tf.keras.callbacks.History:
    """Train with EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True).

    Return training history.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled Keras model to train.
    X_train : array-like of shape (n_samples, n_features)
        Training feature vectors.
    y_train : array-like of shape (n_samples,)
        Training class labels (integers 0, 1, 2).
    X_val : array-like of shape (n_val_samples, n_features)
        Validation feature vectors.
    y_val : array-like of shape (n_val_samples,)
        Validation class labels.
    epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=32
        Mini-batch size for gradient descent.
    verbose : int, default=0
        Verbosity mode (0=silent, 1=progress bar, 2=one line per epoch).

    Returns
    -------
    tf.keras.callbacks.History
        Training history object recording loss and metrics per epoch.
    """
    X_train_arr = np.asarray(X_train, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.int32)
    X_val_arr = np.asarray(X_val, dtype=np.float32)
    y_val_arr = np.asarray(y_val, dtype=np.int32)

    if len(X_train_arr) != len(y_train_arr):
        raise ValueError(
            f"X_train length ({len(X_train_arr)}) does not match y_train length ({len(y_train_arr)})."
        )
    if len(X_val_arr) != len(y_val_arr):
        raise ValueError(
            f"X_val length ({len(X_val_arr)}) does not match y_val length ({len(y_val_arr)})."
        )
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=verbose,
    )

    history = model.fit(
        X_train_arr,
        y_train_arr,
        validation_data=(X_val_arr, y_val_arr),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose,
    )
    return history


def get_mlp_results(
    model: tf.keras.Model,
    X_test: Union[np.ndarray, Sequence, Any],
    y_test: Union[np.ndarray, Sequence, Any],
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return predictions, probabilities, accuracy, classification_report, confusion_matrix.

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    X_test : array-like of shape (n_samples, n_features)
        Test feature vectors.
    y_test : array-like of shape (n_samples,)
        True ground-truth labels.
    target_names : sequence of str, optional
        Display names for target classes. Defaults to ["Safe", "Complex Query", "Malicious Jailbreak"].

    Returns
    -------
    dict
        Dictionary containing:
        - 'predictions': np.ndarray of shape (n_samples,)
        - 'probabilities': np.ndarray of shape (n_samples, n_classes)
        - 'accuracy': float
        - 'classification_report': str
        - 'report_dict': dict
        - 'confusion_matrix': np.ndarray of shape (n_classes, n_classes)
    """
    X_test_arr = np.asarray(X_test, dtype=np.float32)
    y_test_arr = np.asarray(y_test, dtype=np.int32)

    if len(X_test_arr) != len(y_test_arr):
        raise ValueError(
            f"X_test length ({len(X_test_arr)}) does not match y_test length ({len(y_test_arr)})."
        )

    if target_names is None:
        target_names = DEFAULT_CLASS_NAMES

    labels = list(range(len(target_names)))

    probabilities = model.predict(X_test_arr, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    acc = float(accuracy_score(y_test_arr, predictions))
    report_str = classification_report(
        y_test_arr,
        predictions,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test_arr,
        predictions,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_arr, predictions, labels=labels)

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": acc,
        "classification_report": report_str,
        "report_dict": report_dict,
        "confusion_matrix": cm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Interpretable Decision Tree Policy Engine (scikit-learn)
# ─────────────────────────────────────────────────────────────────────────────

def train_decision_tree(
    X_train: Union[np.ndarray, Sequence, Any],
    y_train: Union[np.ndarray, Sequence, Any],
    max_depth: Optional[int] = None,
    criterion: str = "entropy",
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Train a DecisionTreeClassifier with given max_depth and criterion (entropy for Information Gain).

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature vectors.
    y_train : array-like of shape (n_samples,)
        Training class labels.
    max_depth : int, optional
        Maximum tree depth. If None, nodes are expanded until all leaves are pure.
    criterion : str, default='entropy'
        Function to measure split quality ('entropy' for Information Gain or 'gini').
    random_state : int, default=42
        Seed for reproducibility.

    Returns
    -------
    DecisionTreeClassifier
        Fitted Decision Tree model.
    """
    if max_depth is not None and max_depth <= 0:
        raise ValueError(f"max_depth must be positive or None, got {max_depth}")

    X_train_arr = np.asarray(X_train)
    y_train_arr = np.asarray(y_train)

    if len(X_train_arr) != len(y_train_arr):
        raise ValueError(
            f"X_train length ({len(X_train_arr)}) does not match y_train length ({len(y_train_arr)})."
        )

    clf = DecisionTreeClassifier(
        criterion=criterion,
        max_depth=max_depth,
        random_state=random_state,
    )
    clf.fit(X_train_arr, y_train_arr)
    return clf


def get_tree_rules(
    tree_model: DecisionTreeClassifier,
    feature_names: Optional[Sequence[str]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> str:
    """Export tree rules as text using sklearn.tree.export_text.

    Parameters
    ----------
    tree_model : DecisionTreeClassifier
        Fitted Decision Tree model.
    feature_names : sequence of str, optional
        Names of the input features. Defaults to DEFAULT_FEATURE_NAMES if None.
    class_names : sequence of str, optional
        Names of the target classes. Defaults to DEFAULT_CLASS_NAMES if None.

    Returns
    -------
    str
        Textual representation of the decision tree rules.
    """
    if not hasattr(tree_model, "tree_"):
        raise ValueError("tree_model must be fitted before exporting rules.")

    feat_names = list(feature_names) if feature_names is not None else DEFAULT_FEATURE_NAMES
    cls_names = [str(c) for c in class_names] if class_names is not None else DEFAULT_CLASS_NAMES

    return export_text(
        tree_model,
        feature_names=feat_names,
        class_names=cls_names,
    )


def get_tree_results(
    tree_model: DecisionTreeClassifier,
    X_test: Union[np.ndarray, Sequence, Any],
    y_test: Union[np.ndarray, Sequence, Any],
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return predictions, accuracy, classification_report, confusion_matrix, feature_importances.

    Parameters
    ----------
    tree_model : DecisionTreeClassifier
        Fitted Decision Tree model.
    X_test : array-like of shape (n_samples, n_features)
        Test feature vectors.
    y_test : array-like of shape (n_samples,)
        Ground-truth labels.
    target_names : sequence of str, optional
        Target class names for classification report.

    Returns
    -------
    dict
        Dictionary containing:
        - 'predictions': np.ndarray of shape (n_samples,)
        - 'probabilities': np.ndarray of shape (n_samples, n_classes)
        - 'accuracy': float
        - 'classification_report': str
        - 'report_dict': dict
        - 'confusion_matrix': np.ndarray of shape (n_classes, n_classes)
        - 'feature_importances': np.ndarray of shape (n_features,)
    """
    X_test_arr = np.asarray(X_test)
    y_test_arr = np.asarray(y_test)

    if len(X_test_arr) != len(y_test_arr):
        raise ValueError(
            f"X_test length ({len(X_test_arr)}) does not match y_test length ({len(y_test_arr)})."
        )

    if target_names is None:
        target_names = DEFAULT_CLASS_NAMES

    labels = list(range(len(target_names)))

    predictions = tree_model.predict(X_test_arr)
    probabilities = tree_model.predict_proba(X_test_arr) if hasattr(tree_model, "predict_proba") else None
    acc = float(accuracy_score(y_test_arr, predictions))

    report_str = classification_report(
        y_test_arr,
        predictions,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test_arr,
        predictions,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_arr, predictions, labels=labels)
    feature_importances = getattr(tree_model, "feature_importances_", np.array([]))

    results: Dict[str, Any] = {
        "predictions": predictions,
        "accuracy": acc,
        "classification_report": report_str,
        "report_dict": report_dict,
        "confusion_matrix": cm,
        "feature_importances": feature_importances,
    }
    if probabilities is not None:
        results["probabilities"] = probabilities

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Nearest-Neighbor Similarity Matcher (scikit-learn KNN)
# ─────────────────────────────────────────────────────────────────────────────

def train_knn(
    X_train: Union[np.ndarray, Sequence, Any],
    y_train: Union[np.ndarray, Sequence, Any],
    k: int = 5,
    metric: str = "euclidean",
) -> KNeighborsClassifier:
    """Train KNN with given k and distance metric.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature vectors.
    y_train : array-like of shape (n_samples,)
        Training class labels.
    k : int, default=5
        Number of nearest neighbors.
    metric : str, default='euclidean'
        Distance metric ('euclidean', 'manhattan', 'minkowski', etc.).

    Returns
    -------
    KNeighborsClassifier
        Fitted KNeighborsClassifier instance.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    X_train_arr = np.asarray(X_train)
    y_train_arr = np.asarray(y_train)

    if len(X_train_arr) == 0:
        raise ValueError("X_train cannot be empty.")
    if len(X_train_arr) != len(y_train_arr):
        raise ValueError(
            f"X_train length ({len(X_train_arr)}) does not match y_train length ({len(y_train_arr)})."
        )
    if k > len(X_train_arr):
        raise ValueError(
            f"k ({k}) cannot be greater than the number of training samples ({len(X_train_arr)})."
        )

    knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
    knn.fit(X_train_arr, y_train_arr)
    return knn


def knn_cross_validate(
    X_train: Union[np.ndarray, Sequence, Any],
    y_train: Union[np.ndarray, Sequence, Any],
    k_range: Sequence[int] = range(1, 31),
    cv: int = 5,
    metric: str = "euclidean",
) -> Dict[int, float]:
    """Cross-validate KNN for each k in k_range. Return {k: mean_cv_accuracy}.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training feature vectors.
    y_train : array-like of shape (n_samples,)
        Training class labels.
    k_range : sequence of int, default=range(1, 31)
        Sequence of neighbor counts (k) to evaluate.
    cv : int, default=5
        Number of cross-validation folds.
    metric : str, default='euclidean'
        Distance metric.

    Returns
    -------
    dict
        Dictionary mapping each integer k to its mean cross-validation accuracy.
    """
    X_train_arr = np.asarray(X_train)
    y_train_arr = np.asarray(y_train)
    n_samples = len(X_train_arr)

    if n_samples == 0:
        raise ValueError("X_train cannot be empty for cross-validation.")
    if len(X_train_arr) != len(y_train_arr):
        raise ValueError(
            f"X_train length ({len(X_train_arr)}) does not match y_train length ({len(y_train_arr)})."
        )

    actual_cv = min(cv, n_samples)
    if actual_cv < 2:
        raise ValueError("Cross-validation requires at least 2 samples.")

    cv_results: Dict[int, float] = {}
    for k in k_range:
        if k <= 0 or k > n_samples:
            continue
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        scores = cross_val_score(knn, X_train_arr, y_train_arr, cv=actual_cv, scoring="accuracy")
        cv_results[int(k)] = float(np.mean(scores))

    return cv_results


def get_knn_results(
    knn_model: KNeighborsClassifier,
    X_test: Union[np.ndarray, Sequence, Any],
    y_test: Union[np.ndarray, Sequence, Any],
    target_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Return predictions, accuracy, classification_report, confusion_matrix.

    Parameters
    ----------
    knn_model : KNeighborsClassifier
        Fitted KNN model.
    X_test : array-like of shape (n_samples, n_features)
        Test feature vectors.
    y_test : array-like of shape (n_samples,)
        Ground-truth labels.
    target_names : sequence of str, optional
        Target class names for classification report.

    Returns
    -------
    dict
        Dictionary containing:
        - 'predictions': np.ndarray of shape (n_samples,)
        - 'probabilities': np.ndarray of shape (n_samples, n_classes)
        - 'accuracy': float
        - 'classification_report': str
        - 'report_dict': dict
        - 'confusion_matrix': np.ndarray of shape (n_classes, n_classes)
    """
    X_test_arr = np.asarray(X_test)
    y_test_arr = np.asarray(y_test)

    if len(X_test_arr) != len(y_test_arr):
        raise ValueError(
            f"X_test length ({len(X_test_arr)}) does not match y_test length ({len(y_test_arr)})."
        )

    if target_names is None:
        target_names = DEFAULT_CLASS_NAMES

    labels = list(range(len(target_names)))

    predictions = knn_model.predict(X_test_arr)
    probabilities = knn_model.predict_proba(X_test_arr) if hasattr(knn_model, "predict_proba") else None
    acc = float(accuracy_score(y_test_arr, predictions))

    report_str = classification_report(
        y_test_arr,
        predictions,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_test_arr,
        predictions,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test_arr, predictions, labels=labels)

    results: Dict[str, Any] = {
        "predictions": predictions,
        "accuracy": acc,
        "classification_report": report_str,
        "report_dict": report_dict,
        "confusion_matrix": cm,
    }
    if probabilities is not None:
        results["probabilities"] = probabilities

    return results


def find_nearest_neighbors(
    knn_model: KNeighborsClassifier,
    X_train: Union[np.ndarray, Sequence, Any],
    query_point: Union[np.ndarray, Sequence, Any],
    k: int = 5,
) -> np.ndarray:
    """Return indices of k nearest neighbors to query_point.

    Parameters
    ----------
    knn_model : KNeighborsClassifier
        Fitted KNN classifier.
    X_train : array-like of shape (n_samples, n_features)
        Reference training feature vectors.
    query_point : array-like of shape (n_features,) or (1, n_features)
        Query feature vector.
    k : int, default=5
        Number of nearest neighbors to retrieve.

    Returns
    -------
    np.ndarray
        Indices of the k nearest neighbors in the training set (shape: (k,)).
    """
    query_arr = np.asarray(query_point, dtype=np.float64)
    if query_arr.ndim == 1:
        query_arr = query_arr.reshape(1, -1)

    if not hasattr(knn_model, "_fit_X") and X_train is not None:
        X_train_arr = np.asarray(X_train, dtype=np.float64)
        dummy_y = np.zeros(len(X_train_arr), dtype=int)
        knn_model.fit(X_train_arr, dummy_y)

    indices = knn_model.kneighbors(query_arr, n_neighbors=k, return_distance=False)
    if indices.ndim > 1 and indices.shape[0] == 1:
        return indices[0]
    return indices


__all__ = [
    "DEFAULT_CLASS_NAMES",
    "DEFAULT_FEATURE_NAMES",
    "build_mlp",
    "train_mlp",
    "get_mlp_results",
    "train_decision_tree",
    "get_tree_rules",
    "get_tree_results",
    "train_knn",
    "knn_cross_validate",
    "get_knn_results",
    "find_nearest_neighbors",
]
