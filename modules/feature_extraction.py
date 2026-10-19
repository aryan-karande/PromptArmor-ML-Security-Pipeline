"""
PromptArmor — Feature Extraction Module
========================================
Extracts 7 numerical structural security metrics from prompt text for threat
classification and malicious prompt injection detection.

Exposed features:
    1. Override Density: Ratio of security-override keywords to total words.
    2. Symbol Ratio: Ratio of non-alphanumeric, non-space characters to total characters.
    3. Shannon Entropy: Character-level Shannon entropy of the prompt text.
    4. Uppercase Burst: Ratio of uppercase characters to total alphabetic characters.
    5. Prompt Length: Normalized prompt length (len / 1000, capped at 10.0).
    6. Word Complexity: Average length of words in prompt.
    7. Imperative Score: Ratio of imperative/command verbs to total words.
"""

from collections import Counter
import re
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

# ── Feature Definitions & Column Names ───────────────────────────────────────
FEATURE_NAMES: List[str] = [
    'Override Density',
    'Symbol Ratio',
    'Shannon Entropy',
    'Uppercase Burst',
    'Prompt Length',
    'Word Complexity',
    'Imperative Score',
]

# ── Security Keywords ────────────────────────────────────────────────────────
OVERRIDE_KEYWORDS: List[str] = [
    'ignore',
    'override',
    'bypass',
    'disable',
    'unlock',
    'jailbreak',
    'dan',
    'sudo',
    'admin',
    'root',
    'system prompt',
    'previous instructions',
    'forget',
    'disregard',
    'new persona',
    'no restrictions',
    'unrestricted',
]

IMPERATIVE_KEYWORDS: List[str] = [
    'do',
    'make',
    'create',
    'write',
    'generate',
    'tell',
    'show',
    'give',
    'list',
    'explain',
    'describe',
    'output',
    'print',
    'execute',
    'run',
    'pretend',
    'act',
    'become',
    'simulate',
]

# ── Precompiled Regular Expressions ──────────────────────────────────────────
# Compiling patterns at module level optimizes batch feature extraction performance.
def _compile_phrase_pattern(keywords: List[str]) -> re.Pattern:
    """Build a regex pattern matching any keyword with word boundaries,
    handling multi-word phrases with flexible whitespace."""
    patterns = []
    # Sort descending by length so longer phrases match before shorter tokens
    for kw in sorted(keywords, key=len, reverse=True):
        parts = [re.escape(part) for part in kw.split()]
        patterns.append(r'\s+'.join(parts))
    combined = r'\b(?:' + '|'.join(patterns) + r')\b'
    return re.compile(combined, re.IGNORECASE)


OVERRIDE_PATTERN: re.Pattern = _compile_phrase_pattern(OVERRIDE_KEYWORDS)
IMPERATIVE_PATTERN: re.Pattern = _compile_phrase_pattern(IMPERATIVE_KEYWORDS)
WORD_PATTERN: re.Pattern = re.compile(r'\b\w+\b')


# ── Text Cleaning & Preprocessing ───────────────────────────────────────────
def _clean_text(text: Any) -> str:
    """Safely convert any prompt value (including None, NaN, numeric) to string."""
    if text is None:
        return ""
    if isinstance(text, float) and np.isnan(text):
        return ""
    if not isinstance(text, str):
        try:
            if pd.isna(text):
                return ""
        except Exception:
            pass
        return str(text)
    return text


# ── Metric Computation Functions ─────────────────────────────────────────────
def compute_override_density(text: Any, words: Optional[List[str]] = None) -> float:
    """Calculate ratio of security-override keywords to total words.

    Keywords: 'ignore', 'override', 'bypass', 'disable', 'unlock', 'jailbreak',
    'DAN', 'sudo', 'admin', 'root', 'system prompt', 'previous instructions',
    'forget', 'disregard', 'new persona', 'no restrictions', 'unrestricted'.

    Args:
        text: Prompt text.
        words: Optional pre-tokenized words list to avoid duplicate tokenization.

    Returns:
        float: Override density score (>= 0.0).
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return 0.0

    if words is None:
        words = WORD_PATTERN.findall(cleaned)

    if not words:
        return 0.0

    matches = OVERRIDE_PATTERN.findall(cleaned)
    return float(len(matches) / len(words))


def compute_symbol_ratio(text: Any) -> float:
    """Calculate ratio of non-alphanumeric, non-space characters to total characters.

    Captures prompt injection patterns such as <<<>>>, [[[]], ===, etc.

    Args:
        text: Prompt text.

    Returns:
        float: Symbol ratio between 0.0 and 1.0.
    """
    cleaned = _clean_text(text)
    total_len = len(cleaned)
    if total_len == 0:
        return 0.0

    symbol_count = sum(1 for ch in cleaned if not (ch.isalnum() or ch.isspace()))
    return float(symbol_count / total_len)


def compute_shannon_entropy(text: Any) -> float:
    """Calculate character-level Shannon entropy of prompt text.

    Formula: H = -Σ p(x) * log2(p(x)) for each unique character x.
    Higher entropy suggests encoded or obfuscated content (e.g., base64, hex, ciphers).

    Args:
        text: Prompt text.

    Returns:
        float: Shannon entropy score (>= 0.0).
    """
    cleaned = _clean_text(text)
    total_len = len(cleaned)
    if total_len <= 1:
        return 0.0

    counts = np.array(list(Counter(cleaned).values()), dtype=np.float64)
    probs = counts / total_len
    entropy = -np.sum(probs * np.log2(probs))
    return float(max(0.0, entropy))


def compute_uppercase_burst(text: Any) -> float:
    """Calculate ratio of uppercase characters to total alphabetic characters.

    Captures SHOUTING COMMANDS common in jailbreaks.

    Args:
        text: Prompt text.

    Returns:
        float: Ratio between 0.0 and 1.0. Returns 0.0 if no alphabetic characters exist.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return 0.0

    alpha_chars = [ch for ch in cleaned if ch.isalpha()]
    if not alpha_chars:
        return 0.0

    upper_count = sum(1 for ch in alpha_chars if ch.isupper())
    return float(upper_count / len(alpha_chars))


def compute_prompt_length(text: Any) -> float:
    """Calculate normalized prompt length (character count / 1000, capped at 10.0).

    Args:
        text: Prompt text.

    Returns:
        float: Normalized length between 0.0 and 10.0.
    """
    cleaned = _clean_text(text)
    return float(min(len(cleaned) / 1000.0, 10.0))


def compute_word_complexity(text: Any, words: Optional[List[str]] = None) -> float:
    """Calculate average word length in the prompt.

    Technical, obfuscated, or complex prompts tend to have longer words.

    Args:
        text: Prompt text.
        words: Optional pre-tokenized words list to avoid duplicate tokenization.

    Returns:
        float: Average word length (>= 0.0). Returns 0.0 if no words exist.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return 0.0

    if words is None:
        words = WORD_PATTERN.findall(cleaned)

    if not words:
        return 0.0

    return float(np.mean([len(w) for w in words]))


def compute_imperative_score(text: Any, words: Optional[List[str]] = None) -> float:
    """Calculate ratio of imperative/command verbs found in the prompt.

    Keywords: 'do', 'make', 'create', 'write', 'generate', 'tell', 'show', 'give',
    'list', 'explain', 'describe', 'output', 'print', 'execute', 'run',
    'pretend', 'act', 'become', 'simulate'.

    Args:
        text: Prompt text.
        words: Optional pre-tokenized words list to avoid duplicate tokenization.

    Returns:
        float: Imperative score (>= 0.0).
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return 0.0

    if words is None:
        words = WORD_PATTERN.findall(cleaned)

    if not words:
        return 0.0

    matches = IMPERATIVE_PATTERN.findall(cleaned)
    return float(len(matches) / len(words))


# ── Public API Functions ─────────────────────────────────────────────────────
def extract_features(text: str) -> np.ndarray:
    """Extract 7 security features from a single prompt. Returns shape (7,).

    Features in order:
        0. Override Density: Ratio of security-override keywords to total words.
        1. Symbol Ratio: Ratio of non-alphanumeric, non-space characters.
        2. Shannon Entropy: Character-level Shannon entropy.
        3. Uppercase Burst: Ratio of uppercase characters to alphabetic characters.
        4. Prompt Length: Normalized prompt length (len / 1000, capped at 10.0).
        5. Word Complexity: Average length of words in prompt.
        6. Imperative Score: Ratio of imperative/command verbs to total words.

    Args:
        text (str): Prompt text to evaluate. Edge cases (empty string, None,
            single characters, non-strings) are safely handled.

    Returns:
        np.ndarray: 1D numpy array of shape (7,) with dtype float64.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return np.zeros(7, dtype=np.float64)

    words = WORD_PATTERN.findall(cleaned)

    return np.array([
        compute_override_density(cleaned, words=words),
        compute_symbol_ratio(cleaned),
        compute_shannon_entropy(cleaned),
        compute_uppercase_burst(cleaned),
        compute_prompt_length(cleaned),
        compute_word_complexity(cleaned, words=words),
        compute_imperative_score(cleaned, words=words),
    ], dtype=np.float64)


def extract_features_batch(texts: Union[pd.Series, Iterable[Any]]) -> pd.DataFrame:
    """Extract features from a series of prompts. Returns DataFrame with 7 columns.

    Args:
        texts (pd.Series or Iterable): Series or collection of prompt strings.
            Preserves the original series index.

    Returns:
        pd.DataFrame: DataFrame of shape (N, 7) with column names matching
            FEATURE_NAMES and dtype float64.
    """
    if isinstance(texts, pd.Series):
        index = texts.index
        values = texts.tolist()
    elif isinstance(texts, pd.DataFrame):
        first_col = texts.iloc[:, 0]
        index = first_col.index
        values = first_col.tolist()
    else:
        values = list(texts)
        index = None

    if not values:
        return pd.DataFrame(columns=FEATURE_NAMES, index=index, dtype=np.float64)

    feature_matrix = np.array([extract_features(t) for t in values], dtype=np.float64)
    return pd.DataFrame(feature_matrix, columns=FEATURE_NAMES, index=index)


__all__ = [
    'FEATURE_NAMES',
    'OVERRIDE_KEYWORDS',
    'IMPERATIVE_KEYWORDS',
    'extract_features',
    'extract_features_batch',
    'compute_override_density',
    'compute_symbol_ratio',
    'compute_shannon_entropy',
    'compute_uppercase_burst',
    'compute_prompt_length',
    'compute_word_complexity',
    'compute_imperative_score',
]
