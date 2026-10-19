"""
PromptArmor — Symbolic Policy Guardrails & Version Space Boundary Engine
========================================================================
Implements concept learning algorithms (Find-S and Candidate Elimination)
to learn and maintain the version space of valid prompt firewall policies.

Attributes:
    user_tier       : ['Free', 'Pro', 'Enterprise']
    intent_type     : ['Question', 'Instruction', 'Creative', 'Analysis']
    code_execution  : ['Yes', 'No']
    auth_state      : ['Authenticated', 'Anonymous']
    allowed (target): bool (True = Allowed through firewall, False = Blocked)
"""

from typing import Tuple, List, Dict, Any, Optional
import pandas as pd


# ── Policy Attribute Definitions ─────────────────────────────────────────────
ATTRIBUTE_NAMES: List[str] = [
    "user_tier",
    "intent_type",
    "code_execution",
    "auth_state",
]

ATTRIBUTE_DOMAINS: Dict[str, List[str]] = {
    "user_tier": ["Free", "Pro", "Enterprise"],
    "intent_type": ["Question", "Instruction", "Creative", "Analysis"],
    "code_execution": ["Yes", "No"],
    "auth_state": ["Authenticated", "Anonymous"],
}

TARGET_COL: str = "allowed"


# ── Helper Functions for Hypothesis Operations ───────────────────────────────
def is_more_general_or_equal(h1: list, h2: list) -> bool:
    """Check if hypothesis h1 is more general than or equal to hypothesis h2.

    A hypothesis h1 is more general than or equal to h2 iff for every attribute i,
    h1[i] == '?' or h1[i] == h2[i].

    Parameters
    ----------
    h1 : list
        Candidate more general hypothesis (e.g., ['Enterprise', '?', '?', '?']).
    h2 : list
        Hypothesis or instance to compare against.

    Returns
    -------
    bool
        True if h1 is more general than or equal to h2, False otherwise.
    """
    if len(h1) != len(h2):
        return False
    for val1, val2 in zip(h1, h2):
        if val1 != "?" and val1 != val2:
            return False
    return True


def covers(h: list, instance: list) -> bool:
    """Check if hypothesis h covers (matches / classifies as positive) an instance.

    Parameters
    ----------
    h : list
        Hypothesis with attribute constraints and/or '?' wildcards.
    instance : list
        Instance attribute values (without target label).

    Returns
    -------
    bool
        True if hypothesis h covers the instance, False otherwise.
    """
    return is_more_general_or_equal(h, instance)


def format_hypothesis(h: list) -> str:
    """Format a hypothesis as a readable string like <Enterprise, ?, No, Authenticated>.

    Parameters
    ----------
    h : list
        Hypothesis as a list of attribute constraints, or list of hypotheses.

    Returns
    -------
    str
        Human-readable representation (e.g., '<Enterprise, ?, No, Authenticated>').
    """
    if h is None or len(h) == 0:
        return "<∅>"

    # Handle nested hypothesis list (e.g., if G or list of hypotheses is passed)
    if isinstance(h[0], list):
        if len(h) == 1:
            return format_hypothesis(h[0])
        return ", ".join(format_hypothesis(sub) for sub in h)

    # Handle null hypothesis marker
    if all(x == "Φ" for x in h):
        return "<∅>"

    return f"<{', '.join(str(val) for val in h)}>"


def check_version_space_collapse(S: list, G: list) -> bool:
    """Check if the version space has collapsed (S is more general than G or G is empty).

    The version space collapses when no hypothesis exists in H that is consistent
    with all observed training instances. Formally, this occurs when:
    1. G becomes empty (all general hypotheses were eliminated by negative examples).
    2. S becomes empty (S covered a negative example and could not be specialized).
    3. No hypothesis in G is more general than or equal to S (S and G crossed over).

    Parameters
    ----------
    S : list
        The most specific hypothesis (or list of specific hypotheses).
    G : list
        The list of most general hypotheses.

    Returns
    -------
    bool
        True if the version space has collapsed, False otherwise.
    """
    # 1. G is empty
    if not G or len(G) == 0:
        return True

    # 2. S is empty
    if not S or len(S) == 0:
        return True

    # Normalize S if passed as list of hypotheses
    s_hyp = S[0] if (len(S) > 0 and isinstance(S[0], list)) else S
    if not s_hyp or len(s_hyp) == 0 or "Φ" in s_hyp:
        return True

    # 3. Check if at least one g in G is more general than or equal to S
    for g in G:
        if is_more_general_or_equal(g, s_hyp):
            return False

    return True


# ── Training Data Generation ─────────────────────────────────────────────────
def generate_policy_examples(inject_noise: bool = False) -> pd.DataFrame:
    """Generate ~20 labeled examples of security policy requests for concept learning.

    The examples represent a sensible security policy guardrail:
    - Enterprise tier prompts with safe execution (code_execution='No') by
      Authenticated users are ALLOWED through the firewall across all intent types.
    - Anonymous requests are always BLOCKED.
    - Code execution attempts on this standard endpoint are BLOCKED.
    - Unprivileged tiers (Free, Pro) on this enterprise policy boundary are BLOCKED.

    If `inject_noise=True`, contradictory examples (identical attributes but
    opposite label) are appended to demonstrate Version Space collapse.

    Parameters
    ----------
    inject_noise : bool, default=False
        If True, injects 2-3 contradictory examples that cause Version Space collapse.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['user_tier', 'intent_type', 'code_execution',
        'auth_state', 'allowed'].
    """
    # Clean, consistent policy examples (interspersed positive and negative)
    # Target concept: <Enterprise, ?, No, Authenticated>
    data = [
        # Step 1: Initial positive example -> S = <Enterprise, Question, No, Authenticated>
        {"user_tier": "Enterprise", "intent_type": "Question", "code_execution": "No", "auth_state": "Authenticated", "allowed": True},
        # Step 2: Negative tier violation -> G specializes to <Enterprise, ?, ?, ?>
        {"user_tier": "Free", "intent_type": "Question", "code_execution": "No", "auth_state": "Authenticated", "allowed": False},
        # Step 3: Positive instruction -> S generalizes to <Enterprise, ?, No, Authenticated>
        {"user_tier": "Enterprise", "intent_type": "Instruction", "code_execution": "No", "auth_state": "Authenticated", "allowed": True},
        # Step 4: Negative code execution -> G specializes to <Enterprise, ?, No, ?>
        {"user_tier": "Enterprise", "intent_type": "Question", "code_execution": "Yes", "auth_state": "Authenticated", "allowed": False},
        # Step 5: Negative anonymous auth -> G specializes to <Enterprise, ?, No, Authenticated>
        {"user_tier": "Enterprise", "intent_type": "Question", "code_execution": "No", "auth_state": "Anonymous", "allowed": False},
        # Step 6: Positive creative -> S confirms ? on intent_type
        {"user_tier": "Enterprise", "intent_type": "Creative", "code_execution": "No", "auth_state": "Authenticated", "allowed": True},
        # Step 7: Positive analysis -> S confirms ? on intent_type
        {"user_tier": "Enterprise", "intent_type": "Analysis", "code_execution": "No", "auth_state": "Authenticated", "allowed": True},
        # Step 8: Anonymous violation from Pro tier
        {"user_tier": "Pro", "intent_type": "Analysis", "code_execution": "No", "auth_state": "Anonymous", "allowed": False},
        # Step 9: Anonymous violation from Free tier
        {"user_tier": "Free", "intent_type": "Creative", "code_execution": "No", "auth_state": "Anonymous", "allowed": False},
        # Step 10: Anonymous violation from Enterprise tier
        {"user_tier": "Enterprise", "intent_type": "Instruction", "code_execution": "No", "auth_state": "Anonymous", "allowed": False},
        # Step 11: Code execution attempt from Free tier
        {"user_tier": "Free", "intent_type": "Question", "code_execution": "Yes", "auth_state": "Authenticated", "allowed": False},
        # Step 12: Code execution attempt from Pro tier
        {"user_tier": "Pro", "intent_type": "Instruction", "code_execution": "Yes", "auth_state": "Authenticated", "allowed": False},
        # Step 13: Code execution attempt from Enterprise tier
        {"user_tier": "Enterprise", "intent_type": "Instruction", "code_execution": "Yes", "auth_state": "Authenticated", "allowed": False},
        # Step 14: Tier violation: Free user asking question
        {"user_tier": "Free", "intent_type": "Question", "code_execution": "No", "auth_state": "Authenticated", "allowed": False},
        # Step 15: Tier violation: Free user sending instruction
        {"user_tier": "Free", "intent_type": "Instruction", "code_execution": "No", "auth_state": "Authenticated", "allowed": False},
        # Step 16: Tier violation: Pro user asking question
        {"user_tier": "Pro", "intent_type": "Question", "code_execution": "No", "auth_state": "Authenticated", "allowed": False},
        # Step 17: Tier violation: Pro user requesting analysis
        {"user_tier": "Pro", "intent_type": "Analysis", "code_execution": "No", "auth_state": "Authenticated", "allowed": False},
        # Step 18: Code execution attempt from Enterprise analysis
        {"user_tier": "Enterprise", "intent_type": "Analysis", "code_execution": "Yes", "auth_state": "Authenticated", "allowed": False},
        # Step 19: Positive confirmation: Enterprise question
        {"user_tier": "Enterprise", "intent_type": "Question", "code_execution": "No", "auth_state": "Authenticated", "allowed": True},
        # Step 20: Positive confirmation: Enterprise analysis
        {"user_tier": "Enterprise", "intent_type": "Analysis", "code_execution": "No", "auth_state": "Authenticated", "allowed": True},
    ]

    df = pd.DataFrame(data)

    if inject_noise:
        # Add 3 contradictory examples with identical attributes to earlier positive
        # examples but opposite label (allowed=False), triggering Version Space collapse.
        noisy_contradictions = [
            # Contradicts Step 1 / Step 19
            {"user_tier": "Enterprise", "intent_type": "Question", "code_execution": "No", "auth_state": "Authenticated", "allowed": False},
            # Contradicts Step 3
            {"user_tier": "Enterprise", "intent_type": "Instruction", "code_execution": "No", "auth_state": "Authenticated", "allowed": False},
            # Contradicts Step 7 / Step 20
            {"user_tier": "Enterprise", "intent_type": "Analysis", "code_execution": "No", "auth_state": "Authenticated", "allowed": False},
        ]
        df_noise = pd.DataFrame(noisy_contradictions)
        df = pd.concat([df, df_noise], ignore_index=True)

    return df


# ── Find-S Algorithm ─────────────────────────────────────────────────────────
def find_s(examples: pd.DataFrame) -> Tuple[list, list]:
    """Implement Find-S algorithm to find the most specific consistent hypothesis.

    Find-S initializes S to the first positive training instance and generalizes S
    minimally whenever a subsequent positive example is not covered. Negative
    examples are completely ignored by Find-S.

    Parameters
    ----------
    examples : pd.DataFrame
        Training examples with policy attributes and 'allowed' target column.

    Returns
    -------
    Tuple[list, list]
        (final_hypothesis, history_of_hypotheses)
        - final_hypothesis: list of attribute values or '?'
        - history_of_hypotheses: list of hypothesis states after each positive example
    """
    feature_cols = [c for c in ATTRIBUTE_NAMES if c in examples.columns]
    if not feature_cols:
        feature_cols = [c for c in examples.columns if c != TARGET_COL]

    # Find-S only processes positive examples
    pos_df = examples[examples[TARGET_COL] == True]
    if pos_df.empty:
        return [], []

    # Initialize S to the first positive example
    first_row = pos_df.iloc[0]
    S: List[str] = [str(first_row[col]) for col in feature_cols]
    history: List[List[str]] = [list(S)]

    # For each subsequent positive example, generalize S minimally
    for idx in range(1, len(pos_df)):
        row = pos_df.iloc[idx]
        for i, col in enumerate(feature_cols):
            val = str(row[col])
            if S[i] != val:
                S[i] = "?"
        history.append(list(S))

    return S, history


# ── Candidate Elimination Algorithm ──────────────────────────────────────────
def candidate_elimination(examples: pd.DataFrame) -> Tuple[list, list, list, list]:
    """Implement Candidate Elimination algorithm for concept learning.

    Maintains the dual boundaries:
    - S: Most specific hypothesis space, initialized to the first positive example.
    - G: Most general hypothesis space, initialized to [['?', '?', '?', '?']].

    For each training example:
      - If positive: Generalize S minimally to cover instance; remove inconsistent from G.
      - If negative: Specialize G minimally to exclude instance; remove inconsistent from S.

    Parameters
    ----------
    examples : pd.DataFrame
        Training examples with policy attributes and 'allowed' target column.

    Returns
    -------
    Tuple[list, list, list, list]
        (final_S, final_G, S_history, G_history)
        - final_S: final most specific hypothesis (or empty list if collapsed)
        - final_G: final list of most general hypotheses (or empty list if collapsed)
        - S_history: snapshot of S at each step
        - G_history: snapshot of G at each step
    """
    feature_cols = [c for c in ATTRIBUTE_NAMES if c in examples.columns]
    if not feature_cols:
        feature_cols = [c for c in examples.columns if c != TARGET_COL]

    n = len(feature_cols)
    pos_examples = examples[examples[TARGET_COL] == True]

    if pos_examples.empty:
        S = ["Φ"] * n
        G = [["?"] * n]
        return S, G, [list(S)], [list(g) for g in G]

    # Initialize:
    # S = most specific hypothesis (all first positive example values)
    # G = most general hypothesis [['?', '?', '?', '?']]
    first_pos = pos_examples.iloc[0]
    S: List[str] = [str(first_pos[col]) for col in feature_cols]
    G: List[List[str]] = [["?"] * n]

    S_history: List[List[str]] = []
    G_history: List[List[List[str]]] = []
    is_collapsed = False

    for idx in range(len(examples)):
        row = examples.iloc[idx]
        x = [str(row[col]) for col in feature_cols]
        label = bool(row[TARGET_COL])

        if is_collapsed:
            # Once collapsed, record empty sets for remaining history
            S_history.append([])
            G_history.append([])
            continue

        if label:
            # ── Positive Example ─────────────────────────────────────────────
            # 1. Generalize S minimally to cover x
            for i in range(n):
                if S[i] != x[i]:
                    S[i] = "?"

            # 2. Remove from G any hypothesis inconsistent with x (doesn't cover x)
            G = [g for g in G if covers(g, x)]

            # 3. Check consistency: S must remain bounded by at least one g in G
            if len(G) == 0 or not any(is_more_general_or_equal(g, S) for g in G):
                is_collapsed = True
                S = []
                G = []

        else:
            # ── Negative Example ─────────────────────────────────────────────
            # 1. Remove inconsistent from S: if S covers negative example x, S is invalid
            if covers(S, x):
                is_collapsed = True
                S = []
                G = []
            else:
                # 2. Specialize G minimally to exclude negative example x
                new_G: List[List[str]] = []
                for g in G:
                    if not covers(g, x):
                        new_G.append(g)
                    else:
                        # Form minimal specializations of g that do not cover x
                        for i in range(n):
                            if g[i] == "?":
                                # To remain more general than S, specialized attribute must match S[i]
                                if S[i] != "?" and S[i] != x[i]:
                                    h = list(g)
                                    h[i] = S[i]
                                    new_G.append(h)

                # Retain only specializations that are more general than or equal to S
                new_G = [h for h in new_G if is_more_general_or_equal(h, S)]

                # Remove duplicate hypotheses
                unique_G: List[List[str]] = []
                for h in new_G:
                    if h not in unique_G:
                        unique_G.append(h)

                # Remove from unique_G any hypothesis that is strictly less general than another
                pruned_G: List[List[str]] = []
                for i, h1 in enumerate(unique_G):
                    less_general = False
                    for j, h2 in enumerate(unique_G):
                        if i != j and is_more_general_or_equal(h2, h1):
                            less_general = True
                            break
                    if not less_general:
                        pruned_G.append(h1)

                G = pruned_G

                # If G becomes empty, version space has collapsed
                if len(G) == 0:
                    is_collapsed = True
                    S = []
                    G = []

        # Record history snapshot for this step
        S_history.append(list(S) if S else [])
        G_history.append([list(g) for g in G] if G else [])

    return S, G, S_history, G_history


# ── Policy Evaluation Helpers ────────────────────────────────────────────────
def predict_instance(hypothesis: list, instance: Dict[str, str]) -> bool:
    """Classify an incoming prompt request under a specific hypothesis.

    Parameters
    ----------
    hypothesis : list
        Attribute constraints (e.g., ['Enterprise', '?', 'No', 'Authenticated']).
    instance : dict
        Dictionary of attribute values (e.g., {'user_tier': 'Enterprise', ...}).

    Returns
    -------
    bool
        True if the request is permitted by the hypothesis, False otherwise.
    """
    if not hypothesis or "Φ" in hypothesis:
        return False
    inst_list = [instance.get(col, "") for col in ATTRIBUTE_NAMES]
    return covers(hypothesis, inst_list)


def predict_version_space(S: list, G: list, instance: Dict[str, str]) -> str:
    """Classify an instance using Version Space boundary consensus.

    - "Allowed"   : Both S and all hypotheses in G classify the instance as True.
    - "Blocked"   : Neither S nor any hypothesis in G classifies the instance as True.
    - "Uncertain" : Disagreement within the version space boundary.

    Parameters
    ----------
    S : list
        Most specific hypothesis.
    G : list
        List of most general hypotheses.
    instance : dict
        Dictionary of attribute values.

    Returns
    -------
    str
        'Allowed', 'Blocked', or 'Uncertain'.
    """
    if not S or not G:
        return "Blocked"

    inst_list = [instance.get(col, "") for col in ATTRIBUTE_NAMES]
    s_pred = covers(S, inst_list)
    g_preds = [covers(g, inst_list) for g in G]

    if s_pred and all(g_preds):
        return "Allowed"
    elif not s_pred and not any(g_preds):
        return "Blocked"
    else:
        return "Uncertain"
