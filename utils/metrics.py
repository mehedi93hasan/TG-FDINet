"""
Evaluation Metrics for TG-FDINet.

Implements the primary and secondary metrics from the paper:
    - Sensitivity (Recall)    — primary metric (patient safety priority)
    - Precision
    - F1-Score
    - AUC-ROC

Statistical significance testing:
    - McNemar's test (α = 0.01) with Holm–Bonferroni correction

Author: Md Mehedi Hasan
GitHub: https://github.com/mehedi93hasan/TG-FDINet
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Sensitivity, Precision, F1, and AUC-ROC.

    Args:
        y_true:    binary ground-truth labels [N].
        y_score:   predicted probabilities or logits [N].
        threshold: decision threshold for binary classification.

    Returns:
        dict with keys: 'sensitivity', 'precision', 'f1', 'auc'.
    """
    from sklearn.metrics import (
        recall_score, precision_score, f1_score, roc_auc_score
    )

    y_pred = (y_score >= threshold).astype(int)

    sens = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = 0.5   # only one class present

    return {
        'sensitivity': float(sens),
        'precision':   float(prec),
        'f1':          float(f1),
        'auc':         float(auc),
    }


# ---------------------------------------------------------------------------
# McNemar's test with Holm–Bonferroni correction
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true:  np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    continuity: bool = True,
) -> Tuple[float, float]:
    """
    McNemar's test comparing two binary classifiers.

    Constructs the 2×2 contingency table of disagreements:
        b = cases where model1 correct, model2 wrong
        c = cases where model1 wrong,   model2 correct

    Test statistic (with continuity correction):
        χ² = (|b - c| - 1)² / (b + c)

    Args:
        y_true:     ground-truth labels [N].
        y_pred1:    binary predictions from model 1 [N].
        y_pred2:    binary predictions from model 2 [N].
        continuity: apply continuity correction (Edwards, recommended
                    for b+c < 25).

    Returns:
        statistic: χ² test statistic.
        p_value:   two-tailed p-value.
    """
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    b = int(( correct1 & ~correct2).sum())   # model1 right, model2 wrong
    c = int((~correct1 &  correct2).sum())   # model1 wrong, model2 right

    if b + c == 0:
        return 0.0, 1.0

    if continuity:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        stat = (b - c) ** 2 / (b + c)

    p_value = 1.0 - stats.chi2.cdf(stat, df=1)
    return float(stat), float(p_value)


def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Holm–Bonferroni step-down procedure for controlling family-wise
    error rate in multiple hypothesis testing.

    The k-th most significant hypothesis (sorted ascending by p-value)
    is rejected if p_k ≤ α / (m − k + 1), where m is the number of tests.

    Args:
        p_values: list of raw p-values from individual McNemar tests.
        alpha:    family-wise significance level (default 0.01).

    Returns:
        adjusted_p: Bonferroni-Holm adjusted p-values (same order as input).
        rejected:   boolean array indicating which hypotheses are rejected.
    """
    m        = len(p_values)
    order    = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]

    adjusted = np.zeros(m)
    reject   = np.zeros(m, dtype=bool)

    for k, idx in enumerate(order):
        adj = sorted_p[k] * (m - k)
        adjusted[idx] = min(adj, 1.0)

    # Enforce monotonicity
    for k in range(1, m):
        adjusted[order[k]] = max(adjusted[order[k]], adjusted[order[k - 1]])

    reject = adjusted <= alpha

    return adjusted, reject


def pairwise_significance(
    y_true:        np.ndarray,
    model_preds:   Dict[str, np.ndarray],
    reference:     str,
    alpha:         float = 0.01,
    threshold:     float = 0.5,
) -> Dict[str, dict]:
    """
    Perform pairwise McNemar tests between a reference model and all others,
    with Holm–Bonferroni correction.

    Args:
        y_true:      ground-truth labels [N].
        model_preds: dict mapping model name → predicted probabilities [N].
        reference:   name of the reference model (e.g. 'TG-FDINet').
        alpha:       significance level after correction.
        threshold:   probability threshold for binarisation.

    Returns:
        results: dict mapping model name → {
            'statistic', 'raw_p', 'adjusted_p', 'significant', 'delta_sens'
        }
    """
    comparators = [k for k in model_preds if k != reference]
    if not comparators:
        return {}

    ref_bin   = (model_preds[reference] >= threshold).astype(int)
    raw_ps    = []
    stats_all = []

    for name in comparators:
        cmp_bin = (model_preds[name] >= threshold).astype(int)
        stat, p = mcnemar_test(y_true, ref_bin, cmp_bin)
        raw_ps.append(p)
        stats_all.append(stat)

    adj_ps, rejected = holm_bonferroni(raw_ps, alpha=alpha)

    # Sensitivity gains
    ref_sens = float((ref_bin[y_true == 1]).mean()) if y_true.sum() > 0 else 0.0

    results = {}
    for i, name in enumerate(comparators):
        cmp_bin  = (model_preds[name] >= threshold).astype(int)
        cmp_sens = float((cmp_bin[y_true == 1]).mean()) if y_true.sum() > 0 else 0.0
        results[name] = {
            'statistic':   round(stats_all[i], 4),
            'raw_p':       round(raw_ps[i], 6),
            'adjusted_p':  round(float(adj_ps[i]), 6),
            'significant': bool(rejected[i]),
            'delta_sens':  round((ref_sens - cmp_sens) * 100, 2),   # pp
        }

    return results


# ---------------------------------------------------------------------------
# Per-morphology evaluation
# ---------------------------------------------------------------------------

def per_morphology_metrics(
    y_true_dict:  Dict[str, np.ndarray],
    y_score_dict: Dict[str, np.ndarray],
    threshold:    float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for each FDI morphology.

    Args:
        y_true_dict:  dict mapping morphology name → ground-truth labels.
        y_score_dict: dict mapping morphology name → predicted scores.
        threshold:    decision threshold.

    Returns:
        dict mapping morphology name → metric dict.
    """
    return {
        morphology: compute_metrics(
            y_true_dict[morphology],
            y_score_dict[morphology],
            threshold=threshold,
        )
        for morphology in y_true_dict
    }


# ---------------------------------------------------------------------------
# CMFG gate analysis utility
# ---------------------------------------------------------------------------

def summarise_gate_values(
    gate_values:    np.ndarray,   # [N, C, d]
    labels:         np.ndarray,   # [N]
    morphology_ids: np.ndarray,   # [N]  integer morphology code
    morphology_map: dict,         # int → str
) -> Dict[str, float]:
    """
    Compute mean CMFG gate value g̅ per FDI morphology.

    Expected:
        Instant  (Spike)     → g̅ ≈ 0.29  (temporal-dominant)
        Gradual Drift        → g̅ ≈ 0.76  (graph-dominant)

    Args:
        gate_values:    per-sample gate arrays [N, C, d].
        labels:         binary attack labels [N].
        morphology_ids: integer morphology code per sample [N].
        morphology_map: maps integer code → morphology name string.

    Returns:
        dict mapping morphology name → mean g̅ (over attack windows only).
    """
    results = {}
    attack_mask = labels == 1

    for code, name in morphology_map.items():
        morph_mask = (morphology_ids == code) & attack_mask
        if morph_mask.sum() == 0:
            results[name] = float('nan')
        else:
            results[name] = float(gate_values[morph_mask].mean())

    return results
