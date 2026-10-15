"""eval/agreement.py — Task 3.4: Judge–Human Agreement Study & Cohen's κ.

Provides:
- Cohen's κ calculation with quadratic weights for ordinal scoring (1–5 scale).
- Bootstrap 95% Confidence Interval (CI) estimation.
- Pearson & Spearman correlation between local evaluator signals and teacher judge.
- Quantitative comparison framework between Rubric A (0.66) and Rubric B (0.77).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None  # fallback implementation below

logger = logging.getLogger("eval.agreement")


def manual_cohen_kappa(
    rater1: List[int],
    rater2: List[int],
    weights: Optional[str] = "quadratic",
) -> float:
    """Pure-Python Cohen's kappa implementation as resilient fallback."""
    if len(rater1) != len(rater2) or len(rater1) == 0:
        return 0.0

    categories = sorted(list(set(rater1) | set(rater2)))
    k = len(categories)
    if k <= 1:
        return 1.0

    cat_to_idx = {c: i for i, c in enumerate(categories)}
    conf_matrix = np.zeros((k, k), dtype=np.float64)

    for r1, r2 in zip(rater1, rater2):
        conf_matrix[cat_to_idx[r1], cat_to_idx[r2]] += 1.0

    n = np.sum(conf_matrix)
    if n == 0:
        return 0.0

    p_observed = np.trace(conf_matrix) / n
    row_sums = np.sum(conf_matrix, axis=1)
    col_sums = np.sum(conf_matrix, axis=0)
    p_expected = np.sum(row_sums * col_sums) / (n * n)

    if weights == "quadratic":
        weight_mat = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                weight_mat[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

        expected_mat = np.outer(row_sums, col_sums) / n
        observed_weighted = np.sum(weight_mat * conf_matrix) / n
        expected_weighted = np.sum(weight_mat * expected_mat) / n

        if expected_weighted == 0:
            return 1.0
        return float(1.0 - (observed_weighted / expected_weighted))

    if p_expected == 1.0:
        return 1.0
    return float((p_observed - p_expected) / (1.0 - p_expected))


def compute_cohen_kappa(
    rater_a: List[int],
    rater_b: List[int],
    weights: str = "quadratic",
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calculate Cohen's κ with bootstrap 95% confidence interval."""
    if len(rater_a) != len(rater_b):
        raise ValueError(f"Rater lengths must match: {len(rater_a)} vs {len(rater_b)}")

    n = len(rater_a)
    if n == 0:
        return {"kappa": 0.0, "ci_low": 0.0, "ci_high": 0.0, "sample_size": 0}

    # Core score calculation
    if cohen_kappa_score is not None:
        try:
            point_kappa = float(cohen_kappa_score(rater_a, rater_b, weights=weights))
        except Exception:
            point_kappa = manual_cohen_kappa(rater_a, rater_b, weights=weights)
    else:
        point_kappa = manual_cohen_kappa(rater_a, rater_b, weights=weights)

    # Bootstrap 95% Confidence Interval
    rng = np.random.RandomState(seed)
    bootstrap_kappas = []
    a_arr = np.array(rater_a)
    b_arr = np.array(rater_b)

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        sample_a = a_arr[indices].tolist()
        sample_b = b_arr[indices].tolist()
        if cohen_kappa_score is not None:
            try:
                k_val = cohen_kappa_score(sample_a, sample_b, weights=weights)
            except Exception:
                k_val = manual_cohen_kappa(sample_a, sample_b, weights=weights)
        else:
            k_val = manual_cohen_kappa(sample_a, sample_b, weights=weights)
        if not np.isnan(k_val):
            bootstrap_kappas.append(k_val)

    if bootstrap_kappas:
        ci_low = float(np.percentile(bootstrap_kappas, 2.5))
        ci_high = float(np.percentile(bootstrap_kappas, 97.5))
    else:
        ci_low, ci_high = point_kappa, point_kappa

    return {
        "kappa": round(point_kappa, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "sample_size": n,
        "weights": weights,
    }


def compute_correlation(
    x_scores: List[float],
    y_scores: List[float],
) -> Dict[str, float]:
    """Calculate Pearson and Spearman rank correlation between two scorers."""
    if len(x_scores) != len(y_scores) or len(x_scores) < 2:
        return {"pearson": 0.0, "spearman": 0.0}

    x = np.array(x_scores, dtype=np.float64)
    y = np.array(y_scores, dtype=np.float64)

    # Pearson
    x_diff = x - np.mean(x)
    y_diff = y - np.mean(y)
    denom = np.sqrt(np.sum(x_diff ** 2) * np.sum(y_diff ** 2))
    pearson = float(np.sum(x_diff * y_diff) / denom) if denom > 0 else 0.0

    # Spearman via rank
    x_rank = np.argsort(np.argsort(x)).astype(np.float64)
    y_rank = np.argsort(np.argsort(y)).astype(np.float64)
    xr_diff = x_rank - np.mean(x_rank)
    yr_diff = y_rank - np.mean(y_rank)
    r_denom = np.sqrt(np.sum(xr_diff ** 2) * np.sum(yr_diff ** 2))
    spearman = float(np.sum(xr_diff * yr_diff) / r_denom) if r_denom > 0 else 0.0

    return {
        "pearson": round(pearson, 4),
        "spearman": round(spearman, 4),
    }
