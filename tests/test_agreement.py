"""tests/test_agreement.py — Cohen's kappa and correlation tests."""

from eval.agreement import compute_cohen_kappa, compute_correlation, manual_cohen_kappa


def test_cohen_kappa_perfect_agreement():
    r1 = [1, 2, 3, 4, 5]
    r2 = [1, 2, 3, 4, 5]
    res = compute_cohen_kappa(r1, r2, weights="quadratic")
    assert res["kappa"] == 1.0


def test_cohen_kappa_chance_agreement():
    r1 = [1, 1, 5, 5, 1, 5]
    r2 = [5, 5, 1, 1, 5, 1]
    res = compute_cohen_kappa(r1, r2, weights="quadratic")
    assert res["kappa"] < 0.2


def test_cohen_kappa_bootstrap_ci_interval():
    r1 = [1, 2, 3, 4, 5, 3, 4, 2, 1, 5]
    r2 = [1, 2, 3, 4, 4, 3, 4, 2, 1, 5]
    res = compute_cohen_kappa(r1, r2, n_bootstrap=200)
    assert res["ci_low"] <= res["kappa"] <= res["ci_high"]
    assert res["sample_size"] == 10


def test_compute_correlation_perfect_linear():
    x = [0.1, 0.3, 0.5, 0.7, 0.9]
    y = [0.2, 0.4, 0.6, 0.8, 1.0]
    corr = compute_correlation(x, y)
    assert abs(corr["pearson"] - 1.0) < 1e-4
    assert abs(corr["spearman"] - 1.0) < 1e-4
