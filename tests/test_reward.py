"""tests/test_reward.py — Reward scalar and regression gate tests."""

from eval.reward import RewardCalculator


def test_compute_stage_reward_mean():
    calc = RewardCalculator()
    scores = [0.8, 0.9, 1.0]
    mean = calc.compute_stage_reward(scores)
    assert abs(mean - 0.9) < 1e-4


def test_compute_macro_reward():
    calc = RewardCalculator()
    stage_rewards = {1: 0.9, 2: 0.8, 3: 1.0}
    macro = calc.compute_macro_reward(stage_rewards)
    assert abs(macro - 0.9) < 1e-4


def test_check_regression_within_tolerance():
    calc = RewardCalculator()
    is_reg, delta = calc.check_regression(baseline_reward=0.90, current_reward=0.88, tolerance=0.05)
    assert is_reg is False
    assert delta == -0.02


def test_check_regression_exceeds_tolerance():
    calc = RewardCalculator()
    is_reg, delta = calc.check_regression(baseline_reward=0.90, current_reward=0.80, tolerance=0.05)
    assert is_reg is True
    assert delta == -0.10
