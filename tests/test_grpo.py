"""tests/test_grpo.py — Group Relative Policy Optimization (GRPO) unit tests."""

import torch
from rlaif.verifiable_rewards import verify_math_reasoning, verify_json_format, verify_constraint_compliance
from trainer.grpo import compute_group_advantages, grpo_loss


def test_compute_group_advantages_zero_mean():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    adv = compute_group_advantages(rewards)
    assert adv.shape == (4,)
    # Advantages within the group must sum to approximately 0
    assert abs(adv.mean().item()) < 1e-5


def test_compute_group_advantages_identical_rewards_zero():
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
    adv = compute_group_advantages(rewards)
    # When all rollouts have identical reward, all advantages are zero
    assert torch.all(adv == 0.0)


def test_grpo_loss_computation():
    logps_pi = torch.tensor([-0.5, -0.6, -0.7, -0.8])
    logps_old = torch.tensor([-0.5, -0.5, -0.5, -0.5])
    advantages = torch.tensor([1.0, 0.5, -0.5, -1.0])

    tot_loss, surr, kl = grpo_loss(logps_pi, logps_old, advantages, clip_eps=0.2)
    assert not torch.isnan(tot_loss)
    assert not torch.isnan(surr)


def test_verify_math_reasoning_exact():
    prompt = "What is 25 * 4?"
    good_resp = "Let's calculate: 25 * 4 = 100. \\boxed{100}"
    bad_resp = "The answer is 99."

    res_good = verify_math_reasoning(prompt, good_resp, ground_truth=100.0)
    assert res_good["passed"] is True
    assert res_good["reward"] == 1.0

    res_bad = verify_math_reasoning(prompt, bad_resp, ground_truth=100.0)
    assert res_bad["passed"] is False
    assert res_bad["reward"] == 0.0


def test_verify_json_and_constraints():
    json_good = '```json\n{"answer": 42, "reasoning": "step by step"}\n```'
    assert verify_json_format(json_good, required_keys=["answer"])["passed"] is True
    assert verify_json_format("invalid json text")["passed"] is False

    text_good = "This sentence satisfies all constraints."
    assert verify_constraint_compliance(text_good, min_words=3, max_words=10, must_end_with=(".",))["passed"] is True
    assert verify_constraint_compliance("no punctuation", must_end_with=(".",))["passed"] is False
