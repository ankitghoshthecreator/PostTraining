"""tests/test_dpo.py — Direct Preference Optimization (DPO) unit tests."""

import torch
from rlaif.preference_dataset import generate_contrastive_rejected
from trainer.train_dpo import dpo_loss, compute_logps_for_tokens


def test_dpo_loss_formula():
    pi_chosen = torch.tensor([-1.0])
    pi_rejected = torch.tensor([-2.5])
    ref_chosen = torch.tensor([-1.2])
    ref_rejected = torch.tensor([-2.2])

    loss, c_rew, r_rew = dpo_loss(
        policy_chosen_logps=pi_chosen,
        policy_rejected_logps=pi_rejected,
        reference_chosen_logps=ref_chosen,
        reference_rejected_logps=ref_rejected,
        beta=0.1,
    )
    assert loss.item() > 0.0
    assert c_rew.item() > r_rew.item()


def test_dpo_reward_margin_positive():
    pi_chosen = torch.tensor([0.0])
    pi_rejected = torch.tensor([-3.0])
    ref_chosen = torch.tensor([0.0])
    ref_rejected = torch.tensor([0.0])

    loss, c_rew, r_rew = dpo_loss(
        policy_chosen_logps=pi_chosen,
        policy_rejected_logps=pi_rejected,
        reference_chosen_logps=ref_chosen,
        reference_rejected_logps=ref_rejected,
        beta=0.1,
    )
    # Margin should be strictly positive when policy prefers chosen over rejected
    assert (c_rew - r_rew).item() > 0.0


def test_compute_logps_for_tokens():
    logits = torch.randn(2, 5, 10)  # batch=2, seq=5, vocab=10
    labels = torch.tensor([
        [-100, -100, 2, 4, 1],
        [-100, 1, 3, -100, -100],
    ])
    logps = compute_logps_for_tokens(logits, labels, ignore_index=-100)
    assert logps.shape == (2,)
    assert not torch.isnan(logps).any()


def test_generate_contrastive_rejected_all_stages():
    for stage in [1, 2, 3, 4, 5]:
        p = "Sample test prompt"
        c = "This is a clean, well-formed sentence."
        r = generate_contrastive_rejected(p, c, stage=stage)
        assert len(r) > 0
        assert r != c
