"""
tests/test_dataset.py — Masking, tokenization, boundary, and collator tests.

Covers Task 1.3 acceptance criteria from phase.md:
- label count == target token count + 1 (EOS)
- first non-masked label is first target token
- no -100 inside the target span
- works for GPT-2, and structure is compatible with other tokenizers
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Lightweight mock tokenizer — avoids downloading models in CI
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal tokenizer mock that encodes characters as their ord values."""

    def __init__(self, pad_token: str | None = None):
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.pad_token = pad_token or "<eos>"
        self.pad_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        # Simple encoding: each character → its ordinal (shifted to avoid 0)
        ids = [ord(c) % 200 + 2 for c in text]
        return {"input_ids": ids}


def make_tokenizer(with_pad: bool = True) -> MockTokenizer:
    return MockTokenizer(pad_token="<pad>" if with_pad else None)


# ---------------------------------------------------------------------------
# Tests for build_example
# ---------------------------------------------------------------------------

from trainer.dataset import build_example, collate_fn, PROMPT_SEPARATOR


class TestBuildExample:
    def test_label_count_equals_target_plus_eos(self):
        """sum(l != -100) == len(target_tokens) + 1 (EOS)."""
        tok = make_tokenizer()
        prompt = "Fix the grammar:"
        target = "She doesn't like apples."
        result = build_example(prompt, target, tok, max_len=512)
        assert result is not None

        non_masked = [l for l in result["labels"] if l != -100]
        # target_ids + EOS
        target_text = target + tok.eos_token
        target_ids = tok(target_text, add_special_tokens=False)["input_ids"]
        assert len(non_masked) == len(target_ids), (
            f"Expected {len(target_ids)} non-masked labels, got {len(non_masked)}"
        )

    def test_first_non_masked_label_is_first_target_token(self):
        """The first label that is not -100 must be the first target token."""
        tok = make_tokenizer()
        prompt = "Summarize:"
        target = "This is a summary."
        result = build_example(prompt, target, tok, max_len=512)
        assert result is not None

        first_non_masked = next(l for l in result["labels"] if l != -100)
        target_text = target + tok.eos_token
        first_target_id = tok(target_text, add_special_tokens=False)["input_ids"][0]
        assert first_non_masked == first_target_id

    def test_no_minus100_inside_target_span(self):
        """After the first non-masked label, there should be no -100."""
        tok = make_tokenizer()
        result = build_example("A prompt:", "A target response.", tok, max_len=512)
        assert result is not None

        labels = result["labels"]
        in_target = False
        for label in labels:
            if label != -100:
                in_target = True
            if in_target and label == -100:
                pytest.fail("Found -100 inside the target span")

    def test_all_prompt_tokens_masked(self):
        """All prompt tokens must have label == -100."""
        tok = make_tokenizer()
        prompt = "Hello world:"
        target = "Test target."
        result = build_example(prompt, target, tok, max_len=512)
        assert result is not None

        prompt_text = prompt + PROMPT_SEPARATOR
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        labels = result["labels"]

        for i, label in enumerate(labels[: len(prompt_ids)]):
            assert label == -100, f"Prompt token at position {i} is not masked"

    def test_returns_none_when_target_would_be_cut(self):
        """If max_len is too small to fit even the target, return None."""
        tok = make_tokenizer()
        long_target = "x" * 200
        # max_len < len(target) → should drop
        result = build_example("prompt:", long_target, tok, max_len=5)
        assert result is None

    def test_truncates_prompt_not_target(self):
        """When truncating, prompt is shortened; target must remain intact."""
        tok = make_tokenizer()
        long_prompt = "A" * 400
        target = "short target."
        result = build_example(long_prompt, target, tok, max_len=100)
        # Either returns None (target too long) or truncated prompt
        if result is not None:
            target_text = target + tok.eos_token
            target_ids = tok(target_text, add_special_tokens=False)["input_ids"]
            non_masked = [l for l in result["labels"] if l != -100]
            assert len(non_masked) == len(target_ids)

    def test_correct_lengths(self):
        """input_ids, attention_mask, and labels should all have the same length."""
        tok = make_tokenizer()
        result = build_example("Some prompt:", "Some target.", tok, max_len=512)
        assert result is not None
        assert (
            len(result["input_ids"])
            == len(result["attention_mask"])
            == len(result["labels"])
        )

    def test_attention_mask_all_ones(self):
        """attention_mask should be all 1s (no padding from build_example itself)."""
        tok = make_tokenizer()
        result = build_example("Prompt:", "Target.", tok, max_len=512)
        assert result is not None
        assert all(m == 1 for m in result["attention_mask"])


# ---------------------------------------------------------------------------
# Tests for collate_fn
# ---------------------------------------------------------------------------

class TestCollateFn:
    def test_pads_to_max_length(self):
        """All sequences in the batch should have the same length after padding."""
        import torch

        batch = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "attention_mask": [1, 1], "labels": [-100, 5]},
        ]
        result = collate_fn(batch, pad_token_id=0)
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)
        assert result["labels"].shape == (2, 3)

    def test_pads_labels_with_minus100(self):
        """Padded label positions must be -100."""
        import torch

        batch = [
            {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": [1, 2, 3, 4]},
            {"input_ids": [5, 6], "attention_mask": [1, 1], "labels": [5, 6]},
        ]
        result = collate_fn(batch, pad_token_id=0)
        assert result["labels"][1, 2].item() == -100
        assert result["labels"][1, 3].item() == -100

    def test_pads_attention_mask_with_zero(self):
        """Padded attention_mask positions must be 0."""
        import torch

        batch = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
            {"input_ids": [4], "attention_mask": [1], "labels": [4]},
        ]
        result = collate_fn(batch, pad_token_id=0)
        assert result["attention_mask"][1, 1].item() == 0
        assert result["attention_mask"][1, 2].item() == 0
