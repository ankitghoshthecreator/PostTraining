"""
trainer/dataset.py — JSONL loading, schema validation, tokenization, and prompt-label masking.

Schema (each line):
{
  "id": str,
  "stage": int (1-5),
  "task": str,
  "prompt": str,
  "target": str,
  "source": "human" | "synthetic",
  "iteration": int,
  "meta": {...}
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Separator injected between prompt and target to help the tokenizer
# respect BPE boundaries cleanly.
PROMPT_SEPARATOR = "\n### Response:\n"

REQUIRED_KEYS = {"id", "stage", "task", "prompt", "target", "source", "iteration"}
VALID_STAGES = {1, 2, 3, 4, 5}


@dataclass
class Sample:
    """A single training example in its raw (pre-tokenized) form."""
    id: str
    stage: int
    task: str
    prompt: str
    target: str
    source: str
    iteration: int
    meta: dict[str, Any]


def _validate_record(record: dict[str, Any], path: Path, line_no: int) -> None:
    """Raise ValueError on schema violations."""
    missing = REQUIRED_KEYS - record.keys()
    if missing:
        raise ValueError(f"{path}:{line_no} — missing keys: {missing}")
    if not isinstance(record["prompt"], str) or not record["prompt"].strip():
        raise ValueError(f"{path}:{line_no} — 'prompt' must be a non-empty string")
    if not isinstance(record["target"], str) or not record["target"].strip():
        raise ValueError(f"{path}:{line_no} — 'target' must be a non-empty string")
    if record["stage"] not in VALID_STAGES:
        raise ValueError(f"{path}:{line_no} — 'stage' must be one of {VALID_STAGES}, got {record['stage']!r}")
    if len(record["target"]) < 10:
        raise ValueError(f"{path}:{line_no} — 'target' is too short (< 10 chars)")
    if len(record["target"]) > 4096:
        raise ValueError(f"{path}:{line_no} — 'target' exceeds 4096 chars")


def load_jsonl(path: str | Path, expected_stage: Optional[int] = None) -> list[Sample]:
    """
    Load and validate a JSONL file.  Returns a list of Sample objects.

    Args:
        path: Path to the .jsonl file.
        expected_stage: If provided, asserts every record matches this stage.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    samples: list[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} — invalid JSON: {exc}") from exc

            _validate_record(record, path, line_no)

            if expected_stage is not None and record["stage"] != expected_stage:
                raise ValueError(
                    f"{path}:{line_no} — stage mismatch: expected {expected_stage}, got {record['stage']}"
                )

            samples.append(
                Sample(
                    id=record["id"],
                    stage=record["stage"],
                    task=record["task"],
                    prompt=record["prompt"],
                    target=record["target"],
                    source=record["source"],
                    iteration=record["iteration"],
                    meta=record.get("meta", {}),
                )
            )

    logger.info("Loaded %d samples from %s", len(samples), path)
    return samples


# ---------------------------------------------------------------------------
# Tokenisation and prompt-label masking
# ---------------------------------------------------------------------------


def build_example(
    prompt: str,
    target: str,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int = 512,
) -> Optional[dict[str, list[int]]]:
    """
    Tokenize a (prompt, target) pair and return input_ids / attention_mask / labels
    with labels[:prompt_len] = -100 (prompt-label masking).

    Returns None if the *target* would be entirely truncated (we never truncate targets).
    """
    prompt_text = prompt + PROMPT_SEPARATOR
    target_text = target + tokenizer.eos_token

    # Tokenize parts separately to know the boundary.
    prompt_ids: list[int] = tokenizer(
        prompt_text, add_special_tokens=False
    )["input_ids"]
    target_ids: list[int] = tokenizer(
        target_text, add_special_tokens=False
    )["input_ids"]

    # Tokenize the full concatenation to respect BPE merges at the boundary.
    full_ids: list[int] = tokenizer(
        prompt_text + target_text, add_special_tokens=False
    )["input_ids"]

    # BPE boundary check — prefix must align.
    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Re-tokenize with a hard separator to force alignment.
        logger.debug("BPE boundary mismatch detected for id; re-aligning via separator.")
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + target_ids  # safe concat after separator

    prompt_len = len(prompt_ids)
    total_len = len(full_ids)

    if total_len > max_len:
        # Truncate the prompt side first; never truncate the target.
        overflow = total_len - max_len
        new_prompt_len = prompt_len - overflow
        if new_prompt_len <= 0:
            # Target alone already exceeds max_len; drop this sample.
            logger.debug("Dropping sample: target alone exceeds max_len=%d", max_len)
            return None
        full_ids = full_ids[overflow:]  # keep tail of prompt + full target
        prompt_len = new_prompt_len

    labels = full_ids.copy()
    labels[:prompt_len] = [-100] * prompt_len  # mask prompt

    # For GPT-2 (pad_token == eos_token), mask any trailing pad positions in labels.
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        for i in range(len(labels)):
            if full_ids[i] == pad_id and i >= prompt_len:
                # Only mask pads that appear after the EOS — i.e. real padding.
                eos_id = tokenizer.eos_token_id
                if eos_id is not None and i > full_ids.index(eos_id) if eos_id in full_ids else False:
                    labels[i] = -100

    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_fn(
    batch: list[dict[str, list[int]]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """
    Pad a batch of examples to the same length.
    - input_ids  → padded with pad_token_id
    - labels     → padded with -100
    - attention_mask → padded with 0
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)

    input_ids_batch, labels_batch, attn_batch = [], [], []
    for ex in batch:
        seq_len = len(ex["input_ids"])
        pad_len = max_len - seq_len

        input_ids_batch.append(ex["input_ids"] + [pad_token_id] * pad_len)
        labels_batch.append(ex["labels"] + [-100] * pad_len)
        attn_batch.append(ex["attention_mask"] + [0] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attn_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class SFTDataset(Dataset):
    """
    A PyTorch Dataset over pre-tokenized examples.  Accepts a list of
    raw dicts (output of build_example) or Sample objects with a tokenizer.
    """

    def __init__(
        self,
        samples: list[Sample],
        tokenizer: PreTrainedTokenizerBase,
        max_len: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._examples: list[dict[str, list[int]]] = []

        dropped = 0
        for s in samples:
            ex = build_example(s.prompt, s.target, tokenizer, max_len)
            if ex is None:
                dropped += 1
            else:
                self._examples.append(ex)

        if dropped:
            logger.warning("Dropped %d/%d samples (target would be truncated)", dropped, len(samples))

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self._examples[idx]

    def get_collate_fn(self):
        """Returns a collate function pre-bound to this tokenizer's pad id."""
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        return lambda batch: collate_fn(batch, pad_id)
