"""validator/layers.py — Task 2.8: 7-layer validation checks with reason codes."""

from __future__ import annotations

import hashlib
import re
from typing import Optional, Set


def normalize_text(text: str) -> str:
    """Normalize text for invariant comparisons and dedup hashing."""
    return re.sub(r"\s+", " ", text.strip().lower())


def check_l1_schema(record: dict) -> Optional[str]:
    """L1: Verify required schema keys exist and are non-empty."""
    required = ("prompt", "target")
    for k in required:
        if k not in record or not isinstance(record[k], str) or not record[k].strip():
            return f"L1_SCHEMA_INVALID: missing or empty key '{k}'"
    return None


def check_l2_min_length(record: dict, min_len: int = 10) -> Optional[str]:
    """L2: Target must have at least min_len characters."""
    if len(record["target"].strip()) < min_len:
        return f"L2_TOO_SHORT: length {len(record['target'].strip())} < {min_len}"
    return None


def check_l3_max_length(record: dict, max_len: int = 4096) -> Optional[str]:
    """L3: Combined prompt + target length must not exceed max_len."""
    total = len(record["prompt"]) + len(record["target"])
    if total > max_len:
        return f"L3_TOO_LONG: length {total} > {max_len}"
    return None


def check_l4_repetition(record: dict, max_bigram_ratio: float = 0.35) -> Optional[str]:
    """L4: Reject if bigram repetition ratio >= max_bigram_ratio."""
    words = record["target"].lower().split()
    if len(words) < 6:
        return None  # too short for meaningful bigram degradation check
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    unique_bigrams = set(bigrams)
    rep_ratio = 1.0 - (len(unique_bigrams) / len(bigrams))
    if rep_ratio >= max_bigram_ratio:
        return f"L4_REPETITION_HIGH: ratio {rep_ratio:.3f} >= {max_bigram_ratio}"
    return None


def check_l5_identity(record: dict) -> Optional[str]:
    """L5: Reject if target is identical to prompt (echo/identity violation)."""
    if normalize_text(record["prompt"]) == normalize_text(record["target"]):
        return "L5_IDENTITY_VIOLATION: target echoes prompt verbatim"
    return None


def check_l6_circularity(record: dict, max_jaccard: float = 0.85) -> Optional[str]:
    """L6: Reject if target is essentially circular/identical to prompt via Jaccard."""
    p_words = set(normalize_text(record["prompt"]).split())
    t_words = set(normalize_text(record["target"]).split())
    if not p_words or not t_words:
        return None
    jaccard = len(p_words & t_words) / len(p_words | t_words)
    if jaccard >= max_jaccard:
        return f"L6_CIRCULARITY_HIGH: Jaccard similarity {jaccard:.3f} >= {max_jaccard}"
    return None


def check_l7_dedup(
    record: dict, seen_hashes: Set[str]
) -> Optional[str]:
    """L7: Deduplication check against seen normalized prompt+target hashes."""
    combined = normalize_text(record["prompt"]) + "|||" + normalize_text(record["target"])
    digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    if digest in seen_hashes:
        return "L7_DUPLICATE_SAMPLE: identical sample previously seen"
    seen_hashes.add(digest)
    return None
