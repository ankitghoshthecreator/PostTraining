"""
scripts/validate_data.py — Validate all curriculum JSONL files and confirm
                            zero leakage between probe banks and training data.

Usage:
    python scripts/validate_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer.dataset import load_jsonl, VALID_STAGES


DATA_DIR = Path("data") / "curriculum"
PROBE_DIR = Path("data") / "probes"


def _jaccard(a: str, b: str) -> float:
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def validate_curriculum() -> int:
    """Returns error count."""
    errors = 0
    for stage_id in VALID_STAGES:
        pattern = f"stage_{stage_id}_*.jsonl"
        files = sorted(DATA_DIR.glob(pattern))
        if not files:
            print(f"[WARN]  No file matching {pattern} in {DATA_DIR}")
            continue
        for path in files:
            try:
                samples = load_jsonl(path, expected_stage=stage_id)
                print(f"[OK]    {path.name}: {len(samples)} samples, all valid")
            except (ValueError, FileNotFoundError) as exc:
                print(f"[ERROR] {path.name}: {exc}")
                errors += 1
    return errors


def validate_probes() -> int:
    """Returns error count."""
    errors = 0
    for stage_id in VALID_STAGES:
        probe_path = PROBE_DIR / f"stage_{stage_id}.jsonl"
        if not probe_path.exists():
            print(f"[WARN]  Missing probe file: {probe_path}")
            continue
        count = 0
        try:
            with open(probe_path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if "prompt" not in record or not record["prompt"].strip():
                        print(f"[ERROR] {probe_path.name}:{line_no} — missing or empty 'prompt'")
                        errors += 1
                    count += 1
            print(f"[OK]    {probe_path.name}: {count} probes, all valid")
        except Exception as exc:
            print(f"[ERROR] {probe_path.name}: {exc}")
            errors += 1
    return errors


def leakage_test(jaccard_threshold: float = 0.85) -> int:
    """
    Confirm zero probe prompts appear in training data (exact hash or near-duplicate).
    Returns number of leaks found.
    """
    # Build set of training prompt hashes
    train_prompts: list[str] = []
    for stage_id in VALID_STAGES:
        for path in sorted(DATA_DIR.glob(f"stage_{stage_id}_*.jsonl")):
            try:
                samples = load_jsonl(path)
                train_prompts.extend(s.prompt for s in samples)
            except Exception:
                pass

    train_hashes = {
        hashlib.sha256(p.strip().lower().encode()).hexdigest() for p in train_prompts
    }

    leaks = 0
    for stage_id in VALID_STAGES:
        probe_path = PROBE_DIR / f"stage_{stage_id}.jsonl"
        if not probe_path.exists():
            continue
        with open(probe_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                probe_prompt = record.get("prompt", "").strip()

                # Exact hash check
                h = hashlib.sha256(probe_prompt.lower().encode()).hexdigest()
                if h in train_hashes:
                    print(f"[LEAK]  Exact match: probe {probe_path.name}:{line_no} in training data")
                    leaks += 1
                    continue

                # Jaccard near-duplicate check
                for tp in train_prompts:
                    if _jaccard(probe_prompt, tp) >= jaccard_threshold:
                        print(
                            f"[LEAK]  Near-duplicate (Jaccard≥{jaccard_threshold}): "
                            f"probe {probe_path.name}:{line_no}"
                        )
                        leaks += 1
                        break

    if leaks == 0:
        print(f"[OK]    Leakage test: 0 probe/training overlaps found")
    return leaks


def main() -> None:
    print("\n=== Curriculum Data Validation ===")
    errors = validate_curriculum()

    print("\n=== Probe Bank Validation ===")
    errors += validate_probes()

    print("\n=== Probe Leakage Test ===")
    leaks = leakage_test()
    errors += leaks

    print(f"\nResult: {errors} error(s) found.")
    sys.exit(1 if errors > 0 else 0)


if __name__ == "__main__":
    main()
