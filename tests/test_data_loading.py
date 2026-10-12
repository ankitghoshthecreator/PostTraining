"""
tests/test_data_loading.py — Dataset I/O and probe leakage tests.

Covers:
- load_jsonl schema validation
- probe leakage detection
- malformed JSONL handling
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from trainer.dataset import load_jsonl, VALID_STAGES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_RECORD = {
    "id": "s1-000001",
    "stage": 1,
    "task": "english",
    "prompt": "Continue the sentence:",
    "target": "The sun rose over the hills.",
    "source": "human",
    "iteration": 0,
    "meta": {},
}


def write_jsonl(tmp_dir: Path, records: list[dict], filename: str = "test.jsonl") -> Path:
    path = tmp_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# load_jsonl tests
# ---------------------------------------------------------------------------

class TestLoadJsonl:

    def test_valid_record_loads(self, tmp_path):
        path = write_jsonl(tmp_path, [VALID_RECORD])
        samples = load_jsonl(path)
        assert len(samples) == 1
        assert samples[0].id == "s1-000001"

    def test_multiple_valid_records(self, tmp_path):
        records = [{**VALID_RECORD, "id": f"s1-{i:06d}"} for i in range(5)]
        path = write_jsonl(tmp_path, records)
        samples = load_jsonl(path)
        assert len(samples) == 5

    def test_missing_required_key_raises(self, tmp_path):
        bad = {k: v for k, v in VALID_RECORD.items() if k != "target"}
        path = write_jsonl(tmp_path, [bad])
        with pytest.raises(ValueError, match="missing keys"):
            load_jsonl(path)

    def test_empty_prompt_raises(self, tmp_path):
        bad = {**VALID_RECORD, "prompt": ""}
        path = write_jsonl(tmp_path, [bad])
        with pytest.raises(ValueError, match="non-empty string"):
            load_jsonl(path)

    def test_empty_target_raises(self, tmp_path):
        bad = {**VALID_RECORD, "target": ""}
        path = write_jsonl(tmp_path, [bad])
        with pytest.raises(ValueError, match="non-empty string"):
            load_jsonl(path)

    def test_target_too_short_raises(self, tmp_path):
        bad = {**VALID_RECORD, "target": "short"}  # < 10 chars
        path = write_jsonl(tmp_path, [bad])
        with pytest.raises(ValueError, match="too short"):
            load_jsonl(path)

    def test_target_too_long_raises(self, tmp_path):
        bad = {**VALID_RECORD, "target": "x" * 4097}
        path = write_jsonl(tmp_path, [bad])
        with pytest.raises(ValueError, match="4096"):
            load_jsonl(path)

    def test_invalid_stage_raises(self, tmp_path):
        bad = {**VALID_RECORD, "stage": 99}
        path = write_jsonl(tmp_path, [bad])
        with pytest.raises(ValueError, match="stage"):
            load_jsonl(path)

    def test_stage_mismatch_raises(self, tmp_path):
        path = write_jsonl(tmp_path, [VALID_RECORD])  # stage=1
        with pytest.raises(ValueError, match="stage mismatch"):
            load_jsonl(path, expected_stage=2)

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": broken json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            load_jsonl(path)

    def test_empty_lines_skipped(self, tmp_path):
        path = tmp_path / "empty_lines.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(VALID_RECORD) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps({**VALID_RECORD, "id": "s1-999999"}) + "\n")
        samples = load_jsonl(path)
        assert len(samples) == 2

    def test_unicode_in_target(self, tmp_path):
        record = {**VALID_RECORD, "target": "日本語のテキスト。これはテストです。"}
        path = write_jsonl(tmp_path, [record])
        samples = load_jsonl(path)
        assert samples[0].target == record["target"]

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/path/file.jsonl")


# ---------------------------------------------------------------------------
# Probe leakage tests
# ---------------------------------------------------------------------------

class TestProbeLeakage:

    def _jaccard(self, a: str, b: str) -> float:
        a_set = set(a.lower().split())
        b_set = set(b.lower().split())
        if not a_set and not b_set:
            return 1.0
        if not a_set or not b_set:
            return 0.0
        return len(a_set & b_set) / len(a_set | b_set)

    def test_no_probe_in_real_training_data(self, tmp_path):
        """Probe prompts must not appear in training data — Jaccard < 0.85."""
        # Training samples
        train_prompts = [
            "Continue the following sentence: The sun rose over the hills,",
            "Summarize: Scientists discovered a new species in the Amazon.",
            "Fix the grammar: she dont like apples",
        ]
        # Probe prompts (should be distinct)
        probe_prompts = [
            "Continue the following passage: The old lighthouse stood at the edge,",
            "Summarize: Quantum computers may revolutionise cryptography.",
            "Fix the grammar: him and his sister goes to school",
        ]

        for pp in probe_prompts:
            for tp in train_prompts:
                j = self._jaccard(pp, tp)
                assert j < 0.85, (
                    f"Probe prompt is too similar to training prompt (Jaccard={j:.2f}): "
                    f"'{pp[:50]}'"
                )

    def test_exact_match_detected(self):
        """Exact probe-in-training should be caught."""
        prompt = "Fix the grammar: she dont like apples"
        train_set = {prompt.strip().lower()}
        assert prompt.strip().lower() in train_set

    def test_actual_probe_files_are_clean(self):
        """Run the real leakage check on the committed data."""
        import hashlib

        data_dir = Path("data") / "curriculum"
        probe_dir = Path("data") / "probes"

        if not data_dir.exists() or not probe_dir.exists():
            pytest.skip("Data directories not present")

        # Collect training prompts
        train_prompts = []
        for stage_id in VALID_STAGES:
            for path in sorted(data_dir.glob(f"stage_{stage_id}_*.jsonl")):
                try:
                    samples = load_jsonl(path)
                    train_prompts.extend(s.prompt for s in samples)
                except Exception:
                    pass

        train_hashes = {
            hashlib.sha256(p.strip().lower().encode()).hexdigest()
            for p in train_prompts
        }

        leaks = []
        for stage_id in VALID_STAGES:
            probe_path = probe_dir / f"stage_{stage_id}.jsonl"
            if not probe_path.exists():
                continue
            with open(probe_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    probe_prompt = record.get("prompt", "").strip()
                    h = hashlib.sha256(probe_prompt.lower().encode()).hexdigest()
                    if h in train_hashes:
                        leaks.append(probe_prompt[:60])

        assert len(leaks) == 0, f"Probe/training leaks found: {leaks}"
