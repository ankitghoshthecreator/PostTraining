"""tests/test_validator.py — Unit tests for the 7-layer validation pipeline (Task 2.8)."""

import concurrent.futures
import pytest
from validator.pipeline import QualityGatePipeline


def test_valid_sample_passes():
    gate = QualityGatePipeline()
    sample = {
        "prompt": "Summarize this article about solar panels.",
        "target": "Solar panels convert sunlight into electrical power cleanly and efficiently.",
    }
    ok, reason = gate.validate_sample(sample)
    assert ok is True
    assert reason is None


def test_l1_schema_failure():
    gate = QualityGatePipeline()
    ok, reason = gate.validate_sample({"prompt": "Hello"})
    assert ok is False
    assert "L1_SCHEMA_INVALID" in reason


def test_l2_too_short():
    gate = QualityGatePipeline()
    ok, reason = gate.validate_sample({"prompt": "Valid prompt here", "target": "short"})
    assert ok is False
    assert "L2_TOO_SHORT" in reason


def test_l4_repetition_failure():
    gate = QualityGatePipeline()
    repetitive_text = "the dog ran and the dog ran and the dog ran and the dog ran"
    ok, reason = gate.validate_sample({"prompt": "Write a story", "target": repetitive_text})
    assert ok is False
    assert "L4_REPETITION_HIGH" in reason


def test_l5_identity_violation():
    gate = QualityGatePipeline()
    ok, reason = gate.validate_sample({"prompt": "Hello world today", "target": "Hello world today"})
    assert ok is False
    assert "L5_IDENTITY_VIOLATION" in reason


def test_l7_duplicate_detection():
    gate = QualityGatePipeline()
    sample = {
        "prompt": "What is gravity?",
        "target": "Gravity is the natural force of attraction between massive objects.",
    }
    ok1, _ = gate.validate_sample(sample)
    ok2, reason2 = gate.validate_sample(sample)
    assert ok1 is True
    assert ok2 is False
    assert "L7_DUPLICATE_SAMPLE" in reason2


def test_concurrent_dedup():
    gate = QualityGatePipeline()
    sample = {
        "prompt": "Explain photosynthesis.",
        "target": "Photosynthesis is the process plants use to turn sunlight into glucose.",
    }

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(gate.validate_sample, sample) for _ in range(20)]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    passed_count = sum(1 for ok, _ in results if ok)
    assert passed_count == 1
    assert gate.stats["L7_duplicate"] == 19
