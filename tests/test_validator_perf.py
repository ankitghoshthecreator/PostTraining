"""tests/test_validator_perf.py — Validator performance, L3, L6, and short-circuit tests."""

import time
from validator.pipeline import QualityGatePipeline


def test_validator_500_samples_throughput(tmp_path):
    pipeline = QualityGatePipeline(reject_log_path=tmp_path / "rejects.jsonl")
    samples = [
        {
            "id": f"perf-{i}",
            "stage": 1,
            "task": "english",
            "prompt": f"Write sentence number {i} regarding biology.",
            "target": f"Biological organisms continuously adapt to ambient temperature changes in habitat {i}.",
            "source": "synthetic",
        }
        for i in range(500)
    ]
    start = time.perf_counter()
    accepted = pipeline.filter_batch(samples)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert len(accepted) == 500
    # CI generous margin: < 250 ms (target < 9 ms locally)
    assert elapsed_ms < 250.0


def test_validator_l3_too_long(tmp_path):
    pipeline = QualityGatePipeline(reject_log_path=tmp_path / "rejects.jsonl")
    huge_target = "A" * 4097
    samples = [
        {
            "id": "huge-1",
            "stage": 1,
            "task": "english",
            "prompt": "Write a short summary.",
            "target": huge_target,
            "source": "synthetic",
        }
    ]
    accepted = pipeline.filter_batch(samples)
    assert len(accepted) == 0
    assert pipeline.stats.get("L3_max_length", 0) == 1


def test_validator_l6_circularity(tmp_path):
    pipeline = QualityGatePipeline(reject_log_path=tmp_path / "rejects.jsonl")
    samples = [
        {
            "id": "circ-1",
            "stage": 1,
            "task": "english",
            "prompt": "The solar system contains eight planets revolving around the central sun.",
            "target": "The solar system contains eight planets revolving around the central sun.",
            "source": "synthetic",
        }
    ]
    accepted = pipeline.filter_batch(samples)
    assert len(accepted) == 0


def test_validator_order_cheapest_first(tmp_path):
    pipeline = QualityGatePipeline(reject_log_path=tmp_path / "rejects.jsonl")
    samples = [
        {"id": "bad-schema"},  # missing required fields
    ]
    accepted = pipeline.filter_batch(samples)
    assert len(accepted) == 0
    assert pipeline.stats.get("L1_schema", 0) == 1
