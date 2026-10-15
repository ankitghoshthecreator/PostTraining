"""tests/test_gate.py — Severity gate and failure aggregation tests."""

from rlaif.critic import FailureItem
from rlaif.gate import filter_and_aggregate_failures
from rlaif.taxonomy import FailureCategory, Severity


def test_gate_passes_critical_high_only():
    failures = [
        (1, FailureItem(category=FailureCategory.REPETITION, severity=Severity.CRITICAL, explanation="Loop", priority=2), "bad1"),
        (2, FailureItem(category=FailureCategory.OFF_TOPIC, severity=Severity.HIGH, explanation="Off topic", priority=4), "bad2"),
        (3, FailureItem(category=FailureCategory.GRAMMAR_ERROR, severity=Severity.MEDIUM, explanation="Minor typo", priority=3), "bad3"),
        (4, FailureItem(category=FailureCategory.PUNCTUATION_ERROR, severity=Severity.LOW, explanation="Comma missing", priority=5), "bad4"),
    ]
    requests = filter_and_aggregate_failures(failures, budget=10)
    assert len(requests) == 2
    stages_passed = {r.stage for r in requests}
    assert stages_passed == {1, 2}


def test_gate_boundary_priority_5_vs_6():
    failures = [
        (1, FailureItem(category=FailureCategory.REPETITION, severity=Severity.HIGH, explanation="Loop", priority=5), "bad1"),
        (2, FailureItem(category=FailureCategory.OFF_TOPIC, severity=Severity.HIGH, explanation="Off topic", priority=6), "bad2"),
    ]
    requests = filter_and_aggregate_failures(failures, budget=10)
    assert len(requests) == 1
    assert requests[0].priority == 5


def test_gate_respects_request_budget():
    failures = [
        (st, FailureItem(category=FailureCategory.REPETITION, severity=Severity.CRITICAL, explanation=f"Fail {st}", priority=1), f"bad_{st}")
        for st in range(1, 6)
    ]
    requests = filter_and_aggregate_failures(failures, budget=3)
    assert len(requests) == 3


def test_gate_aggregates_exemplars_up_to_3():
    failures = [
        (1, FailureItem(category=FailureCategory.REPETITION, severity=Severity.CRITICAL, explanation=f"Fail {i}", priority=2), f"bad_{i}")
        for i in range(5)
    ]
    requests = filter_and_aggregate_failures(failures, budget=5)
    assert len(requests) == 1
    assert len(requests[0].sample_bad_outputs) <= 3


def test_gate_empty_input_returns_empty():
    assert filter_and_aggregate_failures([], budget=10) == []
