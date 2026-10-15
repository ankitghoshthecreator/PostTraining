"""tests/test_router.py — Tiered evaluation router tests."""

from eval.router import TieredRouter


def test_router_fast_pass_above_threshold():
    router = TieredRouter(local_pass_threshold=0.70)
    # High-quality response passes locally
    prompt = "Summarize the text."
    response = "The telescope detected signals from an ancient galaxy dating back to the dawn of the universe."
    ref = "The telescope observed signals from an ancient cosmic galaxy."
    routed = router.route(prompt, response, stage=2, reference=ref)
    assert routed["routed_to"] == "local"
    assert routed["escalated"] is False
    assert routed["passed"] is True


def test_router_escalates_below_threshold():
    router = TieredRouter(local_pass_threshold=0.85)
    # Average response below 0.85 escalates to teacher judge
    prompt = "Punctuate sentence"
    response = "no punctuation here at all"
    routed = router.route(prompt, response, stage=4)
    assert routed["routed_to"] == "teacher"
    assert routed["escalated"] is True


def test_router_exact_threshold_boundary():
    router = TieredRouter(local_pass_threshold=0.70)
    assert router.threshold == 0.70


def test_router_telemetry_tracking():
    router = TieredRouter(local_pass_threshold=0.70)
    router.reset_telemetry()
    router.route("P1", "Clean well-formed response sentence.", stage=1)
    router.route("P2", "bad", stage=1)
    stats = router.telemetry
    assert stats["total_routed"] == 2
    assert stats["local_passes"] + stats["teacher_escalations"] == 2


def test_router_threshold_sweep():
    router = TieredRouter()
    pairs = [
        {"prompt": "P1", "response": "Clean complete response sentence.", "reference": "Clean response sentence.", "stage": 1, "ground_truth_pass": True},
        {"prompt": "P2", "response": "bad", "reference": "Good response.", "stage": 1, "ground_truth_pass": False},
    ]
    sweep = router.sweep_thresholds(pairs, thresholds=[0.60, 0.70, 0.80])
    assert len(sweep) == 3
    for entry in sweep:
        assert 0.0 <= entry["cost_saved_pct"] <= 100.0
        assert 0.0 <= entry["false_pass_rate"] <= 1.0
