"""tests/test_local_evaluator.py — Ultra-fast local evaluator unit tests."""

from eval.local_evaluator import LocalEvaluator


def test_local_evaluator_range_zero_to_one():
    evaluator = LocalEvaluator()
    res = evaluator.evaluate(
        prompt="Explain photosynthesis in simple terms.",
        response="Photosynthesis is the process by which green plants convert sunlight and carbon dioxide into chemical energy.",
        stage=1,
    )
    assert 0.0 <= res["score"] <= 1.0


def test_local_evaluator_determinism():
    evaluator = LocalEvaluator()
    p = "Write a grammatically correct sentence."
    r = "The curious cat gracefully leaped across the kitchen counter."
    res1 = evaluator.evaluate(p, r, stage=3)
    res2 = evaluator.evaluate(p, r, stage=3)
    assert res1["score"] == res2["score"]
    assert res1["ttr"] == res2["ttr"]


def test_local_evaluator_empty_response():
    evaluator = LocalEvaluator()
    res = evaluator.evaluate("Any prompt", "", stage=1)
    assert res["score"] == 0.0
    assert res["passed"] is False


def test_local_evaluator_severe_repetition():
    evaluator = LocalEvaluator()
    rep_response = "word word word word word word word word word word word"
    res = evaluator.evaluate("Prompt", rep_response, stage=1)
    assert res["ngram_rep"] > 0.8
    assert res["passed"] is False


def test_local_evaluator_unicode_and_special_chars():
    evaluator = LocalEvaluator()
    res = evaluator.evaluate(
        prompt="Translating accents: café naïve résumé",
        response="Café, naïve, and résumé are words with preserved French diacritics.",
        stage=1,
    )
    assert res["score"] > 0.6


def test_local_evaluator_stage4_punctuation_check():
    evaluator = LocalEvaluator()
    # Good punctuation
    res_good = evaluator.evaluate("Punctuate text", "Although it rained, we enjoyed our picnic.", stage=4)
    # Bad punctuation
    res_bad = evaluator.evaluate("Punctuate text", "although it rained we enjoyed our picnic", stage=4)
    assert res_good["rules"] > res_bad["rules"]
    assert res_good["score"] > res_bad["score"]
