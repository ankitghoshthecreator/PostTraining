"""tests/test_metrics.py — NLP reference metrics tests."""

from eval.metrics import (
    compute_exact_match,
    compute_edit_distance,
    compute_normalized_edit_similarity,
    compute_rouge_l,
    compute_bleu_1,
)


def test_compute_exact_match():
    assert compute_exact_match("Hello world.", "hello world.", case_sensitive=False) == 1.0
    assert compute_exact_match("Hello world.", "hello world.", case_sensitive=True) == 0.0
    assert compute_exact_match("Different", "Another") == 0.0


def test_compute_edit_distance():
    assert compute_edit_distance("kitten", "sitting") == 3
    assert compute_edit_distance("same", "same") == 0


def test_compute_normalized_edit_similarity():
    sim = compute_normalized_edit_similarity("apple", "apple")
    assert sim == 1.0
    diff_sim = compute_normalized_edit_similarity("abc", "xyz")
    assert diff_sim == 0.0


def test_compute_rouge_l_identical_strings():
    text = "The rapid development of artificial intelligence has transformed scientific discovery."
    score = compute_rouge_l(text, text)
    assert score == 1.0
