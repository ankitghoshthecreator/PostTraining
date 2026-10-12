"""tests/test_sanitize.py — Unit tests for <think> sanitization (Task 2.4)."""

import pytest
from rlaif.sanitize import sanitize_think_tags, extract_json_payload


def test_standard_think_block_removed():
    text = "<think>Let me analyze this.</think>The final answer is 42."
    assert sanitize_think_tags(text) == "The final answer is 42."


def test_multiline_and_case_insensitive():
    text = """<THINK>
    Line 1
    Line 2
    </THINK>Valid content."""
    assert sanitize_think_tags(text) == "Valid content."


def test_unclosed_think_block():
    text = "Valid start.<think>I was cut off midway..."
    assert sanitize_think_tags(text) == "Valid start."


def test_multiple_think_blocks():
    text = "<think>first</think>Mid part.<think>second</think>End part."
    assert sanitize_think_tags(text) == "Mid part.End part."


def test_no_false_positive_repetition_in_think():
    # If the critic thinks about repetition inside <think>, it should be stripped
    raw = '<think>I suspect repetition error here.</think>{"failures": []}'
    cleaned = extract_json_payload(raw)
    assert "repetition" not in cleaned
    assert cleaned == '{"failures": []}'


def test_extract_json_from_code_fence():
    raw = """<think>reasoning...</think>
```json
{
  "status": "ok",
  "score": 0.95
}
```
"""
    extracted = extract_json_payload(raw)
    assert '"status": "ok"' in extracted
    assert "<think>" not in extracted
