"""tests/test_critic.py — Critic parsing, JSON extraction, and fallback tests."""

import json
from rlaif.critic import TeacherCritic, CritiqueResult
from rlaif.sanitize import extract_json_payload


def test_critic_parses_fenced_json():
    raw = "```json\n{\"score\": 0.85, \"passed\": true, \"failures\": []}\n```"
    extracted = extract_json_payload(raw)
    parsed = json.loads(extracted)
    assert parsed["score"] == 0.85
    assert parsed["passed"] is True


def test_critic_extracts_json_with_surrounding_prose():
    raw = "Here is my evaluation:\n{\"score\": 0.4, \"passed\": false, \"failures\": []}\nHope that helps!"
    extracted = extract_json_payload(raw)
    parsed = json.loads(extracted)
    assert parsed["score"] == 0.4
    assert parsed["passed"] is False


def test_critic_offline_fallback_on_empty_key():
    critic = TeacherCritic(api_key="", cache_dir="logs/test_critic_cache")
    result = critic.critique(
        prompt="Write a complete sentence.",
        response="The telescope observed a distant galaxy.",
        task="english",
    )
    assert isinstance(result, CritiqueResult)
    assert 0.0 <= result.score <= 1.0


def test_critic_flags_empty_response():
    critic = TeacherCritic(api_key="", cache_dir="logs/test_critic_cache")
    result = critic.critique(
        prompt="Summarize this text.",
        response="",
        task="summarization",
    )
    assert result.passed is False
    assert result.score == 0.0
    assert len(result.failures) > 0


def test_critic_cache_avoids_repeated_call(tmp_path):
    critic = TeacherCritic(api_key="", cache_dir=tmp_path)
    res1 = critic.critique("Test prompt", "Valid response sentence.", "english")
    res2 = critic.critique("Test prompt", "Valid response sentence.", "english")
    assert res1.score == res2.score
    assert len(list(tmp_path.glob("*.json"))) == 1
