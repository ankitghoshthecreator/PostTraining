"""tests/test_generator_failover.py — Synthetic generator failover and fallback tests."""

from rlaif.generator import SyntheticGenerator


def test_generator_offline_fallback_produces_valid_samples():
    gen = SyntheticGenerator()
    samples = gen._generate_offline_fallback(stage=3, count=4)
    assert len(samples) == 4
    for s in samples:
        assert s["stage"] == 3
        assert len(s["prompt"]) > 5
        assert len(s["target"]) > 5


def test_generator_failover_when_ollama_disabled(monkeypatch):
    monkeypatch.delenv("USE_OLLAMA", raising=False)
    gen = SyntheticGenerator()
    assert gen.is_ollama_available() is False
    samples = gen.generate("Test prompt", stage=1, expected_count=3)
    assert len(samples) == 3


def test_generator_failover_when_connection_fails(monkeypatch):
    monkeypatch.setenv("USE_OLLAMA", "1")
    # Point to nonexistent port to trigger failover
    gen = SyntheticGenerator(ollama_url="http://127.0.0.1:59999")
    samples = gen.generate("Test prompt", stage=2, expected_count=2)
    assert len(samples) == 2


def test_generator_parse_valid_jsonl_output():
    gen = SyntheticGenerator()
    raw = '{"prompt": "P1", "target": "T1"}\n{"prompt": "P2", "target": "T2"}\n'
    parsed = gen._parse_jsonl_output(raw, stage=4, source="test")
    assert len(parsed) == 2
    assert parsed[0]["prompt"] == "P1"
    assert parsed[0]["stage"] == 4


def test_generator_skips_malformed_json_lines():
    gen = SyntheticGenerator()
    raw = '{"prompt": "P1", "target": "T1"}\nNOT_JSON\n{"prompt": "P2"}\n'
    parsed = gen._parse_jsonl_output(raw, stage=5, source="test")
    assert len(parsed) == 1
    assert parsed[0]["prompt"] == "P1"
