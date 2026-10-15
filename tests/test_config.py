"""tests/test_config.py — Pydantic v2 configuration schema validation tests."""

import pytest
from config import load, CurriculumConfig, ModelConfig, RLAIFConfig, ValidatorConfig, EvalConfig


def test_load_curriculum_config():
    cfg = load("configs/curriculum.yaml")
    assert isinstance(cfg, CurriculumConfig)
    assert len(cfg.stages) == 5
    assert cfg.stages[0].replay_ratio == 0.0
    assert cfg.stages[-1].replay_ratio == 0.4


def test_load_model_config():
    cfg = load("configs/model.yaml")
    assert isinstance(cfg, ModelConfig)
    assert cfg.lora.r == 16
    assert cfg.lora.lora_alpha == 32
    assert cfg.quantization.load_in_4bit is True


def test_load_rlaif_and_validator_configs():
    rlaif_cfg = load("configs/rlaif.yaml")
    assert isinstance(rlaif_cfg, RLAIFConfig)
    assert rlaif_cfg.priority_max == 5

    val_cfg = load("configs/validator.yaml")
    assert isinstance(val_cfg, ValidatorConfig)
    assert val_cfg.min_chars == 10
    assert val_cfg.bigram_repetition_max == 0.35


def test_load_eval_and_nonexistent_file():
    eval_cfg = load("configs/eval.yaml")
    assert isinstance(eval_cfg, EvalConfig)
    assert eval_cfg.local_pass_threshold == 0.70

    with pytest.raises(FileNotFoundError):
        load("configs/nonexistent_config_file.yaml")
