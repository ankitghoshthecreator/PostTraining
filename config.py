"""Configuration loader with Pydantic v2 validation and frozen schemas."""

from pathlib import Path
from typing import List, Literal, Optional, Union
import yaml
from pydantic import BaseModel, ConfigDict, Field


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "float16"


class LoraConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    learning_rate: float = 2.0e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    gradient_accumulation_steps: int = 8
    per_device_train_batch_size: int = 1
    gradient_checkpointing: bool = True
    use_cache: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    max_seq_len: int = 512


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


class StageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    file: str
    epochs: int = 3
    replay_ratio: float = Field(default=0.0, ge=0.0, le=1.0)


class EarlyStopConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric: str = "probe_reward"
    patience: int = 3


class CurriculumConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stages: List[StageConfig]
    max_steps_per_stage: int = 2000
    early_stop: EarlyStopConfig = Field(default_factory=EarlyStopConfig)


class RLAIFConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    severity_gate: List[str] = Field(default_factory=lambda: ["CRITICAL", "HIGH"])
    priority_max: int = 5
    max_iterations: int = 200
    requests_per_iteration: int = 40
    samples_per_request: int = 8
    min_new_samples: int = 5
    regress_tolerance: float = 0.05
    target_reward: float = 0.95


class ValidatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    min_chars: int = 10
    max_chars: int = 4096
    bigram_repetition_max: float = 0.35
    jaccard_circularity_max: float = 0.85
    dedup: str = "sha256"


class EvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    local_pass_threshold: float = 0.70
    rubric_version: str = "B"
    teacher_model: str = "llama-3-70b"


ConfigType = Union[CurriculumConfig, ModelConfig, RLAIFConfig, ValidatorConfig, EvalConfig]


def load(path: Union[str, Path]) -> ConfigType:
    """Load and validate YAML configuration into a frozen Pydantic object."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    filename = config_path.name.lower()
    if "curriculum" in filename or "stages" in data:
        return CurriculumConfig.model_validate(data)
    elif "model" in filename or "model_name" in data:
        return ModelConfig.model_validate(data)
    elif "rlaif" in filename or "severity_gate" in data:
        return RLAIFConfig.model_validate(data)
    elif "validator" in filename or "bigram_repetition_max" in data:
        return ValidatorConfig.model_validate(data)
    elif "eval" in filename or "local_pass_threshold" in data:
        return EvalConfig.model_validate(data)
    else:
        raise ValueError(f"Unable to determine config type for: {config_path}")
