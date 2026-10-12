"""rlaif/taxonomy.py — Task 2.2: Failure taxonomy and data request templates.

Defines:
- Severity levels (CRITICAL, HIGH, MEDIUM, LOW)
- FailureCategory (7 categories: repetition, off_topic, instruction_ignored,
                    grammar_error, punctuation_error, hallucination, format_violation)
- Mapping from (FailureCategory, stage) -> data request template for targeted generation.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FailureCategory(str, Enum):
    REPETITION = "repetition"
    OFF_TOPIC = "off_topic"
    INSTRUCTION_IGNORED = "instruction_ignored"
    GRAMMAR_ERROR = "grammar_error"
    PUNCTUATION_ERROR = "punctuation_error"
    HALLUCINATION = "hallucination"
    FORMAT_VIOLATION = "format_violation"


# Descriptions of failure modes for prompts
FAILURE_DESCRIPTIONS: Dict[FailureCategory, str] = {
    FailureCategory.REPETITION: "Model loops words, phrases, or full sentences repeatedly without progressing.",
    FailureCategory.OFF_TOPIC: "Model drifts away from the core subject or introduces unrelated tangents.",
    FailureCategory.INSTRUCTION_IGNORED: "Model fails to perform the requested operation (e.g. summarize, correct, answer).",
    FailureCategory.GRAMMAR_ERROR: "Model produces syntactically incorrect, ungrammatical, or malformed sentences.",
    FailureCategory.PUNCTUATION_ERROR: "Model misses required punctuation, uses excessive commas, or omits sentence boundaries.",
    FailureCategory.HALLUCINATION: "Model asserts fabricated, factually contradictory, or nonsensical information.",
    FailureCategory.FORMAT_VIOLATION: "Model fails required structure, word-count limits, or markdown constraints.",
}

# Targeted templates per (stage, category)
STAGE_NAMES: Dict[int, str] = {
    1: "english_continuation",
    2: "summarization",
    3: "grammar_correction",
    4: "punctuation",
    5: "multi_turn_conversation",
}


def get_data_request_template(stage: int, category: FailureCategory) -> str:
    """Return a guidance string for the generator model to produce training samples

    addressing this specific failure mode in this stage.
    """
    stage_name = STAGE_NAMES.get(stage, f"stage_{stage}")
    cat_desc = FAILURE_DESCRIPTIONS.get(category, "Address model failure.")

    return (
        f"Stage: {stage} ({stage_name})\n"
        f"Target Failure to Mitigate: {category.value} — {cat_desc}\n"
        f"Goal: Produce diverse, high-quality prompt/target pairs demonstrating correct behavior "
        f"and explicitly counteracting {category.value}."
    )
