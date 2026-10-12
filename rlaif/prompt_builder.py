"""rlaif/prompt_builder.py — Task 2.6: Targeted prompt builder for synthetic generator.

Constructs synthetic data generation prompts that focus specifically on
mitigating the observed failures, providing positive and negative exemplars
while enforcing a strict probe leakage guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Set

from rlaif.gate import TargetedDataRequest
from rlaif.taxonomy import FAILURE_DESCRIPTIONS, STAGE_NAMES


def get_held_out_probe_prompts(probe_dir: str | Path = "data/probes") -> Set[str]:
    """Collect all probe prompts to guard against exemplar leakage."""
    prompts = set()
    for p in Path(probe_dir).glob("stage_*.jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    prompts.add(item["prompt"].strip().lower())
    return prompts


def build_generation_prompt(
    request: TargetedDataRequest,
    held_out_prompts: Set[str],
) -> str:
    """Build a prompt instructing the generator model to create synthetic pairs."""
    stage_name = STAGE_NAMES.get(request.stage, f"Stage {request.stage}")
    cat_desc = FAILURE_DESCRIPTIONS.get(request.category, "")

    # Leakage Guard: Filter out any bad exemplars that match held-out probe prompts
    safe_bad_outputs = [
        b for b in request.sample_bad_outputs
        if b.strip().lower() not in held_out_prompts
    ]

    bad_examples_str = ""
    if safe_bad_outputs:
        bad_examples_str = "\n".join(f"- Negative student output to avoid: \"{b}\"" for b in safe_bad_outputs)
        bad_examples_str = f"\nObserved Student Weaknesses:\n{bad_examples_str}\n"

    prompt = f"""You are a specialized training data synthesis agent.
Task Curriculum: {stage_name} (Stage {request.stage})
Target Failure Mode to Mitigate: {request.category.value}
Description: {cat_desc}
{bad_examples_str}
Instructions:
1. Generate exactly {request.num_samples_to_generate} diverse, high-quality training pairs for Stage {request.stage}.
2. Ensure the responses explicitly demonstrate how to solve the task correctly without exhibiting '{request.category.value}'.
3. Output format must be strictly JSON Lines, one sample per line:
{{"prompt": "...", "target": "..."}}
Do not output markdown fences or commentary. Only raw JSON lines.
"""
    return prompt.strip()
