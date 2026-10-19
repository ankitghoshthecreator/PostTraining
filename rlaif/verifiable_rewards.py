"""rlaif/verifiable_rewards.py — Task 4.3: Deterministic Verifiable Rewards.

Computes objective, rule-verifiable rewards without LLM-as-a-judge:
1. Math & Arithmetic Reasoning: exact answer extraction and numerical comparison.
2. Format & Schema Verification: JSON validity, required keys, regex structure.
3. Constraint Compliance: word count, terminal punctuation, case constraints.

Returns binary reward R in {0.0, 1.0} or soft bounded reward in [0.0, 1.0].
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def extract_boxed_or_terminal_number(text: str) -> Optional[float]:
    """Extract final numerical answer from reasoning trace.

    Supports:
    - \\boxed{...}
    - "Answer: ..."
    - Final integer/float in text
    """
    # 1. Look for LaTeX boxed: \boxed{123}
    boxed = re.findall(r"\\boxed\{([^\}]+)\}", text)
    if boxed:
        try:
            return float(boxed[-1].strip().replace(",", ""))
        except ValueError:
            pass

    # 2. Look for "Answer: 123"
    ans_match = re.findall(r"(?:answer|result)(?:\s*is|\s*:)\s*(-?[\d\.]+)", text, flags=re.IGNORECASE)
    if ans_match:
        try:
            return float(ans_match[-1].strip().replace(",", ""))
        except ValueError:
            pass

    # 3. Look for final standalone number
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass

    return None


def verify_math_reasoning(
    prompt: str,
    response: str,
    ground_truth: float,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """Verify mathematical response against ground truth."""
    extracted = extract_boxed_or_terminal_number(response)
    if extracted is None:
        return {
            "reward": 0.0,
            "passed": False,
            "extracted": None,
            "ground_truth": ground_truth,
            "reason": "No numerical answer found in response",
        }

    is_correct = abs(extracted - ground_truth) <= tolerance
    return {
        "reward": 1.0 if is_correct else 0.0,
        "passed": is_correct,
        "extracted": extracted,
        "ground_truth": ground_truth,
        "reason": "Exact match" if is_correct else f"Mismatch: {extracted} != {ground_truth}",
    }


def verify_json_format(response: str, required_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Verify that response contains valid JSON with required keys."""
    resp = response.strip()
    # Strip potential code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", resp)
    payload_str = fence_match.group(1) if fence_match else resp

    try:
        data = json.loads(payload_str)
        if not isinstance(data, dict):
            return {"reward": 0.0, "passed": False, "reason": "Parsed JSON is not a dictionary"}

        if required_keys:
            missing = [k for k in required_keys if k not in data]
            if missing:
                return {"reward": 0.0, "passed": False, "reason": f"Missing required keys: {missing}"}

        return {"reward": 1.0, "passed": True, "reason": "Valid JSON schema"}
    except Exception as e:
        return {"reward": 0.0, "passed": False, "reason": f"JSON parse error: {str(e)}"}


def verify_constraint_compliance(
    response: str,
    max_words: Optional[int] = None,
    min_words: Optional[int] = None,
    must_end_with: Optional[Tuple[str, ...]] = (".", "!", "?"),
) -> Dict[str, Any]:
    """Verify structural formatting constraints."""
    resp = response.strip()
    if not resp:
        return {"reward": 0.0, "passed": False, "reason": "Empty text"}

    words = resp.split()
    if min_words and len(words) < min_words:
        return {"reward": 0.0, "passed": False, "reason": f"Word count {len(words)} < min {min_words}"}

    if max_words and len(words) > max_words:
        return {"reward": 0.0, "passed": False, "reason": f"Word count {len(words)} > max {max_words}"}

    if must_end_with and not resp.endswith(must_end_with):
        return {"reward": 0.0, "passed": False, "reason": f"Does not end with required punctuation {must_end_with}"}

    return {"reward": 1.0, "passed": True, "reason": "All constraints satisfied"}
