"""rlaif/sanitize.py — Task 2.4: <think> sanitization.

Sanitizes reasoning model outputs by removing complete and unclosed <think> blocks,
handling arbitrary casing, stray tags, nested tags, and avoiding false positives
where reasoning tokens leak into JSON or failure extraction.
"""

from __future__ import annotations

import re


def sanitize_think_tags(text: str) -> str:
    """Strip all <think>...</think> sections from text, including unclosed tags.

    Handles:
    - Standard paired tags (<think>...</think>)
    - Case-insensitive variants (<THINK>...</THINK>, <Think>...)
    - Unclosed tags (<think>... at end of stream)
    - Multiline reasoning blocks
    """
    if not text:
        return ""

    # Remove closed <think>...</think> blocks (case-insensitive, multiline, dot matches newline)
    cleaned = re.sub(r"(?is)<\s*think\s*>.*?<\s*/\s*think\s*>", "", text)

    # Remove unclosed <think>... blocks that run to the end of string
    cleaned = re.sub(r"(?is)<\s*think\s*>.*$", "", cleaned)

    # Remove stray opening or closing tags
    cleaned = re.sub(r"(?is)<\s*/?\s*think\s*>", "", cleaned)

    return cleaned.strip()


def extract_json_payload(raw_text: str) -> str:
    """Extract valid JSON substring from LLM response after stripping <think> tags

    and markdown code fences.
    """
    text = sanitize_think_tags(raw_text)

    # Match ```json ... ``` or ``` ... ```
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, text)
    if match:
        candidate = match.group(1).strip()
        if candidate.startswith("{") or candidate.startswith("["):
            return candidate

    # Fallback: find first '{' and last matching '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text.strip()
