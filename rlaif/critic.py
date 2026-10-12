"""rlaif/critic.py — Task 2.3: Teacher Critic Client (Groq/OpenAI compatible).

Uses Llama-3-70B on Groq to critique student outputs according to rubric & taxonomy.
Includes:
- Pydantic v2 schemas: FailureItem, CritiqueResult
- Robust parsing: sanitize <think> -> extract JSON -> validate
- Disk-backed caching to avoid re-calling teacher API for identical responses
- Fallback/mock mode for offline/test environments
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from rlaif.sanitize import extract_json_payload
from rlaif.taxonomy import FailureCategory, Severity

logger = logging.getLogger("rlaif.critic")


class FailureItem(BaseModel):
    category: FailureCategory
    severity: Severity
    explanation: str
    priority: int = Field(default=3, ge=1, le=10)
    suggested_correction: Optional[str] = None


class CritiqueResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    failures: List[FailureItem] = Field(default_factory=list)


CRITIC_SYSTEM_PROMPT = """You are an expert AI evaluator judging student LLM outputs.
Evaluate the model response against the prompt, task requirements, and reference.
You must categorize any failure into exactly one of: repetition, off_topic, instruction_ignored, grammar_error, punctuation_error, hallucination, format_violation.
Severity must be one of: CRITICAL, HIGH, MEDIUM, LOW.
Assign a priority from 1 (highest priority to fix) to 10 (lowest).

Respond ONLY with valid JSON in this schema:
{
  "score": float between 0.0 and 1.0,
  "passed": bool,
  "failures": [
    {
      "category": "repetition|off_topic|instruction_ignored|grammar_error|punctuation_error|hallucination|format_violation",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "explanation": "...",
      "priority": int,
      "suggested_correction": "..."
    }
  ]
}
"""


class TeacherCritic:
    """Evaluates student responses using 70B teacher model with cache & retry."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        cache_dir: str | Path = "logs/critic_cache",
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, prompt: str, response: str, task: str) -> str:
        h = hashlib.sha256(f"{self.model_name}:{task}:{prompt}:{response}".encode()).hexdigest()
        return h

    def critique(self, prompt: str, response: str, task: str, reference: str = "") -> CritiqueResult:
        cache_file = self.cache_dir / f"{self._cache_key(prompt, response, task)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return CritiqueResult(**data)
            except Exception:
                pass

        # If no API key is provided, execute heuristic fallback critic
        if not self.api_key:
            return self._heuristic_fallback_critique(prompt, response, task, reference)

        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            user_msg = (
                f"Task: {task}\n"
                f"Prompt:\n{prompt}\n\n"
                f"Reference:\n{reference}\n\n"
                f"Student Response:\n{response}"
            )

            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw_content = completion.choices[0].message.content or "{}"
            json_str = extract_json_payload(raw_content)
            result = CritiqueResult.model_validate_json(json_str)

            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json())

            return result

        except Exception as e:
            logger.warning("Groq API critique failed (%s). Falling back to heuristic critique.", e)
            return self._heuristic_fallback_critique(prompt, response, task, reference)

    def _heuristic_fallback_critique(
        self, prompt: str, response: str, task: str, reference: str
    ) -> CritiqueResult:
        """Lightweight offline rule critic ensuring tests and local loops execute

        even without an active Groq API key.
        """
        resp = response.strip()
        failures: List[FailureItem] = []

        if not resp:
            failures.append(
                FailureItem(
                    category=FailureCategory.INSTRUCTION_IGNORED,
                    severity=Severity.CRITICAL,
                    explanation="Empty generation returned.",
                    priority=1,
                )
            )
            return CritiqueResult(score=0.0, passed=False, failures=failures)

        # Repetition check
        words = resp.lower().split()
        if len(words) >= 4 and len(set(words)) / len(words) < 0.4:
            failures.append(
                FailureItem(
                    category=FailureCategory.REPETITION,
                    severity=Severity.HIGH,
                    explanation="Significant token repetition detected in output.",
                    priority=2,
                )
            )

        # Grammar / punctuation checks
        if task == "punctuation" and ("," not in resp and "." not in resp):
            failures.append(
                FailureItem(
                    category=FailureCategory.PUNCTUATION_ERROR,
                    severity=Severity.HIGH,
                    explanation="Missing expected punctuation marks.",
                    priority=3,
                )
            )

        score = 1.0 if not failures else max(0.1, 1.0 - 0.3 * len(failures))
        return CritiqueResult(score=score, passed=len(failures) == 0, failures=failures)
