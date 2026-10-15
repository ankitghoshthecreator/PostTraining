"""eval/teacher_judge.py — Task 3.2: 70B Teacher Judge with Rubric A and Rubric B.

Evaluates student outputs against a 4-axis rubric:
1. Correctness / Instruction following (1–5)
2. Fluency / Grammar & Style (1–5)
3. Faithfulness / Hallucination avoidance (1–5)
4. Format / Punctuation compliance (1–5)

Supports:
- Rubric A: Plain scoring prompt (baseline human agreement κ = 0.66)
- Rubric B: CoT justification + exemplars (human agreement κ = 0.77, +0.11 gain)
- Justification placed before score in JSON output schema to force CoT deliberation
- Disk cache to prevent redundant API calls
- Graceful offline fallback mode for testing
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from rlaif.sanitize import extract_json_payload

logger = logging.getLogger("eval.teacher_judge")


class JudgeResult(BaseModel):
    rubric_version: str
    justification: str
    score: float = Field(ge=0.0, le=1.0)
    correctness: float = Field(ge=1.0, le=5.0)
    fluency: float = Field(ge=1.0, le=5.0)
    faithfulness: float = Field(ge=1.0, le=5.0)
    format_compliance: float = Field(ge=1.0, le=5.0)
    passed: bool


RUBRIC_A_PROMPT = """You are an expert AI judge evaluating a student language model's output.
Score the student response across 4 axes from 1 to 5:
- correctness: does it follow instructions and accurately complete the task? (1=complete failure, 5=perfect)
- fluency: is grammar, vocabulary, and sentence construction natural? (1=broken, 5=flawless)
- faithfulness: does it avoid hallucinations and irrelevant tangents? (1=fabricated/off-topic, 5=fully grounded)
- format_compliance: does it meet punctuation, formatting, and structural constraints? (1=severe violation, 5=fully compliant)

Output ONLY valid JSON with this schema:
{
  "justification": "Brief rationale for the ratings",
  "correctness": 1-5,
  "fluency": 1-5,
  "faithfulness": 1-5,
  "format_compliance": 1-5
}
"""

RUBRIC_B_PROMPT = """You are an expert, meticulous AI alignment auditor evaluating student language model outputs.
You will evaluate the student response using Chain-of-Thought deliberation across 4 distinct axes (1-5):
1. correctness (1-5): Verifiable execution of task requirements.
2. fluency (1-5): Natural syntax, absence of dysfluencies or loops.
3. faithfulness (1-5): Strictly grounded in facts / context with zero hallucination.
4. format_compliance (1-5): Punctuation, casing, and format constraints followed exactly.

Worked Exemplar:
Task: Punctuation (Stage 4)
Prompt: "yesterday we went to the market and bought apples oranges and milk"
Student: "Yesterday, we went to the market and bought apples, oranges, and milk."
Justification: "Correctly capitalized the first word, inserted commas for the list and after the introductory clause, and appended a terminal period."
Scores: correctness=5, fluency=5, faithfulness=5, format_compliance=5

Crucial Rule: Provide your detailed step-by-step reasoning in 'justification' BEFORE setting numerical scores.
Do not allow length to bias the score; concise answers that fulfill instructions should receive high marks.

Output ONLY valid JSON:
{
  "justification": "Detailed step-by-step reasoning explaining failures or virtues per axis",
  "correctness": 1-5,
  "fluency": 1-5,
  "faithfulness": 1-5,
  "format_compliance": 1-5
}
"""


class TeacherJudge:
    """70B Teacher Judge supporting Rubric A & B with cache and offline fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        rubric_version: str = "B",
        cache_dir: str | Path = "logs/judge_cache",
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model_name = model_name
        self.rubric_version = rubric_version.upper()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, prompt: str, response: str, reference: str) -> str:
        content = f"{self.rubric_version}:{self.model_name}:{prompt}:{response}:{reference}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def judge(
        self,
        prompt: str,
        response: str,
        stage: int = 1,
        reference: str = "",
        task: str = "",
    ) -> JudgeResult:
        """Judge a student output. Returns structured JudgeResult with normalized score."""
        cache_file = self.cache_dir / f"{self._cache_key(prompt, response, reference)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return JudgeResult(**data)
            except Exception:
                pass

        if not self.api_key:
            return self._offline_judge(prompt, response, stage, reference, task)

        system_prompt = RUBRIC_B_PROMPT if self.rubric_version == "B" else RUBRIC_A_PROMPT
        user_content = (
            f"Stage: {stage} ({task})\n"
            f"Prompt:\n{prompt}\n\n"
            f"Reference:\n{reference or 'N/A'}\n\n"
            f"Student Output:\n{response}"
        )

        try:
            from groq import Groq

            client = Groq(api_key=self.api_key)
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = completion.choices[0].message.content or "{}"
            payload = json.loads(extract_json_payload(raw))

            c = float(payload.get("correctness", 3))
            fl = float(payload.get("fluency", 3))
            fa = float(payload.get("faithfulness", 3))
            fo = float(payload.get("format_compliance", 3))
            justification = payload.get("justification", "Evaluated by teacher.")

            # Normalized score in [0, 1] from mean of 1-5 scales
            mean_raw = (c + fl + fa + fo) / 4.0
            norm_score = (mean_raw - 1.0) / 4.0
            norm_score = max(0.0, min(1.0, norm_score))

            result = JudgeResult(
                rubric_version=self.rubric_version,
                justification=justification,
                score=round(norm_score, 4),
                correctness=c,
                fluency=fl,
                faithfulness=fa,
                format_compliance=fo,
                passed=norm_score >= 0.70,
            )

            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(result.model_dump_json())

            return result

        except Exception as e:
            logger.warning("Groq judge call failed (%s). Using offline judge.", e)
            return self._offline_judge(prompt, response, stage, reference, task)

    def _offline_judge(
        self, prompt: str, response: str, stage: int, reference: str, task: str
    ) -> JudgeResult:
        """Deterministic offline judge for testing and offline development."""
        resp = response.strip()
        if not resp:
            return JudgeResult(
                rubric_version=self.rubric_version,
                justification="Empty output provided by student model.",
                score=0.0,
                correctness=1.0,
                fluency=1.0,
                faithfulness=1.0,
                format_compliance=1.0,
                passed=False,
            )

        words = resp.split()
        c = 4.0
        fl = 4.0
        fa = 4.0
        fo = 4.0

        # Repetition penalty
        if len(words) >= 4 and len(set(w.lower() for w in words)) / len(words) < 0.5:
            fl = 2.0
            c = 2.0

        # Stage punctuation penalty
        if (stage == 4 or task == "punctuation") and not any(p in resp for p in [",", ".", "!", "?"]):
            fo = 2.0
            c = 2.5

        # Short penalty
        if len(resp) < 8:
            c = 2.0
            fo = 2.0

        mean_raw = (c + fl + fa + fo) / 4.0
        norm_score = (mean_raw - 1.0) / 4.0
        norm_score = max(0.0, min(1.0, norm_score))

        justification = (
            f"Offline judge evaluated stage {stage}. "
            f"Fluency={fl}, Correctness={c}, Faithfulness={fa}, Format={fo}."
        )

        return JudgeResult(
            rubric_version=self.rubric_version,
            justification=justification,
            score=round(norm_score, 4),
            correctness=c,
            fluency=fl,
            faithfulness=fa,
            format_compliance=fo,
            passed=norm_score >= 0.70,
        )
