"""eval/multi_teacher.py — Task 4.5: Multi-Teacher Ensemble & Disagreement Router.

Ensembles evaluations from multiple teacher models or rubric variants:
1. Teacher A: 70B Teacher Judge (Rubric B: Chain-of-Thought + Exemplars)
2. Teacher B: 70B Teacher Judge (Rubric A: Plain Direct Scoring)
3. Teacher C: Calibrated Local Evaluator (Rule & Signal Baseline)

Computes inter-teacher variance and routes high-disagreement edge cases
to a dedicated human review log, preventing single-teacher bias.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field

from eval.local_evaluator import LocalEvaluator
from eval.teacher_judge import TeacherJudge

logger = logging.getLogger("eval.multi_teacher")


class MultiTeacherResult(BaseModel):
    consensus_score: float = Field(ge=0.0, le=1.0)
    disagreement_delta: float = Field(ge=0.0, le=1.0)
    disagreement_std: float = Field(ge=0.0)
    requires_human_review: bool
    teacher_scores: Dict[str, float]
    details: Dict[str, Any]


class MultiTeacherEnsemble:
    """Multi-Teacher Judge ensemble with disagreement detection and audit logging."""

    def __init__(
        self,
        disagreement_threshold: float = 0.25,
        review_log_path: str | Path = "logs/human_review/disagreements.jsonl",
    ):
        self.threshold = disagreement_threshold
        self.review_log = Path(review_log_path)
        self.review_log.parent.mkdir(parents=True, exist_ok=True)

        self.teacher_rubric_b = TeacherJudge(rubric_version="B")
        self.teacher_rubric_a = TeacherJudge(rubric_version="A")
        self.local_evaluator = LocalEvaluator()

    def evaluate(
        self,
        prompt: str,
        response: str,
        stage: int = 1,
        reference: str = "",
        task: str = "",
    ) -> MultiTeacherResult:
        """Evaluate response with multiple teachers and measure consensus/disagreement."""
        # 1. Teacher 1: Rubric B
        res_b = self.teacher_rubric_b.judge(prompt, response, stage, reference, task)
        score_b = res_b.score

        # 2. Teacher 2: Rubric A
        res_a = self.teacher_rubric_a.judge(prompt, response, stage, reference, task)
        score_a = res_a.score

        # 3. Teacher 3: Local Evaluator
        res_local = self.local_evaluator.evaluate(prompt, response, stage, reference, task)
        score_local = res_local["score"]

        scores = [score_b, score_a, score_local]
        delta = max(scores) - min(scores)
        std_dev = float(np.std(scores))
        consensus = float(np.mean(scores))

        requires_review = delta >= self.threshold

        result = MultiTeacherResult(
            consensus_score=round(consensus, 4),
            disagreement_delta=round(delta, 4),
            disagreement_std=round(std_dev, 4),
            requires_human_review=requires_review,
            teacher_scores={
                "teacher_rubric_b": round(score_b, 4),
                "teacher_rubric_a": round(score_a, 4),
                "local_evaluator": round(score_local, 4),
            },
            details={
                "justification_b": res_b.justification,
                "justification_a": res_a.justification,
                "local_signals": res_local,
            },
        )

        # Log for human review if disagreement is substantial
        if requires_review:
            log_entry = {
                "prompt": prompt,
                "response": response,
                "stage": stage,
                "consensus_score": result.consensus_score,
                "disagreement_delta": result.disagreement_delta,
                "teacher_scores": result.teacher_scores,
            }
            with open(self.review_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.warning(
                "High inter-teacher disagreement (delta=%.3f >= %.3f). Logged to %s",
                delta, self.threshold, self.review_log
            )

        return result
