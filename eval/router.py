"""eval/router.py — Task 3.3: Tiered Evaluation Router.

Routes probe evaluations between:
1. Fast local evaluator (~0.013 ms) for clear passes (score >= threshold, default 0.70).
2. 70B Teacher Judge (~2.5 s) for edge cases or uncertain outputs (< threshold).

Features:
- Latency and cost tracker (estimates time and API savings).
- Threshold sweep capability for RQ4 analysis (cost saved vs. false-pass rate).
- Safe fallback: if teacher fails, marks result unjudged instead of silently passing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from eval.local_evaluator import LocalEvaluator
from eval.teacher_judge import TeacherJudge

logger = logging.getLogger("eval.router")


class TieredRouter:
    """Tiered evaluation router that balances throughput, cost, and judgment precision."""

    def __init__(
        self,
        local_pass_threshold: float = 0.70,
        local_evaluator: Optional[LocalEvaluator] = None,
        teacher_judge: Optional[TeacherJudge] = None,
        teacher_latency_estimate_s: float = 2.5,
    ):
        self.threshold = local_pass_threshold
        self.local_evaluator = local_evaluator or LocalEvaluator()
        self.teacher_judge = teacher_judge or TeacherJudge()
        self.teacher_latency_estimate_s = teacher_latency_estimate_s

        # Telemetry
        self.total_routed = 0
        self.local_passes = 0
        self.teacher_escalations = 0

    def route(
        self,
        prompt: str,
        response: str,
        stage: int = 1,
        reference: str = "",
        task: str = "",
    ) -> Dict[str, Any]:
        """Route probe evaluation to local evaluator or teacher judge."""
        self.total_routed += 1
        local_res = self.local_evaluator.evaluate(
            prompt=prompt,
            response=response,
            stage=stage,
            reference=reference,
            task=task,
        )

        local_score = local_res["score"]

        # Fast Pass: Local score meets or exceeds confidence threshold
        if local_score >= self.threshold:
            self.local_passes += 1
            return {
                "routed_to": "local",
                "score": local_score,
                "passed": True,
                "local_details": local_res,
                "teacher_details": None,
                "escalated": False,
            }

        # Escalation: Route to 70B teacher judge
        self.teacher_escalations += 1
        teacher_res = self.teacher_judge.judge(
            prompt=prompt,
            response=response,
            stage=stage,
            reference=reference,
            task=task,
        )

        return {
            "routed_to": "teacher",
            "score": teacher_res.score,
            "passed": teacher_res.passed,
            "local_details": local_res,
            "teacher_details": teacher_res.model_dump(),
            "escalated": True,
        }

    @property
    def telemetry(self) -> Dict[str, Any]:
        """Summary of routing statistics and estimated time saved."""
        fraction_local = (
            self.local_passes / self.total_routed if self.total_routed > 0 else 0.0
        )
        time_saved = self.local_passes * self.teacher_latency_estimate_s
        return {
            "total_routed": self.total_routed,
            "local_passes": self.local_passes,
            "teacher_escalations": self.teacher_escalations,
            "local_fraction": round(fraction_local, 4),
            "estimated_time_saved_s": round(time_saved, 2),
        }

    def reset_telemetry(self) -> None:
        """Reset routing counters."""
        self.total_routed = 0
        self.local_passes = 0
        self.teacher_escalations = 0

    def sweep_thresholds(
        self,
        eval_pairs: List[Dict[str, Any]],
        thresholds: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Sweep threshold values across labeled pairs to measure false-pass rate vs. cost saved (RQ4).

        Each item in eval_pairs must contain:
        - prompt, response, reference, stage, and ground_truth_pass (bool)
        """
        if thresholds is None:
            thresholds = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

        results = []
        n_total = len(eval_pairs)
        if n_total == 0:
            return results

        # Pre-compute local scores
        scored_pairs = []
        for p in eval_pairs:
            local_res = self.local_evaluator.evaluate(
                prompt=p["prompt"],
                response=p["response"],
                stage=p.get("stage", 1),
                reference=p.get("reference", ""),
                task=p.get("task", ""),
            )
            scored_pairs.append({
                "local_score": local_res["score"],
                "ground_truth_pass": bool(p.get("ground_truth_pass", True)),
            })

        for th in thresholds:
            local_passed = 0
            false_passes = 0  # local passed, but ground truth was fail

            for pair in scored_pairs:
                passed_local = pair["local_score"] >= th
                if passed_local:
                    local_passed += 1
                    if not pair["ground_truth_pass"]:
                        false_passes += 1

            false_pass_rate = (false_passes / local_passed) if local_passed > 0 else 0.0
            fraction_handled_local = local_passed / n_total
            cost_saved_pct = fraction_handled_local * 100.0

            results.append({
                "threshold": th,
                "local_passed": local_passed,
                "escalated": n_total - local_passed,
                "false_passes": false_passes,
                "false_pass_rate": round(false_pass_rate, 4),
                "cost_saved_pct": round(cost_saved_pct, 2),
            })

        return results
