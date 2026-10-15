"""eval/reward.py — Task 3.3: Reward Scalar & Regression Gate.

Computes:
- Per-stage reward in [0, 1] as mean probe score across held-out stage probes.
- Macro reward across all active curriculum stages.
- Min-stage reward to prevent catastrophic stage collapse masked by averages.
- Regression detection comparing checkpoint rewards against tolerance thresholds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval.router import TieredRouter

logger = logging.getLogger("eval.reward")


class RewardCalculator:
    """Computes scalar alignment rewards and verifies regression boundaries."""

    def __init__(self, router: Optional[TieredRouter] = None):
        self.router = router or TieredRouter()

    @staticmethod
    def compute_stage_reward(scores: List[float]) -> float:
        """Compute mean score for a single stage in [0.0, 1.0]."""
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 4)

    @staticmethod
    def compute_macro_reward(stage_rewards: Dict[int, float]) -> float:
        """Macro average reward across all stages."""
        if not stage_rewards:
            return 0.0
        return round(sum(stage_rewards.values()) / len(stage_rewards), 4)

    @staticmethod
    def check_regression(
        baseline_reward: float,
        current_reward: float,
        tolerance: float = 0.05,
    ) -> Tuple[bool, float]:
        """Check if current reward has regressed beyond tolerance vs. baseline.

        Returns (is_regression, delta).
        """
        delta = current_reward - baseline_reward
        is_regression = delta < -tolerance
        return is_regression, round(delta, 4)

    def evaluate_stage_probes(
        self,
        probe_path: str | Path,
        stage: int,
        student_responses: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a stage probe bank file.

        If student_responses is provided, scores them; otherwise scores the references.
        """
        path = Path(probe_path)
        if not path.exists():
            raise FileNotFoundError(f"Probe bank not found: {path}")

        probes = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    probes.append(json.loads(line))

        scores: List[float] = []
        details: List[Dict[str, Any]] = []

        for idx, probe in enumerate(probes):
            prompt = probe.get("prompt", "")
            reference = probe.get("reference", "")
            task = probe.get("task", "")

            # If response not provided, evaluate reference as golden probe check
            if student_responses and idx < len(student_responses):
                response = student_responses[idx]
            else:
                response = reference or prompt

            routed = self.router.route(
                prompt=prompt,
                response=response,
                stage=stage,
                reference=reference,
                task=task,
            )
            scores.append(routed["score"])
            details.append({
                "probe_id": probe.get("id", f"s{stage}-{idx}"),
                "prompt": prompt,
                "score": routed["score"],
                "passed": routed["passed"],
                "routed_to": routed["routed_to"],
            })

        mean_reward = self.compute_stage_reward(scores)
        pass_rate = (sum(1 for s in scores if s >= 0.70) / len(scores)) if scores else 0.0

        return {
            "stage": stage,
            "probe_count": len(probes),
            "mean_reward": mean_reward,
            "pass_rate": round(pass_rate, 4),
            "probe_details": details,
        }

    def evaluate_all_stages(
        self,
        probe_dir: str | Path = "data/probes",
        stages: Tuple[int, ...] = (1, 2, 3, 4, 5),
    ) -> Dict[str, Any]:
        """Evaluate all stage probe banks and compute comprehensive reward metrics."""
        probe_dir = Path(probe_dir)
        stage_rewards: Dict[int, float] = {}
        pass_rates: Dict[int, float] = {}
        stage_reports: Dict[int, Any] = {}

        for stage in stages:
            probe_file = probe_dir / f"stage_{stage}.jsonl"
            if not probe_file.exists():
                logger.warning("Stage %d probe file missing: %s", stage, probe_file)
                continue

            report = self.evaluate_stage_probes(probe_file, stage=stage)
            stage_rewards[stage] = report["mean_reward"]
            pass_rates[stage] = report["pass_rate"]
            stage_reports[stage] = report

        macro_reward = self.compute_macro_reward(stage_rewards)
        min_reward = min(stage_rewards.values()) if stage_rewards else 0.0

        return {
            "macro_reward": macro_reward,
            "min_stage_reward": round(min_reward, 4),
            "stage_rewards": stage_rewards,
            "stage_pass_rates": pass_rates,
            "stage_reports": stage_reports,
            "router_telemetry": self.router.telemetry,
        }
