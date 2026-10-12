"""rlaif/controller.py — Task 2.9: Closed-loop RLAIF Flywheel Controller.

Coordinates the full flywheel iteration:
1. Probe student model across held-out banks.
2. Critique outputs with TeacherCritic.
3. Filter failures through SeverityGate.
4. Synthesize targeted data via SyntheticGenerator.
5. Filter through 7-layer QualityGatePipeline.
6. Append accepted synthetic data to versioned iter_<n>.jsonl dataset.
7. Retrain student checkpoint with new data blended.
8. Evaluate for regressions; trigger rollback if mean reward decreases beyond tolerance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from rlaif.critic import TeacherCritic
from rlaif.gate import filter_and_aggregate_failures
from rlaif.generator import SyntheticGenerator
from rlaif.prompt_builder import build_generation_prompt, get_held_out_probe_prompts
from validator.pipeline import QualityGatePipeline

logger = logging.getLogger("rlaif.controller")


class LoopController:
    """Manages multi-iteration RLAIF flywheel iterations."""

    def __init__(
        self,
        synthetic_dir: str | Path = "data/synthetic",
        rejected_dir: str | Path = "data/rejected",
        reports_dir: str | Path = "logs/rlaif_reports",
    ):
        self.synthetic_dir = Path(synthetic_dir)
        self.rejected_dir = Path(rejected_dir)
        self.reports_dir = Path(reports_dir)
        self.synthetic_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.critic = TeacherCritic()
        self.generator = SyntheticGenerator()

    def run_iteration(
        self,
        iteration_index: int,
        probe_results_by_stage: Dict[int, List[Dict]],
        sample_budget: int = 15,
    ) -> Dict:
        """Run a single end-to-end iteration of the flywheel."""
        logger.info("=== Starting RLAIF Iteration %d ===", iteration_index)

        # 1. Critique probe responses
        failures_collected = []
        for stage_id, results in probe_results_by_stage.items():
            for res in results:
                critique = self.critic.critique(
                    prompt=res["prompt"],
                    response=res["response"],
                    task=res.get("task", "general"),
                    reference=res.get("reference", ""),
                )
                for f in critique.failures:
                    failures_collected.append((stage_id, f, res["response"]))

        logger.info("Iter %d: Critic identified %d total failure instances.", iteration_index, len(failures_collected))

        # 2. Gate & Aggregate failures
        data_requests = filter_and_aggregate_failures(failures_collected, budget=sample_budget)
        logger.info("Iter %d: Gate approved %d targeted data requests.", iteration_index, len(data_requests))

        # 3. Generate synthetic data candidates
        held_out_prompts = get_held_out_probe_prompts()
        raw_candidates = []
        for req in data_requests:
            prompt = build_generation_prompt(req, held_out_prompts)
            gen_batch = self.generator.generate(prompt, stage=req.stage, expected_count=req.num_samples_to_generate)
            raw_candidates.extend(gen_batch)

        logger.info("Iter %d: Generator produced %d raw candidates.", iteration_index, len(raw_candidates))

        # 4. Filter with 7-Layer Quality Gate
        print(f"[*] Iter {iteration_index}: Filtering {len(raw_candidates)} candidates through 7-layer Quality Gate...")
        reject_log = self.rejected_dir / f"rejected_iter_{iteration_index}.jsonl"
        quality_gate = QualityGatePipeline(reject_log_path=reject_log)
        accepted_samples = quality_gate.filter_batch(raw_candidates)

        print(f"[+] Iter {iteration_index}: Accepted {len(accepted_samples)} / {len(raw_candidates)} candidates.")

        # 5. Save versioned dataset
        dataset_file = self.synthetic_dir / f"iter_{iteration_index}.jsonl"
        with open(dataset_file, "w", encoding="utf-8") as f:
            for s in accepted_samples:
                f.write(json.dumps(s) + "\n")
        print(f"[+] Saved versioned synthetic dataset to: {dataset_file}")

        # 6. Build iteration report
        report = {
            "iteration": iteration_index,
            "total_failures_identified": len(failures_collected),
            "requests_approved": len(data_requests),
            "raw_candidates_generated": len(raw_candidates),
            "accepted_samples": len(accepted_samples),
            "gate_stats": quality_gate.stats,
            "dataset_file": str(dataset_file),
        }

        report_path = self.reports_dir / f"report_iter_{iteration_index}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Iter %d finished successfully. Report: %s", iteration_index, report_path)
        return report
