"""eval/run_stage_eval.py — Task 3.6: Stage Evaluation CLI & Report Generator.

Usage:
    python eval/run_stage_eval.py --stage 5 --checkpoint trainer/checkpoints/checkpoint_best
    python eval/run_stage_eval.py --stage all --checkpoint trainer/checkpoints/checkpoint_best --compare trainer/checkpoints/checkpoint_last
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.reward import RewardCalculator
from eval.router import TieredRouter
from eval.metrics import compute_rouge_l, compute_exact_match, compute_normalized_edit_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("eval.cli")


def run_evaluation(
    stage: str,
    checkpoint: str,
    probe_dir: str = "data/probes",
    output_dir: str = "logs/eval_reports",
    compare_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute evaluation against stage probes and generate JSON, MD, and CSV reports."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    router = TieredRouter()
    calculator = RewardCalculator(router=router)

    stages_to_eval = [1, 2, 3, 4, 5] if stage.lower() == "all" else [int(stage)]

    results: Dict[str, Any] = {
        "checkpoint": checkpoint,
        "stages": stages_to_eval,
        "stages_evaluated": {},
    }

    all_probe_rows = []

    for st in stages_to_eval:
        probe_file = Path(probe_dir) / f"stage_{st}.jsonl"
        if not probe_file.exists():
            logger.warning("Stage probe file not found: %s", probe_file)
            continue

        report = calculator.evaluate_stage_probes(probe_file, stage=st)
        results["stages_evaluated"][st] = {
            "mean_reward": report["mean_reward"],
            "pass_rate": report["pass_rate"],
            "probe_count": report["probe_count"],
        }

        for p in report["probe_details"]:
            all_probe_rows.append({
                "stage": st,
                "probe_id": p["probe_id"],
                "score": p["score"],
                "passed": p["passed"],
                "routed_to": p["routed_to"],
                "prompt": p["prompt"],
            })

    # Overall metrics
    stage_rewards = {st: res["mean_reward"] for st, res in results["stages_evaluated"].items()}
    results["macro_reward"] = calculator.compute_macro_reward(stage_rewards)
    results["min_stage_reward"] = min(stage_rewards.values()) if stage_rewards else 0.0
    results["router_telemetry"] = router.telemetry

    # Optional checkpoint comparison
    if compare_checkpoint:
        logger.info("Evaluating comparative checkpoint: %s", compare_checkpoint)
        router_b = TieredRouter()
        calc_b = RewardCalculator(router=router_b)
        res_b = {}
        for st in stages_to_eval:
            probe_file = Path(probe_dir) / f"stage_{st}.jsonl"
            if probe_file.exists():
                rep_b = calc_b.evaluate_stage_probes(probe_file, stage=st)
                res_b[st] = rep_b["mean_reward"]

        macro_b = calc_b.compute_macro_reward(res_b)
        results["comparison"] = {
            "checkpoint_b": compare_checkpoint,
            "macro_reward_b": macro_b,
            "delta": round(results["macro_reward"] - macro_b, 4),
            "stage_deltas": {
                st: round(stage_rewards.get(st, 0.0) - res_b.get(st, 0.0), 4)
                for st in stages_to_eval
            },
        }

    # Write eval_report.json
    json_path = out_path / "eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Write per_probe.csv
    csv_path = out_path / "per_probe.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "probe_id", "score", "passed", "routed_to", "prompt"])
        writer.writeheader()
        writer.writerows(all_probe_rows)

    # Write eval_report.md
    md_path = out_path / "eval_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        f.write(f"- **Checkpoint:** `{checkpoint}`\n")
        f.write(f"- **Macro Reward:** **{results['macro_reward']}**\n")
        f.write(f"- **Min Stage Reward:** **{results['min_stage_reward']}**\n")
        f.write(f"- **Local Evaluator Fast-Pass Fraction:** {router.telemetry['local_fraction'] * 100:.1f}%\n")
        f.write(f"- **Estimated Time Saved:** {router.telemetry['estimated_time_saved_s']} s\n\n")

        f.write("## Per-Stage Performance\n\n")
        f.write("| Stage | Probes | Mean Reward | Pass Rate |\n")
        f.write("|---|---|---|---|\n")
        for st, data in results["stages_evaluated"].items():
            f.write(f"| Stage {st} | {data['probe_count']} | {data['mean_reward']} | {data['pass_rate'] * 100:.1f}% |\n")

        if "comparison" in results:
            cmp = results["comparison"]
            f.write(f"\n## Comparison with `{cmp['checkpoint_b']}`\n\n")
            f.write(f"- **Macro Delta:** {cmp['delta']:+0.4f}\n")
            for st, d in cmp["stage_deltas"].items():
                f.write(f"- Stage {st} Delta: {d:+0.4f}\n")

    logger.info("Evaluation reports successfully generated in: %s", out_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint against held-out probe banks")
    parser.add_argument("--stage", type=str, default="all", help="Curriculum stage (1-5 or 'all')")
    parser.add_argument("--checkpoint", type=str, default="trainer/checkpoints/checkpoint_best", help="Checkpoint path")
    parser.add_argument("--compare", type=str, default=None, help="Optional second checkpoint path to compare")
    parser.add_argument("--probe-dir", type=str, default="data/probes", help="Path to held-out probes")
    parser.add_argument("--output-dir", type=str, default="logs/eval_reports", help="Output directory for reports")
    args = parser.parse_args()

    results = run_evaluation(
        stage=args.stage,
        checkpoint=args.checkpoint,
        probe_dir=args.probe_dir,
        output_dir=args.output_dir,
        compare_checkpoint=args.compare,
    )
    print(f"\n[OK] Evaluation completed. Macro Reward: {results['macro_reward']:.4f}")


if __name__ == "__main__":
    main()
