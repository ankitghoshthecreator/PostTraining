"""scripts/run_flywheel_iteration.py — Task 2.9 & 2.10: CLI runner for RLAIF flywheel iterations.

Usage:
    python scripts/run_flywheel_iteration.py --iterations 3 --model gpt2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlaif.controller import LoopController
from rlaif.probe import run_probes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("flywheel_runner")


def main():
    parser = argparse.ArgumentParser(description="Run RLAIF closed-loop iteration")
    parser.add_argument("--iterations", type=int, default=1, help="Number of flywheel iterations to run")
    parser.add_argument("--model", type=str, default="gpt2", help="Model checkpoint name or path")
    args = parser.parse_args()

    controller = LoopController()

    for i in range(1, args.iterations + 1):
        logger.info("=== Running Flywheel Cycle %d/%d ===", i, args.iterations)
        # 1. Run probes on student model
        probe_results = run_probes(
            model_path=args.model,
            probe_dir="data/probes",
            output_dir=f"logs/run_flywheel/iter_{i}",
            stages=[1, 2, 3, 4, 5],
            max_new_tokens=48,
        )

        # 2. Execute full closed loop
        report = controller.run_iteration(
            iteration_index=i,
            probe_results_by_stage=probe_results,
            sample_budget=10,
        )
        print(f"\n[OK] Iteration {i} finished: {report['accepted_samples']} synthetic samples accepted into {report['dataset_file']}")


if __name__ == "__main__":
    main()
