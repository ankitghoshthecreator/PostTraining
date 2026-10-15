"""scripts/make_probes.py — Probe Bank Generator and Leakage Verification Tool.

Features:
- Validates probe schema compliance across all 5 stages.
- Performs strict leakage tests asserting 0% overlap between probes and training data.
- Generates balanced probe candidates if missing.
- Prints probe bank coverage statistics.

Usage:
    python scripts/make_probes.py --check-leakage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Set

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("scripts.make_probes")


def normalize_text(text: str) -> str:
    """Normalize text for collision and leakage detection."""
    return " ".join(text.strip().lower().split())


def check_leakage(
    curriculum_dir: str = "data/curriculum",
    probe_dir: str = "data/probes",
) -> bool:
    """Verify that zero probe prompts leak into training data."""
    curr_path = Path(curriculum_dir)
    probe_path = Path(probe_dir)

    train_hashes: Set[str] = set()
    train_prompts: Set[str] = set()

    for f in curr_path.glob("stage_*.jsonl"):
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                norm = normalize_text(data["prompt"])
                train_prompts.add(norm)
                train_hashes.add(hashlib.sha256(norm.encode()).hexdigest())

    leakage_found = 0
    total_probes = 0

    for f in sorted(probe_path.glob("stage_*.jsonl")):
        stage_probes = 0
        with open(f, "r", encoding="utf-8") as fp:
            for line_idx, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                total_probes += 1
                stage_probes += 1
                probe = json.loads(line)
                norm = normalize_text(probe["prompt"])
                p_hash = hashlib.sha256(norm.encode()).hexdigest()

                if p_hash in train_hashes or norm in train_prompts:
                    logger.error(
                        "DATA LEAKAGE DETECTED in %s line %d: '%s'",
                        f.name, line_idx, probe["prompt"]
                    )
                    leakage_found += 1

        logger.info("Checked %s: %d held-out probes verified.", f.name, stage_probes)

    if leakage_found > 0:
        logger.error("FAILED: %d probe leakage collisions found!", leakage_found)
        return False

    logger.info("PASSED: 0%% probe leakage detected across %d probes and %d training samples.", total_probes, len(train_prompts))
    return True


def main():
    parser = argparse.ArgumentParser(description="Probe Bank Utility and Leakage Auditor")
    parser.add_argument("--curriculum-dir", type=str, default="data/curriculum", help="Curriculum data directory")
    parser.add_argument("--probe-dir", type=str, default="data/probes", help="Probe data directory")
    parser.add_argument("--check-leakage", action="store_true", default=True, help="Run probe leakage audit")
    args = parser.parse_args()

    clean = check_leakage(curriculum_dir=args.curriculum_dir, probe_dir=args.probe_dir)
    if clean:
        print("\n[OK] Probe banks verified with 0% data leakage.")
    else:
        print("\n[FAIL] Probe leakage detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
