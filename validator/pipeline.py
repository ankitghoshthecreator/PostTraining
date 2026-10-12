"""validator/pipeline.py — Task 2.8: 7-layer validation pipeline with thread-safety and reject logging."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from validator.layers import (
    check_l1_schema,
    check_l2_min_length,
    check_l3_max_length,
    check_l4_repetition,
    check_l5_identity,
    check_l6_circularity,
    check_l7_dedup,
)


class QualityGatePipeline:
    """Thread-safe 7-layer quality gate pipeline for filtering synthetic training data."""

    def __init__(self, reject_log_path: Optional[str | Path] = None):
        self.seen_hashes: Set[str] = set()
        self.lock = threading.Lock()
        self.reject_log_path = Path(reject_log_path) if reject_log_path else None
        if self.reject_log_path:
            self.reject_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.stats: Dict[str, int] = {
            "total_evaluated": 0,
            "passed": 0,
            "L1_schema": 0,
            "L2_min_length": 0,
            "L3_max_length": 0,
            "L4_repetition": 0,
            "L5_identity": 0,
            "L6_circularity": 0,
            "L7_duplicate": 0,
        }

    def validate_sample(self, record: dict) -> Tuple[bool, Optional[str]]:
        """Validate a single record against layers L1 through L7 in order of computational cost."""
        with self.lock:
            self.stats["total_evaluated"] += 1

        # L1: Schema check
        err = check_l1_schema(record)
        if err:
            self._record_reject("L1_schema", record, err)
            return False, err

        # L2: Min length
        err = check_l2_min_length(record)
        if err:
            self._record_reject("L2_min_length", record, err)
            return False, err

        # L3: Max length
        err = check_l3_max_length(record)
        if err:
            self._record_reject("L3_max_length", record, err)
            return False, err

        # L4: Bigram repetition
        err = check_l4_repetition(record)
        if err:
            self._record_reject("L4_repetition", record, err)
            return False, err

        # L5: Identity / echo
        err = check_l5_identity(record)
        if err:
            self._record_reject("L5_identity", record, err)
            return False, err

        # L6: Circularity
        err = check_l6_circularity(record)
        if err:
            self._record_reject("L6_circularity", record, err)
            return False, err

        # L7: Dedup (requires lock for thread safety)
        with self.lock:
            err = check_l7_dedup(record, self.seen_hashes)
            if err:
                self.stats["L7_duplicate"] += 1
                self._log_reject_to_disk(record, err)
                return False, err

            self.stats["passed"] += 1

        return True, None

    def _record_reject(self, layer_key: str, record: dict, reason: str):
        with self.lock:
            self.stats[layer_key] += 1
        self._log_reject_to_disk(record, reason)

    def _log_reject_to_disk(self, record: dict, reason: str):
        if not self.reject_log_path:
            return
        entry = dict(record)
        entry["reject_reason"] = reason
        with self.lock:
            with open(self.reject_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def filter_batch(self, batch: List[dict]) -> List[dict]:
        """Filter a list of candidates and return only passing samples."""
        passed = []
        for sample in batch:
            ok, _ = self.validate_sample(sample)
            if ok:
                passed.append(sample)
        return passed
