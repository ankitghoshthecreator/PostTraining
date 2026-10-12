"""
trainer/replay_buffer.py — In-memory replay buffer for curriculum SFT.

Stores pre-tokenized Sample objects per stage and blends earlier-stage
samples into the current stage's training set at a configured ratio.

Performance target: <0.05 s per stage transition.
"""

from __future__ import annotations

import random
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainer.dataset import Sample

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    In-memory replay buffer for multi-stage curriculum training.

    Samples are stored per stage and blended into later stages to
    prevent catastrophic forgetting of earlier skills.

    Ratio semantics:
        replay_ratio is the fraction of the *final mixed set* that will
        come from the replay pool.  E.g. ratio=0.25 on 100 current
        samples → 25 replay samples added → 25/(100+25)=0.20 of final set.

        More precisely: n_replay = round(n_current * ratio / (1 - ratio))
    """

    def __init__(self, seed: int = 42) -> None:
        self.by_stage: dict[int, list] = {}  # stage → list[Sample]
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_stage(self, stage: int, samples: list) -> None:
        """Register a stage's samples into the buffer.

        Args:
            stage:   Stage id (1-5).
            samples: List of Sample objects.
        """
        if not samples:
            logger.warning("ReplayBuffer.add_stage(%d): received empty sample list.", stage)
        self.by_stage[stage] = list(samples)
        logger.debug("ReplayBuffer: stored %d samples for stage %d.", len(samples), stage)

    # ------------------------------------------------------------------
    # Blending
    # ------------------------------------------------------------------

    def blend(
        self,
        current_stage: int,
        current_samples: list,
        ratio: float,
    ) -> list:
        """
        Blend replay samples from stages < current_stage into current_samples.

        Args:
            current_stage:   The stage being trained.
            current_samples: Training samples for the current stage.
            ratio:           Target replay fraction of the *final* mixed set.
                             Must be in [0, 1).  0 → no replay.

        Returns:
            Shuffled list of current + replay samples.
        """
        if ratio <= 0.0:
            return list(current_samples)

        if ratio >= 1.0:
            raise ValueError(f"replay_ratio must be < 1.0, got {ratio}")

        pool = [
            s
            for stage_id, stage_samples in self.by_stage.items()
            if stage_id < current_stage
            for s in stage_samples
        ]

        if not pool:
            logger.debug(
                "ReplayBuffer.blend(stage=%d): pool is empty (no earlier stages), skipping replay.",
                current_stage,
            )
            return list(current_samples)

        n_current = len(current_samples)
        n_replay = round(n_current * ratio / (1.0 - ratio))
        n_replay = min(n_replay, len(pool))

        # Sample evenly across earlier stages to avoid stage-1 dominance.
        replay = self._sample_evenly(current_stage, n_replay)

        mixed = list(current_samples) + replay
        self.rng.shuffle(mixed)

        logger.debug(
            "ReplayBuffer.blend(stage=%d, ratio=%.2f): %d current + %d replay = %d total.",
            current_stage,
            ratio,
            n_current,
            len(replay),
            len(mixed),
        )
        return mixed

    def _sample_evenly(self, current_stage: int, n_total: int) -> list:
        """Sample ~equally from each stage < current_stage."""
        earlier_stages = sorted(k for k in self.by_stage if k < current_stage)
        if not earlier_stages:
            return []

        per_stage = max(1, n_total // len(earlier_stages))
        sampled: list = []

        for stage_id in earlier_stages:
            pool = self.by_stage[stage_id]
            k = min(per_stage, len(pool))
            sampled.extend(self.rng.sample(pool, k))

        # Trim or top-up to exactly n_total.
        if len(sampled) > n_total:
            sampled = self.rng.sample(sampled, n_total)
        elif len(sampled) < n_total:
            # Pool of all earlier samples.
            full_pool = [
                s
                for sid in earlier_stages
                for s in self.by_stage[sid]
                if s not in sampled
            ]
            extra = min(n_total - len(sampled), len(full_pool))
            if extra > 0:
                sampled.extend(self.rng.sample(full_pool, extra))

        return sampled

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[int, int]:
        """Return {stage: sample_count} for all stored stages."""
        return {k: len(v) for k, v in sorted(self.by_stage.items())}

    def total_samples(self) -> int:
        return sum(len(v) for v in self.by_stage.values())
