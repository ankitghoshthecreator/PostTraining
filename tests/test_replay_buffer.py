"""
tests/test_replay_buffer.py — Replay buffer correctness tests.

Covers Task 1.6 acceptance criteria from phase.md:
- ratio math
- empty pool at stage 1
- determinism with seed
- no stage-k samples duplicated into replay
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from dataclasses import dataclass
from trainer.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------------
# Minimal sample stub
# ---------------------------------------------------------------------------

@dataclass
class FakeSample:
    id: str
    stage: int
    prompt: str = ""
    target: str = ""


def make_samples(stage: int, n: int) -> list[FakeSample]:
    return [FakeSample(id=f"s{stage}-{i:04d}", stage=stage) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReplayBuffer:

    def test_empty_pool_at_stage_1(self):
        """Stage 1 has no earlier stages — blend should return current samples unchanged."""
        buf = ReplayBuffer(seed=0)
        samples = make_samples(1, 10)
        result = buf.blend(current_stage=1, current_samples=samples, ratio=0.3)
        assert len(result) == 10

    def test_ratio_math(self):
        """With ratio=0.25 and n_current=100, replay count ≈ 33 → total ≈ 133."""
        buf = ReplayBuffer(seed=0)
        buf.add_stage(1, make_samples(1, 200))
        current = make_samples(2, 100)
        result = buf.blend(current_stage=2, current_samples=current, ratio=0.25)
        n_replay = len(result) - 100
        # n_replay = round(100 * 0.25 / 0.75) = round(33.33) = 33
        assert n_replay == 33, f"Expected 33 replay samples, got {n_replay}"
        assert len(result) == 133

    def test_zero_ratio_returns_current_only(self):
        """ratio=0.0 → no replay, return current samples."""
        buf = ReplayBuffer(seed=0)
        buf.add_stage(1, make_samples(1, 50))
        current = make_samples(2, 20)
        result = buf.blend(current_stage=2, current_samples=current, ratio=0.0)
        assert len(result) == 20

    def test_determinism(self):
        """Same seed → same blend order."""
        def run_blend(seed):
            buf = ReplayBuffer(seed=seed)
            buf.add_stage(1, make_samples(1, 50))
            current = make_samples(2, 20)
            return [s.id for s in buf.blend(2, current, ratio=0.3)]

        assert run_blend(42) == run_blend(42)

    def test_different_seeds_differ(self):
        """Different seeds → (almost certainly) different order."""
        def run_blend(seed):
            buf = ReplayBuffer(seed=seed)
            buf.add_stage(1, make_samples(1, 50))
            current = make_samples(2, 20)
            return [s.id for s in buf.blend(2, current, ratio=0.3)]

        # With 50 samples it's astronomically unlikely both orderings are identical.
        assert run_blend(1) != run_blend(99)

    def test_no_stage_k_samples_in_replay(self):
        """Replay must NOT include samples from current_stage."""
        buf = ReplayBuffer(seed=0)
        stage1_samples = make_samples(1, 30)
        stage2_samples = make_samples(2, 30)
        buf.add_stage(1, stage1_samples)
        buf.add_stage(2, stage2_samples)  # Simulate: stage 2 was already trained

        current = make_samples(3, 20)
        result = buf.blend(current_stage=3, current_samples=current, ratio=0.3)

        for s in result:
            assert s.stage != 3 or s in current, (
                f"Stage-3 sample {s.id} found in replay (should only be in current)"
            )

    def test_replay_capped_by_pool_size(self):
        """Replay count is capped by actual pool size."""
        buf = ReplayBuffer(seed=0)
        buf.add_stage(1, make_samples(1, 3))  # very small pool
        current = make_samples(2, 100)
        # ratio=0.5 would request 100 replay samples but pool only has 3
        result = buf.blend(current_stage=2, current_samples=current, ratio=0.5)
        n_replay = len(result) - 100
        assert n_replay <= 3

    def test_stats(self):
        """stats() returns correct per-stage counts."""
        buf = ReplayBuffer(seed=0)
        buf.add_stage(1, make_samples(1, 7))
        buf.add_stage(2, make_samples(2, 5))
        assert buf.stats() == {1: 7, 2: 5}
        assert buf.total_samples() == 12

    def test_ratio_must_be_less_than_1(self):
        """ratio >= 1.0 should raise ValueError."""
        buf = ReplayBuffer(seed=0)
        buf.add_stage(1, make_samples(1, 10))
        with pytest.raises(ValueError, match="replay_ratio must be < 1.0"):
            buf.blend(current_stage=2, current_samples=make_samples(2, 10), ratio=1.0)
