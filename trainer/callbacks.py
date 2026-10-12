"""
trainer/callbacks.py — Per-step CSV logging, best-checkpoint tracker,
                        early stopping, and run metadata writer.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------


class CSVLogger:
    """
    Writes per-step training metrics to a CSV file.

    Columns: step, epoch, stage, lr, loss, grad_norm, tokens_seen,
             gpu_mem_mb, wall_time_s
    """

    TRAIN_COLS = [
        "step", "epoch", "stage", "lr", "loss",
        "grad_norm", "tokens_seen", "gpu_mem_mb", "wall_time_s",
    ]
    EVAL_COLS = [
        "step", "stage", "probe_reward", "wall_time_s",
    ]

    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self._train_path = log_dir / "train_steps.csv"
        self._eval_path = log_dir / "eval_steps.csv"
        self._start_time = time.time()
        self._train_writer: Optional[csv.DictWriter] = None
        self._eval_writer: Optional[csv.DictWriter] = None
        self._train_fh = None
        self._eval_fh = None
        self._open_files()

    def _open_files(self) -> None:
        self._train_fh = open(self._train_path, "w", newline="", encoding="utf-8")
        self._eval_fh = open(self._eval_path, "w", newline="", encoding="utf-8")

        self._train_writer = csv.DictWriter(self._train_fh, fieldnames=self.TRAIN_COLS)
        self._eval_writer = csv.DictWriter(self._eval_fh, fieldnames=self.EVAL_COLS)

        self._train_writer.writeheader()
        self._eval_writer.writeheader()

    def log_train_step(
        self,
        step: int,
        epoch: int,
        stage: int,
        lr: float,
        loss: float,
        grad_norm: float,
        tokens_seen: int,
    ) -> None:
        gpu_mem = 0.0
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

        row = {
            "step": step,
            "epoch": epoch,
            "stage": stage,
            "lr": f"{lr:.6e}",
            "loss": f"{loss:.6f}",
            "grad_norm": f"{grad_norm:.4f}",
            "tokens_seen": tokens_seen,
            "gpu_mem_mb": f"{gpu_mem:.1f}",
            "wall_time_s": f"{time.time() - self._start_time:.2f}",
        }
        self._train_writer.writerow(row)
        self._train_fh.flush()

    def log_eval_step(self, step: int, stage: int, probe_reward: float) -> None:
        row = {
            "step": step,
            "stage": stage,
            "probe_reward": f"{probe_reward:.6f}",
            "wall_time_s": f"{time.time() - self._start_time:.2f}",
        }
        self._eval_writer.writerow(row)
        self._eval_fh.flush()

    def close(self) -> None:
        if self._train_fh:
            self._train_fh.close()
        if self._eval_fh:
            self._eval_fh.close()

    def __del__(self) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Checkpoint tracker
# ---------------------------------------------------------------------------


class CheckpointTracker:
    """
    Saves model/adapter weights when held-out probe reward improves.
    Also maintains checkpoint_last/ for resumption.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        model,
        tokenizer,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.best_dir = checkpoint_dir / "checkpoint_best"
        self.last_dir = checkpoint_dir / "checkpoint_last"
        self.model = model
        self.tokenizer = tokenizer
        self.best_reward: float = -1.0
        self._step: int = 0

    def step(self, step: int, probe_reward: float) -> bool:
        """
        Call after each eval.  Saves best checkpoint if probe_reward
        improved.  Always saves last checkpoint.

        Returns True if a new best was saved.
        """
        self._step = step
        self._save(self.last_dir)
        logger.debug("Saved last checkpoint at step %d (reward=%.4f)", step, probe_reward)

        if probe_reward > self.best_reward:
            self.best_reward = probe_reward
            self._save(self.best_dir)
            logger.info(
                "New best checkpoint at step %d — reward %.4f → saved to %s",
                step, probe_reward, self.best_dir,
            )
            return True
        return False

    def _save(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        # Save only LoRA adapter weights (not the full frozen base model).
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(dest)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(dest)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopping:
    """
    Stops training when probe_reward has not improved for `patience` evaluations.
    """
    patience: int = 3
    _no_improve_count: int = field(default=0, init=False)
    _best_reward: float = field(default=-1.0, init=False)

    def update(self, reward: float) -> bool:
        """Returns True if training should stop."""
        if reward > self._best_reward:
            self._best_reward = reward
            self._no_improve_count = 0
            return False
        self._no_improve_count += 1
        if self._no_improve_count >= self.patience:
            logger.info(
                "Early stopping triggered: no improvement for %d evals (best=%.4f).",
                self._no_improve_count, self._best_reward,
            )
            return True
        return False

    def reset(self) -> None:
        self._no_improve_count = 0
        self._best_reward = -1.0


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


def write_run_metadata(
    log_dir: Path,
    config_path: str,
    model_name: str,
    seed: int,
    dataset_hashes: dict[str, str],
    git_sha: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> None:
    """
    Write run.json with reproducibility metadata.
    """
    import hashlib
    import subprocess

    if git_sha is None:
        try:
            git_sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            git_sha = "unknown"

    if config_hash is None:
        try:
            config_hash = hashlib.md5(Path(config_path).read_bytes()).hexdigest()
        except Exception:
            config_hash = "unknown"

    # Library versions
    versions: dict[str, str] = {}
    for lib in ["torch", "transformers", "peft", "bitsandbytes"]:
        try:
            import importlib
            mod = importlib.import_module(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not_installed"

    metadata = {
        "git_sha": git_sha,
        "config_path": config_path,
        "config_hash": config_hash,
        "model_name": model_name,
        "seed": seed,
        "dataset_hashes": dataset_hashes,
        "library_versions": versions,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = log_dir / "run.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Run metadata written to %s", out_path)
