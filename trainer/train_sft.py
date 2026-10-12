"""
trainer/train_sft.py — 4-bit NF4 QLoRA training loop with curriculum + replay.

Entry point:
    python trainer/train_sft.py --config configs/curriculum.yaml \
                                 --model-config configs/model.yaml \
                                 [--stages 1 5] [--resume]

This module covers Tasks 1.5, 1.6, and 1.7 from phase.md.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import sys
import time
from pathlib import Path

# Ensure repo root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("train_sft")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_hash(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def log_vram(label: str) -> None:
    if torch.cuda.is_available():
        used = torch.cuda.max_memory_allocated() / (1024 ** 2)
        logger.info("VRAM [%s]: %.0f MB peak", label, used)


def check_oom_guard(threshold_mb: float = 3800.0) -> None:
    """Abort with a clear message if we are about to hit OOM."""
    if torch.cuda.is_available():
        used = torch.cuda.max_memory_allocated() / (1024 ** 2)
        if used > threshold_mb:
            raise MemoryError(
                f"VRAM usage {used:.0f} MB exceeds abort threshold {threshold_mb:.0f} MB. "
                "Reduce batch size, max_seq_len, or switch to a smaller model."
            )


# ---------------------------------------------------------------------------
# Simple probe scorer for early stopping (no 70B teacher needed for Phase 1)
# ---------------------------------------------------------------------------

def score_probe_response(generated: str, reference: str, task: str) -> float:
    """Lightweight local scorer returning a float in [0, 1]."""
    gen = generated.strip()
    ref = reference.strip()
    if not gen:
        return 0.0

    if task in ("grammar", "punctuation"):
        # Exact match after normalisation
        if gen.lower() == ref.lower():
            return 1.0
        # Edit distance as proxy
        from difflib import SequenceMatcher
        return SequenceMatcher(None, gen.lower(), ref.lower()).ratio()

    if task == "summarization":
        # ROUGE-L style: LCS / len(ref)
        gen_tokens = gen.lower().split()
        ref_tokens = ref.lower().split()
        if not ref_tokens:
            return 0.0
        from difflib import SequenceMatcher
        m = SequenceMatcher(None, gen_tokens, ref_tokens).find_longest_match(
            0, len(gen_tokens), 0, len(ref_tokens)
        )
        recall = m.size / len(ref_tokens)
        precision = m.size / len(gen_tokens) if gen_tokens else 0.0
        if recall + precision == 0:
            return 0.0
        return 2 * recall * precision / (recall + precision)

    # English / conversation — Jaccard similarity as fallback
    gen_set = set(gen.lower().split())
    ref_set = set(ref.lower().split())
    if not ref_set:
        return 0.0
    return len(gen_set & ref_set) / len(gen_set | ref_set)


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # -----------------------------------------------------------------------
    # Imports that require torch/transformers (allow import of the module
    # without them for schema tests)
    # -----------------------------------------------------------------------
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        get_linear_schedule_with_warmup,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from config import load, CurriculumConfig, ModelConfig
    from trainer.dataset import SFTDataset, load_jsonl
    from trainer.replay_buffer import ReplayBuffer
    from trainer.callbacks import (
        CSVLogger,
        CheckpointTracker,
        EarlyStopping,
        write_run_metadata,
    )

    # ---- Config -----------------------------------------------------------
    curriculum_cfg: CurriculumConfig = load(args.config)
    model_cfg: ModelConfig = load(args.model_config)

    seed = model_cfg.training.seed
    set_seed(seed)

    # ---- Run directory ----------------------------------------------------
    run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path("logs") / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path("trainer") / "checkpoints"

    csv_log = CSVLogger(log_dir)
    early_stopper = EarlyStopping(patience=curriculum_cfg.early_stop.patience)

    # ---- Tokenizer --------------------------------------------------------
    model_name = model_cfg.model_name
    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # GPT-2 has no pad token; set it to eos but we'll mask those positions.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token (%s)", tokenizer.eos_token)

    # ---- Quantization + model loading ------------------------------------
    logger.info("Loading model with 4-bit NF4 quantization…")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=model_cfg.quantization.load_in_4bit,
        bnb_4bit_quant_type=model_cfg.quantization.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_cfg.quantization.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=torch.float16
        if model_cfg.quantization.bnb_4bit_compute_dtype == "float16"
        else torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    log_vram("after model load")

    # ---- LoRA -------------------------------------------------------------
    lora_cfg = LoraConfig(
        r=model_cfg.lora.r,
        lora_alpha=model_cfg.lora.lora_alpha,
        lora_dropout=model_cfg.lora.lora_dropout,
        task_type=model_cfg.lora.task_type,
        target_modules=model_cfg.lora.target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    log_vram("after LoRA wrapping")

    # ---- Checkpoint tracker -----------------------------------------------
    ckpt_tracker = CheckpointTracker(checkpoint_dir, model, tokenizer)

    # ---- Determine active stages ------------------------------------------
    stage_range = range(args.stages[0], args.stages[1] + 1)
    data_dir = Path("data") / "curriculum"
    probe_dir = Path("data") / "probes"

    # ---- Dataset hashes for run metadata ---------------------------------
    dataset_hashes: dict[str, str] = {}
    for stage_cfg in curriculum_cfg.stages:
        p = data_dir / stage_cfg.file
        if p.exists():
            dataset_hashes[stage_cfg.file] = file_hash(p)

    write_run_metadata(
        log_dir,
        config_path=str(args.config),
        model_name=model_name,
        seed=seed,
        dataset_hashes=dataset_hashes,
    )

    # ---- Replay buffer ---------------------------------------------------
    replay_buf = ReplayBuffer(seed=seed)

    # ---- Optimizer -------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=model_cfg.training.learning_rate,
        weight_decay=model_cfg.training.weight_decay,
    )

    global_step = 0
    total_tokens_seen = 0

    # ======================================================================
    # Main curriculum loop
    # ======================================================================
    for stage_cfg in curriculum_cfg.stages:
        stage_id = stage_cfg.id
        if stage_id not in stage_range:
            continue

        logger.info("=" * 60)
        logger.info("Starting Stage %d | file=%s | epochs=%d | replay=%.2f",
                    stage_id, stage_cfg.file, stage_cfg.epochs, stage_cfg.replay_ratio)

        # ---- Load stage data --------------------------------------------
        stage_path = data_dir / stage_cfg.file
        raw_samples = load_jsonl(stage_path, expected_stage=stage_id)

        # ---- Blend replay ------------------------------------------------
        blended = replay_buf.blend(stage_id, raw_samples, stage_cfg.replay_ratio)
        logger.info("Stage %d blended set: %d samples", stage_id, len(blended))

        # ---- Build dataset -----------------------------------------------
        dataset = SFTDataset(blended, tokenizer, max_len=model_cfg.training.max_seq_len)
        collate = dataset.get_collate_fn()
        loader = DataLoader(
            dataset,
            batch_size=model_cfg.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
        )

        if len(dataset) == 0:
            logger.warning("Stage %d: dataset is empty after tokenization. Skipping.", stage_id)
            continue

        # ---- LR scheduler ------------------------------------------------
        accum_steps = model_cfg.training.gradient_accumulation_steps
        total_steps = stage_cfg.epochs * len(loader) // accum_steps
        total_steps = min(total_steps, curriculum_cfg.max_steps_per_stage)
        warmup_steps = max(1, int(total_steps * model_cfg.training.warmup_ratio))

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ---- Training epochs --------------------------------------------
        model.train()
        stage_step = 0
        early_stopper.reset()

        for epoch in range(1, stage_cfg.epochs + 1):
            epoch_loss = 0.0
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(loader):
                if stage_step >= curriculum_cfg.max_steps_per_stage:
                    logger.info("Stage %d: max_steps_per_stage=%d reached.",
                                stage_id, curriculum_cfg.max_steps_per_stage)
                    break

                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / accum_steps
                loss.backward()

                total_tokens_seen += int(attention_mask.sum().item())

                if (batch_idx + 1) % accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), model_cfg.training.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    stage_step += 1
                    global_step += 1

                    current_lr = scheduler.get_last_lr()[0]
                    csv_log.log_train_step(
                        step=global_step,
                        epoch=epoch,
                        stage=stage_id,
                        lr=current_lr,
                        loss=float(outputs.loss),
                        grad_norm=float(grad_norm),
                        tokens_seen=total_tokens_seen,
                    )
                    epoch_loss += float(outputs.loss)

                    if global_step % 50 == 0:
                        logger.info(
                            "Step %d | Stage %d | Epoch %d | Loss %.4f | LR %.2e",
                            global_step, stage_id, epoch, float(outputs.loss), current_lr,
                        )
                        log_vram(f"step-{global_step}")
                        check_oom_guard()

            # ---- Eval probe bank after each epoch -----------------------
            probe_file = probe_dir / f"stage_{stage_id}.jsonl"
            probe_reward = 0.0
            if probe_file.exists():
                probe_reward = _eval_probes(model, tokenizer, probe_file, stage_id, model_cfg)
                csv_log.log_eval_step(global_step, stage_id, probe_reward)
                logger.info(
                    "Stage %d | Epoch %d | Probe reward: %.4f",
                    stage_id, epoch, probe_reward,
                )

                # Best checkpoint
                ckpt_tracker.step(global_step, probe_reward)

                # Early stopping
                if early_stopper.update(probe_reward):
                    logger.info("Early stopping at stage %d, epoch %d.", stage_id, epoch)
                    break
            else:
                logger.warning("No probe file found for stage %d.", stage_id)

        # ---- Add this stage to replay buffer after training it -----------
        replay_buf.add_stage(stage_id, raw_samples)
        log_vram(f"end-of-stage-{stage_id}")

    csv_log.close()
    logger.info("Training complete. Best checkpoint reward: %.4f", ckpt_tracker.best_reward)


# ---------------------------------------------------------------------------
# Probe evaluation (Phase-1 lightweight scorer)
# ---------------------------------------------------------------------------

def _eval_probes(model, tokenizer, probe_file: Path, stage_id: int, model_cfg) -> float:
    """Generate responses for each probe and return mean reward."""
    import json

    probes = []
    with open(probe_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(json.loads(line))

    if not probes:
        return 0.0

    model.eval()
    rewards = []
    with torch.no_grad():
        for probe in probes:
            prompt_text = probe["prompt"] + "\n### Response:\n"
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=model_cfg.training.max_seq_len // 2,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            out_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            # Decode only generated tokens
            gen_ids = out_ids[0][inputs["input_ids"].shape[-1]:]
            generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            reference = probe.get("reference", "")
            task = probe.get("task", "english")
            score = score_probe_response(generated, reference, task)
            rewards.append(score)

    model.train()
    return float(sum(rewards) / len(rewards)) if rewards else 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA curriculum SFT trainer")
    p.add_argument("--config", default="configs/curriculum.yaml", help="Curriculum YAML config")
    p.add_argument("--model-config", default="configs/model.yaml", help="Model YAML config")
    p.add_argument(
        "--stages", nargs=2, type=int, default=[1, 5], metavar=("START", "END"),
        help="Stage range to train (inclusive). Default: 1 5",
    )
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint_last/")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
