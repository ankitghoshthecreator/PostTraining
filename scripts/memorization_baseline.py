"""scripts/memorization_baseline.py

Task 1.4 — Reproduce the memorization failure (on purpose).
This is a research artifact demonstrating the catastrophic failure of naive
unmasked SFT (training loss collapses to near-zero, model memorizes targets
verbatim without learning generalization).

Trains naive mixed SFT (no prompt masking, no replay) on the 35 samples for
a high number of steps, logs loss curve to CSV, runs greedy probe generation,
and saves the loss collapse plot to docs/figures/memorization_baseline.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainer.dataset import load_jsonl, PROMPT_SEPARATOR


def build_naive_example(prompt: str, target: str, tokenizer, max_len: int = 512):
    """Tokenize prompt + target WITHOUT masking prompt labels (naive SFT)."""
    full_text = prompt + PROMPT_SEPARATOR + target + tokenizer.eos_token
    encoded = tokenizer(
        full_text,
        max_length=max_len,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    # Naive SFT: calculate loss on ALL tokens (including prompt!)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def evaluate_exact_match(
    model, tokenizer, dataset: List[Dict], device: torch.device
) -> float:
    """Greedy decode training prompts and compute exact match rate on target."""
    model.eval()
    matches = 0
    total = len(dataset)

    with torch.no_grad():
        for item in dataset:
            prompt_text = item.prompt + PROMPT_SEPARATOR
            target_text = item.target.strip()

            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=len(tokenizer(target_text)["input_ids"]) + 10,
                do_sample=False,  # greedy
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            # Slice generated output beyond the prompt
            gen_tokens = output_tokens[0][inputs["input_ids"].shape[1] :]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            if gen_text == target_text:
                matches += 1

    return matches / max(total, 1)


def plot_loss_curve(log_file: Path, output_image: Path):
    """Plot the training loss curve and save figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed; skipping plot generation.")
        return

    steps, losses = [], []
    with open(log_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))

    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, label="Naive SFT (Unmasked, No Replay)", color="#e63946")
    plt.xlabel("Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Memorization Collapse: Naive SFT Overfitting on 35 Samples")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"[OK] Saved loss curve plot to {output_image}")


def main():
    parser = argparse.ArgumentParser(description="Run Task 1.4 memorization baseline.")
    parser.add_argument("--model", type=str, default="gpt2", help="Model identifier")
    parser.add_argument("--steps", type=int, default=150, help="Number of training steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="logs/memorization_baseline")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_csv = out_dir / "loss.csv"
    fig_path = Path("docs/figures/memorization_baseline.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    # Load 35 training samples
    all_samples: List[Dict] = []
    for stage_num in range(1, 6):
        pattern = f"stage_{stage_num}_*.jsonl"
        for p in Path("data/curriculum").glob(pattern):
            all_samples.extend(load_jsonl(p))
    print(f"[*] Loaded {len(all_samples)} total curriculum samples.")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    # Pre-tokenize naive batch
    batches = [build_naive_example(s.prompt, s.target, tokenizer) for s in all_samples]

    optimizer = AdamW(model.parameters(), lr=args.lr)
    model.train()

    print(f"[*] Training for {args.steps} steps...")
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "exact_match_rate", "time_s"])

        start_time = time.time()
        for step in range(1, args.steps + 1):
            # Pick a sample
            input_ids, attention_mask, labels = batches[(step - 1) % len(batches)]
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if step % 25 == 0 or step == args.steps:
                em = evaluate_exact_match(model, tokenizer, all_samples, device)
                model.train()
                elapsed = round(time.time() - start_time, 2)
                writer.writerow([step, round(loss.item(), 5), round(em, 4), elapsed])
                f.flush()
                print(f"Step {step:4d}/{args.steps} | Loss: {loss.item():.4f} | Exact Match: {em*100:.1f}%")

    plot_loss_curve(log_csv, fig_path)
    print(f"[DONE] Baseline complete. Results written to {log_csv}")


if __name__ == "__main__":
    main()
