"""rlaif/probe.py — Task 2.1: Probe runner for evaluating student checkpoints.

Loads student model checkpoint (4-bit base + LoRA adapter or standard base),
runs deterministic fixed-decoding generation over stage probe banks,
and logs probe outputs with latency.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("rlaif.probe")


def run_probes(
    model_path: str,
    probe_dir: str | Path = "data/probes",
    output_dir: str | Path = "logs/probes",
    stages: Optional[List[int]] = None,
    max_new_tokens: int = 128,
    device: Optional[str] = None,
) -> Dict[int, List[Dict]]:
    """Execute evaluation on probe banks and record outputs.

    Returns mapping of stage_id -> list of result dicts.
    """
    if stages is None:
        stages = [1, 2, 3, 4, 5]

    probe_path = Path(probe_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading model from %s on %s...", model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    results_by_stage: Dict[int, List[Dict]] = {}
    prompt_sep = "\n### Response:\n"

    try:
        for s in stages:
            probe_file = probe_path / f"stage_{s}.jsonl"
            if not probe_file.exists():
                logger.warning("Probe file not found: %s", probe_file)
                continue

            stage_results = []
            with open(probe_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    probe_item = json.loads(line)
                    probe_id = probe_item.get("id", f"p{s}-{len(stage_results)}")
                    prompt = probe_item["prompt"]
                    reference = probe_item.get("reference", probe_item.get("target", ""))
                    task = probe_item.get("task", "general")

                    input_text = prompt + prompt_sep
                    inputs = tokenizer(input_text, return_tensors="pt").to(device)

                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,  # deterministic greedy decoding
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    latency = round(time.perf_counter() - t0, 4)

                    gen_tokens = out[0][inputs["input_ids"].shape[1] :]
                    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                    res_record = {
                        "probe_id": probe_id,
                        "stage": s,
                        "task": task,
                        "prompt": prompt,
                        "reference": reference,
                        "response": response,
                        "latency_s": latency,
                    }
                    stage_results.append(res_record)

            results_by_stage[s] = stage_results
            out_file = out_path / f"probes_stage_{s}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for r in stage_results:
                    f.write(json.dumps(r) + "\n")
            logger.info("Stage %d: Evaluated %d probes -> %s", s, len(stage_results), out_file)

    finally:
        # Free memory cleanly
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results_by_stage
