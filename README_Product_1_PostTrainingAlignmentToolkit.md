# Post-Training Alignment Toolkit — SFT · RLAIF · Eval Harness for Open-Weight LLMs

> **Role target:** AI Research Engineer / AI Research Intern
> **Target companies:** Meta (Llama Post-Training) · Anthropic (Applied AI, Finetuning Engineer) · Google (AI Innovation & Research) · Cohere · Mistral

---

## TL;DR

A reproducible **post-training alignment pipeline** for open-weight LLMs combining **Supervised Fine-Tuning (SFT)**, a **closed-loop RLAIF data flywheel**, and an **automated evaluation harness** — all under 4GB VRAM. Built to answer the exact research question Meta's Llama post-training team works on: *can AI feedback reliably replace human annotation in a low-resource alignment setting?*

---

## Why This Project Exists

<cite index="75-1">Meta's Llama post-training team looks for interns who can develop state-of-the-art generative AI algorithms and data to more effectively align LLMs during post-training stages.</cite> <cite index="80-1">Anthropic's Applied AI Fine-tuning Engineer role requires driving adoption of frontier AI by developing bespoke fine-tuned LLM solutions.</cite> <cite index="74-1">Google's AI research internships focus on understanding limitations of ML architectures, trade-offs, and optimizations.</cite>

This project sits at the intersection of all three — practical post-training research on real models, under real constraints.

---

## Research Questions

1. Can **RLAIF** (AI feedback from a 70B teacher) replace human annotation for small-LLM alignment?
2. Does **curriculum learning** outperform mixed-dataset fine-tuning for multi-task conversational LLMs?
3. What **synthetic data quality filters** are necessary and sufficient to prevent data poisoning in a closed loop?
4. How well does a **lightweight local evaluator** (TTR, n-gram, Jaccard) proxy for a large teacher model's judgment?

---

## Pipeline Overview

```
Phase 1 — SFT (Supervised Fine-Tuning)
  ┌─────────────────────────────────────────────────────────────┐
  │ Curriculum Dataset (285+ samples, 5 stages)                 │
  │ → Prompt-Label Masking (labels[:prompt_len] = -100)         │
  │ → Replay Buffer Blending (stage-specific ratios)            │
  │ → 4-bit NF4 QLoRA (r=16, α=32, AdamW lr=2e-4)             │
  │ → Per-step CSV loss logging + best-checkpoint tracker       │
  └─────────────────────────────────────────────────────────────┘
              ↓
Phase 2 — RLAIF Closed Loop
  ┌─────────────────────────────────────────────────────────────┐
  │ Probe Model → Groq 70B Critique → Failure Taxonomy          │
  │ → Severity Gate (CRITICAL/HIGH only, priority ≤ 5)          │
  │ → Targeted Prompt → Ollama Generator → 7-Layer Validator    │
  │ → Enriched Dataset → SFT Retrain (loop)                     │
  └─────────────────────────────────────────────────────────────┘
              ↓
Phase 3 — Evaluation
  ┌─────────────────────────────────────────────────────────────┐
  │ Tiered Harness: Local (0.013ms/probe) → 70B Escalation      │
  │ Per-Stage Probe Bank → Reward Scalar [0,1]                  │
  │ Judge–Human Agreement Study (Cohen's κ = 0.77)              │
  │ 95 Automated Regression Tests (100% pass)                   │
  └─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Details |
|---|---|
| **Training** | PyTorch, Hugging Face Transformers, PEFT, bitsandbytes |
| **Models** | GPT-2 (117M), Qwen2.5-1.5B-Instruct, Phi-3-mini-4k (3.8B) |
| **Fine-tuning** | 4-bit NF4 QLoRA, LoRA adapters, AdamW, linear warmup |
| **Teacher Critic** | Groq API (llama-3-70b), structured JSON rubric, Pydantic v2 |
| **Generator** | Ollama (qwen2.5-coder:7b), local inference, instant failover |
| **Evaluation** | TTR, n-gram windowing, Jaccard, ROUGE-L, BERTScore |
| **Infra** | Docker multi-stage, Kubernetes (HPA 1→4), pytest (95 tests) |

---

## Key Contributions

### 1. Curriculum SFT with Replay Buffer
Discovered that naive mixed-dataset SFT caused catastrophic memorization: **~56K steps on 35 samples**, loss **8.78 → 0.0001**, model reproduced training data verbatim. Fixed with:
- **5-stage curriculum** (English → Summarization → Grammar → Punctuation → Conversation)
- In-memory **replay buffer** with stage-specific ratios (0.0–0.4)
- Stage-transition blending: **~4.5s → <0.05s (>90x)** vs disk I/O
- **Prompt-label masking**: `labels[:prompt_len] = -100` — recovers instruction-following without curriculum changes

### 2. RLAIF Data Flywheel
Closed-loop where AI feedback commissions its own corrective data:
- **Failure taxonomy** with 7 categories and severity enum (CRITICAL/HIGH/MEDIUM/LOW)
- **`<think>` CoT-tag sanitization** — removes false-positive detections from reasoning-model critics
- **Severity gating**: only CRITICAL/HIGH trigger generation, priority ≤ 5
- RLAIF verification run: **r = 1.000** across all 5 curriculum stages

### 3. Synthetic Data Quality Gate (7 layers)
Without quality gating, the loop poisons itself. Gate enforces:

| Layer | Filter | Threshold |
|---|---|---|
| L1–L3 | Schema, min/max length | — / 10 / 4096 chars |
| L4 | Bigram repetition ratio | < 0.35 |
| L5 | Identity (input ≡ target) | case-insensitive |
| L6 | Jaccard circularity | < 0.85 |
| L7 | SHA-256 deduplication | unique per batch |

**Throughput**: 500 samples in <9ms; 1,000 in <250ms; thread-safe under 10 workers.

### 4. Evaluation Harness
- **Local evaluator**: TTR, n-gram, Jaccard → **0.013ms/probe** (50 probes in 0.64ms)
- **Teacher judge**: Groq 70B, structured JSON rubric, 4-axis scoring
- **Judge–human agreement**: Cohen's κ = 0.77 (CoT+exemplar rubric)
- **Tiered routing**: local fast-passes ≥ 0.70; teacher handles edge cases

---

## Results

| Metric | Baseline | Final | Delta |
|---|---|---|---|
| Training data | 35 samples | 285+ samples | +714% |
| Replay blending speed | ~4.5s | <0.05s | >90x |
| Evaluation latency | ~2.5s/probe | 0.013ms (local) | >99.9% |
| SHA-256 dedup (500 samples) | unchecked | <9ms | quality gate added |
| Endpoint hang time per batch | ~200s | ~0s (failover) | eliminated |
| Regression tests | — | 95 tests, 100% pass | — |
| RLAIF reward | — | 1.000 (all 5 stages) | — |
| Judge–human κ | 0.66 (Rubric A) | 0.77 (Rubric B) | +0.11 |

---

## CIRCLE Integration

This toolkit is the extracted research layer of **CIRCLE** (Continuous Iterative Refinement through Critique & Learning Engine). See [github.com/ankitghoshthecreator/CIRCLE](https://github.com/ankitghoshthecreator/CIRCLE).

---

## Resume Bullets (AI Research Engineer / AI Research Intern — Product Companies)

- Designed a **post-training alignment pipeline** combining **5-stage curriculum SFT**, **RLAIF-style data flywheel** (**Groq 70B critic** → **severity-gated synthetic data commissioning** → **Ollama generator** → **7-layer quality gate** → **QLoRA retraining**), and an **automated eval harness** (**0.013ms/probe local evaluator**, **Cohen's κ = 0.77** judge–human agreement) — targeting the exact research stack Meta's Llama post-training team and Anthropic's Applied Fine-tuning team work on.
- Fixed **catastrophic memorization** (**~56K steps on 35 samples**, loss **8.78 → 0.0001**) with **prompt-label masking** and **replay-buffer blending** (stage-transition **~4.5s → <0.05s, >90x speedup**); grew training corpus **35 → 285+ samples** via closed-loop **synthetic data generation** with **SHA-256 deduplication** and **Jaccard circularity filtering** (**500 samples in <9ms**).
- Built **95 automated regression tests (100% pass)** across 4 stress rounds, **tiered evaluation routing** (local **0.013ms** vs **~2.5s** 70B API, **>99.9% latency reduction**), and **endpoint failover** eliminating up to **200s socket hangs** — all on a **4GB consumer GPU** under strict compute constraints.

---

## Reproducing Results

```bash
git clone https://github.com/ankitghoshthecreator/CIRCLE
pip install -r requirements.txt
export GROQ_API_KEY=your_key

# Full alignment pipeline
python scripts/rl_loop.py --stages 1 5 --max-iterations 200

# Eval harness only
python eval/run_stage_eval.py --stage 5 --checkpoint trainer/checkpoints/checkpoint_best

# Run test suite
pytest tests/ -v --tb=short
```

---

## What's Next

- Scale to Llama 3.1-8B with DPO preference optimization after SFT
- Add GRPO reward shaping for math/reasoning tasks
- Publish per-stage held-out probe pass rate benchmark table
- Integrate W&B for training run tracking and hyperparameter sweeps
