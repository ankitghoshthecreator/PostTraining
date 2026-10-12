# Post-Training Alignment Toolkit
### SFT · RLAIF Data Flywheel · Tiered Evaluation Harness for Open-Weight LLMs (4GB VRAM)

> **Role target:** AI Research Engineer / AI Research Intern
> **Target teams:** Meta (Llama post-training) · Anthropic (Applied AI, Finetuning) · Google (AI research) · Cohere · Mistral
> **Parent system:** CIRCLE, the Continuous Iterative Refinement through Critique & Learning Engine (`github.com/ankitghoshthecreator/CIRCLE`)
> **Companion file:** `phase.md` splits this project into 3 build phases with task-level detail.

---

## Table of Contents

1. [What this project is](#1-what-this-project-is)
2. [Research questions, hypotheses, and how each is answered](#2-research-questions-hypotheses-and-how-each-is-answered)
3. [System architecture](#3-system-architecture)
4. [Repository layout](#4-repository-layout)
5. [Environment, hardware budget, and setup](#5-environment-hardware-budget-and-setup)
6. [Data layer](#6-data-layer)
7. [Phase 1 component: Curriculum SFT](#7-phase-1-component-curriculum-sft)
8. [Phase 2 component: RLAIF closed loop](#8-phase-2-component-rlaif-closed-loop)
9. [Phase 3 component: Evaluation harness](#9-phase-3-component-evaluation-harness)
10. [Testing strategy (95 tests)](#10-testing-strategy-95-tests)
11. [Infrastructure: Docker and Kubernetes](#11-infrastructure-docker-and-kubernetes)
12. [Configuration reference](#12-configuration-reference)
13. [Experiments and ablations to run](#13-experiments-and-ablations-to-run)
14. [Results](#14-results)
15. [Failure log and troubleshooting](#15-failure-log-and-troubleshooting)
16. [Roadmap](#16-roadmap)
17. [Reproducing results](#17-reproducing-results)
18. [Resume bullets](#18-resume-bullets)
19. [Conventions, limitations, and honesty notes](#19-conventions-limitations-and-honesty-notes)

---

## 1. What this project is

A reproducible **post-training alignment pipeline** for small open-weight language models. It has three cooperating subsystems:

| Subsystem | Purpose | Core idea |
|---|---|---|
| **Curriculum SFT** | Teach a small base model multiple conversational skills without forgetting | 5-stage curriculum + replay buffer + prompt-label masking + 4-bit QLoRA |
| **RLAIF flywheel** | Replace human annotation with AI feedback | A 70B teacher critiques the student, a failure taxonomy converts critiques into targeted data requests, a local generator produces the data, a 7-layer gate filters it, and the student retrains |
| **Evaluation harness** | Measure progress cheaply and trustworthily | Tiered routing: a ~0.013 ms local evaluator handles easy probes, the 70B teacher handles the hard ones, and a Cohen's κ study validates the judge against humans |

Everything runs on a **4GB consumer GPU** (RTX 3050 class). The constraint is part of the research question: *can AI feedback reliably replace human annotation in a low-resource alignment setting?*

### Design principles

1. **Measure before optimizing.** Every claim in the results table is backed by a logged number (CSV, JSON, or test output).
2. **Closed loops need gates.** Any loop where a model generates its own training data must be protected against poisoning (the 7-layer validator).
3. **Cheap judge first, expensive judge second.** The teacher API is called only where the local evaluator is uncertain.
4. **Fail over, never hang.** Any network or local endpoint can die; the pipeline must continue (the ~200 s hang per batch was eliminated by failover).
5. **Everything is a config.** Stage ratios, thresholds, and model names live in YAML, not code.

---

## 2. Research questions, hypotheses, and how each is answered

| # | Question | Hypothesis | Experiment | Primary metric | Where in repo |
|---|---|---|---|---|---|
| RQ1 | Can RLAIF from a 70B teacher replace human annotation for small-LLM alignment? | Targeted AI-commissioned data closes failure categories faster than random synthetic data | Run the flywheel vs. a random-synthetic baseline with equal sample budget | Per-stage reward, failure-category counts per iteration | `scripts/rl_loop.py`, `eval/` |
| RQ2 | Does curriculum learning beat mixed-dataset fine-tuning for multi-task conversational LLMs? | Curriculum + replay beats mixed on later-stage tasks and retains earlier-stage skill | Train the same model on (a) mixed shuffled data, (b) curriculum without replay, (c) curriculum with replay | Held-out probe pass rate per stage, forgetting delta | `trainer/`, `configs/curriculum.yaml` |
| RQ3 | Which synthetic-data quality filters are necessary and sufficient to prevent poisoning? | The 7-layer gate is sufficient, and L4 (repetition), L6 (circularity), and L7 (dedup) are individually necessary | Ablate one layer at a time, run N loop iterations, track diversity and eval drift | Distinct-n, bigram repetition ratio, reward trend | `validator/`, `tests/test_validator*.py` |
| RQ4 | How well does a lightweight local evaluator proxy a 70B judge? | TTR + n-gram + Jaccard score correlates strongly with teacher scores on easy probes | Score the same probes with both; compute Pearson/Spearman and agreement at the 0.70 threshold | Correlation, routing precision/recall, cost saved | `eval/local_evaluator.py`, `eval/agreement.py` |

For each RQ, record the *answer*, even if it is negative. A research-intern portfolio reads stronger with an honest "the hypothesis was partly wrong, here is why" than with only wins.

---

## 3. System architecture

```
                         ┌───────────────────────────────────────────┐
                         │               configs/*.yaml              │
                         └───────────────────────────────────────────┘
                                            │
┌──────────────── PHASE 1: SFT ─────────────┼───────────────────────────────┐
│ data/curriculum/stage_{1..5}.jsonl        ▼                               │
│   → schema validation → tokenization → prompt-label masking (-100)        │
│   → ReplayBuffer (in-memory, stage-specific ratios 0.0–0.4)               │
│   → 4-bit NF4 QLoRA (r=16, α=32, AdamW lr=2e-4, linear warmup)            │
│   → per-step CSV logging → best-checkpoint tracker → checkpoint_best/     │
└───────────────────────────────────────────┬───────────────────────────────┘
                                            ▼
┌──────────────── PHASE 2: RLAIF LOOP ──────────────────────────────────────┐
│ 1. Probe the student on the stage probe bank                               │
│ 2. Teacher (Groq llama-3-70b) critiques each response → JSON (Pydantic v2) │
│ 3. <think> sanitization → Failure Taxonomy (7 categories, 4 severities)    │
│ 4. Severity gate: CRITICAL/HIGH only, priority ≤ 5                         │
│ 5. Build targeted generation prompt                                        │
│ 6. Local generator (Ollama, qwen2.5-coder:7b) with instant failover        │
│ 7. 7-layer validator (L1–L7)                                               │
│ 8. Append accepted samples to dataset → trigger SFT retrain → loop         │
└───────────────────────────────────────────┬───────────────────────────────┘
                                            ▼
┌──────────────── PHASE 3: EVALUATION ──────────────────────────────────────┐
│ Tiered harness:  local evaluator (TTR / n-gram / Jaccard)                  │
│                  ≥ 0.70 → fast pass                                        │
│                  < 0.70 → escalate to 70B teacher (4-axis JSON rubric)     │
│ Reward scalar in [0,1] per stage → regression gate for the loop            │
│ Judge–human agreement study (Cohen's κ)                                    │
│ 95 pytest regression tests, CI                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

**Data flow summary:** curriculum files → SFT → checkpoint → probes → critique → taxonomy → targeted data → gate → enriched dataset → SFT. Evaluation reads the same checkpoint and produces the reward scalar that decides whether the loop continues.

---

## 4. Repository layout

```
CIRCLE/
├── README.md
├── phase.md
├── requirements.txt
├── pyproject.toml
├── .env.example                  # GROQ_API_KEY, OLLAMA_HOST, WANDB_API_KEY (optional)
├── configs/
│   ├── model.yaml                # base model, quantization, LoRA
│   ├── curriculum.yaml           # stages, ratios, epochs, LR per stage
│   ├── rlaif.yaml                # severity gate, priority cap, iteration caps
│   ├── validator.yaml            # L1–L7 thresholds
│   └── eval.yaml                 # local thresholds, escalation band, rubric version
├── data/
│   ├── curriculum/
│   │   ├── stage_1_english.jsonl
│   │   ├── stage_2_summarization.jsonl
│   │   ├── stage_3_grammar.jsonl
│   │   ├── stage_4_punctuation.jsonl
│   │   └── stage_5_conversation.jsonl
│   ├── synthetic/                # accepted generator output, versioned per iteration
│   ├── rejected/                 # validator rejects with reason codes (for audit)
│   └── probes/                   # per-stage probe banks (held out from training)
├── trainer/
│   ├── dataset.py                # loading, tokenization, masking
│   ├── replay_buffer.py          # in-memory blending
│   ├── train_sft.py              # main training entry
│   ├── callbacks.py              # CSV logging, best-checkpoint tracker
│   └── checkpoints/
├── rlaif/
│   ├── probe.py                  # run the student on probes
│   ├── critic.py                 # Groq 70B client + Pydantic schemas
│   ├── taxonomy.py               # FailureCategory, Severity, mapping logic
│   ├── sanitize.py               # <think> tag removal
│   ├── gate.py                   # severity gating + priority
│   ├── prompt_builder.py         # targeted generation prompts
│   ├── generator.py              # Ollama client + failover
│   └── controller.py             # loop orchestration
├── validator/
│   ├── layers.py                 # L1–L7 individual checks
│   └── pipeline.py               # ordered execution, reason codes, stats
├── eval/
│   ├── local_evaluator.py        # TTR, n-gram, Jaccard
│   ├── teacher_judge.py          # 70B rubric judge
│   ├── router.py                 # tiered routing
│   ├── reward.py                 # reward scalar
│   ├── agreement.py              # Cohen's κ, correlation
│   ├── metrics.py                # ROUGE-L, BERTScore wrappers
│   └── run_stage_eval.py
├── scripts/
│   ├── rl_loop.py                # CLI for the full pipeline
│   └── make_probes.py
├── tests/                        # 95 tests
├── docker/
│   ├── Dockerfile                # multi-stage
│   └── docker-compose.yml
├── k8s/
│   ├── deployment.yaml
│   ├── hpa.yaml                  # 1→4 replicas
│   └── configmap.yaml
├── logs/                         # per-step loss CSV, loop logs
└── docs/
    └── FAILURES_AND_OPTIMIZATIONS.md
```

---

## 5. Environment, hardware budget, and setup

### 5.1 Hardware target

| Resource | Target | Notes |
|---|---|---|
| GPU | RTX 3050, 4GB VRAM | The binding constraint for all training choices |
| Python | 3.10 or 3.11 | |
| CUDA | 11.8 or 12.x, matched to your PyTorch build | bitsandbytes must match |
| Local generator | Ollama | A 7B model will not fully fit in 4GB. Use a 4-bit quantized build with partial CPU offload, or fall back to a 3B model |

### 5.2 VRAM budget (planning table)

Training and generation must **not** hold the GPU at the same time. The controller serializes them.

| Model | Precision | Approx. weights | Fits 4GB for QLoRA? |
|---|---|---|---|
| GPT-2 (117M) | fp16 or 4-bit | < 0.3GB | Yes, large headroom. Use for fast iteration |
| Qwen2.5-1.5B-Instruct | 4-bit NF4 | ~1.0–1.2GB | Yes, with gradient checkpointing and small batch |
| Phi-3-mini-4k (3.8B) | 4-bit NF4 | ~2.2–2.6GB | Tight. Needs batch 1, grad accumulation, short sequences (≤ 512), gradient checkpointing |

Techniques to stay in budget: gradient checkpointing, micro-batch 1–2 with accumulation 8–16, max sequence length 256–512, paged AdamW 8-bit if needed, `torch.cuda.empty_cache()` between phases, and unload Ollama's model (`keep_alive: 0`) before training.

### 5.3 Setup

```bash
git clone https://github.com/ankitghoshthecreator/CIRCLE
cd CIRCLE
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # fill GROQ_API_KEY, optionally WANDB_API_KEY
ollama pull qwen2.5-coder:7b   # or a smaller tag if VRAM-limited
pytest tests/ -q               # sanity check before anything else
```

### 5.4 Pinned dependency categories (`requirements.txt`)

`torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `pydantic>=2`, `httpx` or `groq`, `ollama`, `rouge-score`, `bert-score`, `scikit-learn` (for Cohen's κ), `pandas`, `pyyaml`, `pytest`, `pytest-cov`, `tenacity` (retries), `wandb` (optional). Pin exact versions once the pipeline works, and commit the lockfile.

---

## 6. Data layer

### 6.1 Curriculum stages

| Stage | Skill | Example task | Replay ratio (from earlier stages) |
|---|---|---|---|
| 1 | English fluency | Continue / complete coherent English | 0.0 |
| 2 | Summarization | "Summarize the following…" | tune within 0.0–0.4 |
| 3 | Grammar correction | "Fix the grammar…" | tune within 0.0–0.4 |
| 4 | Punctuation restoration | "Add punctuation (, . ! ?)…" | tune within 0.0–0.4 |
| 5 | Conversation | Multi-turn, instruction-following dialogue | up to 0.4 |

Stage-specific ratios are config values (`curriculum.yaml`). Start with a ramp such as 0.0 / 0.2 / 0.3 / 0.3 / 0.4, then tune by measuring forgetting on earlier-stage probes.

### 6.2 Sample schema

One JSON object per line:

```json
{
  "id": "s3-000142",
  "stage": 3,
  "task": "grammar",
  "prompt": "Fix the grammar: she dont like apples",
  "target": "She doesn't like apples.",
  "source": "human|synthetic",
  "iteration": 0,
  "meta": {"generator": "ollama:qwen2.5-coder:7b", "critic_failure_id": null}
}
```

Validation rules (they double as validator layers L1–L3): all required keys exist, `prompt` and `target` are non-empty strings, `target` length is within 10–4096 characters, and `stage` is in 1–5.

### 6.3 Dataset growth

- **Baseline:** 35 hand-built samples.
- **Final:** 285+ samples across 5 stages (+714%), the extra samples coming from the validated RLAIF generator.
- Keep human-authored and synthetic samples in separate files so ablations ("human only" vs. "human + synthetic") are a one-flag change.

### 6.4 Probe banks (held out)

- Separate files in `data/probes/stage_{n}.jsonl`.
- **Never** appear in any training file. Enforce this with a test that hashes every probe prompt and checks for collisions with training prompts (including near-duplicates by Jaccard ≥ 0.85).
- Each probe carries an optional `reference` (for ROUGE-L / Jaccard) and a `checks` list (for rule-based checks like "must contain a period").

---

## 7. Phase 1 component: Curriculum SFT

### 7.1 Model loading (4-bit NF4)

```python
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,   # bfloat16 if the GPU supports it
)
model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model)
```

Set `model.config.use_cache = False` during training, and enable gradient checkpointing.

### 7.2 LoRA configuration

| Hyperparameter | Value |
|---|---|
| Rank `r` | 16 |
| Alpha `α` | 32 |
| Dropout | 0.05 (recommended default) |
| Target modules | Architecture-specific (e.g. `q_proj,k_proj,v_proj,o_proj` for Qwen/Phi; `c_attn,c_proj` for GPT-2). Store in `model.yaml` per model |
| Task type | `CAUSAL_LM` |

### 7.3 Optimizer and schedule

- AdamW, **lr = 2e-4**, weight decay 0.0–0.01, linear warmup (5–10% of steps) then linear or cosine decay.
- Effective batch size 8–16 via gradient accumulation.
- Gradient clipping at 1.0.
- Fixed seeds (`torch`, `numpy`, `random`) written into every run's metadata file.

### 7.3.1 The memorization failure, and why it matters

**What happened:** the first runs trained ~**56,000 steps on 35 samples**. Loss fell **8.78 → 0.0001** and the model reproduced training data verbatim. Zero loss on a tiny set is a warning sign, not a success.

**Diagnosis checklist** (keep this as a documented procedure):
1. Is train loss near zero while held-out probe quality is poor? → overfit.
2. Does greedy decoding of a training prompt reproduce the target token-for-token? → memorization.
3. Is the step count wildly larger than `epochs × samples / batch`? → misconfigured schedule.

**Fixes applied:** a 5-stage curriculum, a replay buffer, prompt-label masking, and a larger corpus via the flywheel. Also add early stopping on held-out probe reward, and cap steps per stage.

### 7.4 Prompt-label masking

Train only on the *response* tokens. Mask the prompt so the model is not rewarded for copying the instruction.

```python
prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
full_ids   = tok(prompt_text + target_text + tok.eos_token, add_special_tokens=False)["input_ids"]
labels     = full_ids.copy()
labels[:len(prompt_ids)] = [-100] * len(prompt_ids)   # ← the core trick
```

Details that commonly go wrong:
- **Tokenization boundary:** tokenize prompt and full text with the same function and verify `full_ids[:len(prompt_ids)] == prompt_ids`. If a BPE merge crosses the boundary, add an explicit separator (e.g. `"\n### Response:\n"`) included in the prompt part.
- **Truncation:** truncate the *prompt* side first, never the target, and drop samples where the target would be cut.
- **Padding:** pad `input_ids` with `pad_token_id`, `labels` with `-100`, `attention_mask` with 0. For GPT-2 (no pad token) set `pad_token = eos_token`, but then mask the pad positions in `labels` explicitly, so the real EOS is still learned.
- **Unit test:** assert that for every sample, `sum(l != -100 for l in labels) == len(target_tokens)+1`.

Effect: recovers instruction-following without changing the curriculum.

### 7.5 Replay buffer

Purpose: when moving from stage *k* to *k+1*, mix a fraction of earlier-stage samples into each epoch so earlier skills are not forgotten.

```python
class ReplayBuffer:
    def __init__(self, seed=0):
        self.by_stage: dict[int, list[Sample]] = {}
        self.rng = random.Random(seed)

    def add_stage(self, stage, samples):        # kept in RAM, tokenized once
        self.by_stage[stage] = samples

    def blend(self, current_stage, current_samples, ratio):
        pool = [s for k, v in self.by_stage.items() if k < current_stage for s in v]
        n = int(len(current_samples) * ratio / (1 - ratio)) if ratio < 1 else 0
        replay = self.rng.sample(pool, min(n, len(pool)))
        mixed = current_samples + replay
        self.rng.shuffle(mixed)
        return mixed
```

- **Why in-memory:** earlier disk-based blending cost ~**4.5 s** per stage transition. In-memory pre-tokenized blending takes **< 0.05 s** (> 90× faster).
- **Ratio semantics:** define it unambiguously, e.g. "replay fraction of the *final* mixed set". Document it in `curriculum.yaml` and test it.
- **Balance:** sample replay evenly across earlier stages, so stage 1 doesn't dominate.
- **Determinism:** the seeded RNG makes blends reproducible.

### 7.6 Logging and checkpointing

- **Per-step CSV** (`logs/<run_id>/train_steps.csv`): `step, epoch, stage, lr, loss, grad_norm, tokens_seen, gpu_mem_mb, wall_time_s`.
- **Eval CSV** (`eval_steps.csv`): per-stage probe reward every N steps.
- **Best-checkpoint tracker:** a callback that saves to `checkpoint_best/` only when the held-out reward improves (not on train loss). Also write `checkpoint_last/` for resume.
- **Run metadata** (`run.json`): git commit, config hash, seeds, model name, dataset hashes, library versions.

### 7.7 CLI

```bash
python trainer/train_sft.py --config configs/curriculum.yaml --stages 1 5 --model qwen2.5-1.5b
```

---

## 8. Phase 2 component: RLAIF closed loop

### 8.1 Probe step (`rlaif/probe.py`)

Generate student responses for each probe in the current stage using fixed decoding (temperature 0 or a low-temperature, seeded config). Save `{probe_id, prompt, response, latency}`. Free VRAM afterwards.

### 8.2 Teacher critique (`rlaif/critic.py`)

- **Model:** Groq `llama-3-70b` (confirm the current model ID in Groq's catalog, since hosted model names change).
- **Request:** system prompt with the rubric and failure taxonomy, user content with the probe, reference (if any), and student response. Request JSON-only output and set low temperature.
- **Schema (Pydantic v2):**

```python
class Severity(str, Enum):
    CRITICAL="CRITICAL"; HIGH="HIGH"; MEDIUM="MEDIUM"; LOW="LOW"

class FailureCategory(str, Enum):
    # 7 categories. Suggested set, adapt to your taxonomy:
    REPETITION="repetition"; OFF_TOPIC="off_topic"; INSTRUCTION_IGNORED="instruction_ignored"
    GRAMMAR_ERROR="grammar_error"; PUNCTUATION_ERROR="punctuation_error"
    HALLUCINATION="hallucination"; FORMAT_VIOLATION="format_violation"

class Critique(BaseModel):
    probe_id: str
    score: float = Field(ge=0, le=1)
    failures: list[Failure]       # category, severity, evidence, suggested_fix
    summary: str
    priority: int = Field(ge=1, le=10)   # 1 = most urgent
```

- **Robustness:** retries with exponential backoff (`tenacity`), strip markdown code fences before `json.loads`, validate with Pydantic, and on validation failure re-ask once with the error message, then skip and log.
- **Rate limits:** a token-bucket limiter, a cached response keyed by `hash(prompt+response+rubric_version)`.

### 8.3 `<think>` sanitization (`rlaif/sanitize.py`)

Reasoning-model critics emit chain-of-thought inside `<think>…</think>`. If that text contains words like "repetition" or "hallucination" in the *reasoning*, a naive parser counts them as detections, which is a **false positive** source.

```python
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
def strip_think(text): 
    text = THINK_RE.sub("", text)
    return re.sub(r"</?think>", "", text).strip()   # handle unclosed/stray tags
```

Always sanitize *before* JSON extraction, and test with: nested tags, unclosed tag, multiple blocks, and tags inside JSON string values.

### 8.4 Failure taxonomy and severity gate (`rlaif/taxonomy.py`, `gate.py`)

- 7 categories × 4 severities (CRITICAL / HIGH / MEDIUM / LOW).
- **Gate rule:** only failures with severity ∈ {CRITICAL, HIGH} **and** `priority ≤ 5` trigger data generation. MEDIUM/LOW are logged for the report but do not spend generator budget.
- **Aggregation:** group failures by `(stage, category)`; request data per group, not per probe, so the generator gets a *pattern* ("model drops commas after introductory clauses") instead of a single anecdote.
- **Per-iteration budget cap:** e.g. max 40 generation requests per iteration.

### 8.5 Targeted prompt builder (`rlaif/prompt_builder.py`)

Template contents: the stage's task definition, the failure pattern in plain words, 2–3 *good* exemplars, 1–2 *bad* exemplars (from the student's actual failures), the required JSON-lines output format, a count (e.g. 8 samples), and a diversity instruction (vary topic, length, sentence structure). Never include probe-bank items verbatim (protects held-out integrity).

### 8.6 Generator with failover (`rlaif/generator.py`)

- **Primary:** Ollama (`qwen2.5-coder:7b`) via `http://localhost:11434`.
- **Timeouts:** explicit connect (≈3 s) and read (≈60 s) timeouts. The ~200 s per-batch hang came from default socket behavior against a dead or overloaded endpoint.
- **Health probe:** a cheap `GET /api/tags` before each batch.
- **Failover chain:** Ollama primary → (smaller local model) → (teacher API in "generation" mode, flagged `source: "teacher"` for auditing) → skip batch and log. Result: batch hang time ~200 s → ~0 s.
- **Serialization:** unload the student's GPU memory before Ollama loads, and the reverse before training.

### 8.7 The 7-layer synthetic data quality gate (`validator/`)

| Layer | Check | Threshold / rule | Reject code |
|---|---|---|---|
| L1 | Schema valid | required keys, types | `L1_SCHEMA` |
| L2 | Minimum length | target ≥ 10 chars | `L2_TOO_SHORT` |
| L3 | Maximum length | target ≤ 4096 chars | `L3_TOO_LONG` |
| L4 | Bigram repetition ratio | `1 − unique_bigrams/total_bigrams` < 0.35 | `L4_REPETITIVE` |
| L5 | Identity | input ≠ target (case-insensitive, whitespace-normalized) | `L5_IDENTITY` |
| L6 | Jaccard circularity | token-set Jaccard(input, target) < 0.85 (and vs. existing corpus, to avoid near-copies) | `L6_CIRCULAR` |
| L7 | SHA-256 dedup | hash of normalized `prompt\x1ftarget`, unique within batch *and* corpus | `L7_DUPLICATE` |

Implementation notes:
- **Order is cheapest-first** and short-circuits: schema → length → identity → repetition → circularity → dedup.
- **Normalize before hashing:** lowercase, collapse whitespace, strip. Otherwise trivial variants evade dedup.
- **Thread safety:** the dedup set sits behind a `threading.Lock`; the stress test runs 10 worker threads.
- **Counters:** per-layer reject counts are written to the iteration report, since a sudden spike in one layer is an early warning of generator drift.
- **Throughput targets:** 500 samples in < 9 ms; 1,000 in < 250 ms.
- **Rejected sample log:** written to `data/rejected/` with reason codes, which is what makes RQ3 ablations possible.

### 8.8 Loop controller (`rlaif/controller.py`)

```
for iteration in range(max_iterations):
    1. load checkpoint_best
    2. probe → critique → sanitize → taxonomy → gate
    3. if no gated failures AND reward ≥ target:   stop (converged)
    4. build prompts → generate (with failover) → validate → append accepted
    5. if accepted_count < min_new_samples:        stop (starved) or widen budget
    6. retrain SFT (stages from config; replay includes new data)
    7. run eval harness → reward
    8. if reward drops > regress_tol vs. best:     roll back dataset version, flag iteration
    9. write iteration report (JSON + markdown)
```

**Stop conditions:** convergence (reward ≥ target on all stages), starvation, max iterations (200 in the reference command), a regression streak, or an API budget exhausted.
**Verification run:** reward **r = 1.000** across all 5 curriculum stages. When reporting this, state the probe bank size per stage and whether probes were held out; a perfect score on a small probe bank invites skepticism, so also report a larger held-out set in the Phase 3 benchmark table.

---

## 9. Phase 3 component: Evaluation harness

### 9.1 Local evaluator (`eval/local_evaluator.py`)

Signals (all cheap, pure Python):

| Signal | Definition | Detects |
|---|---|---|
| **TTR** | unique tokens / total tokens | degenerate or looping text (low TTR) |
| **n-gram windowing** | max repeated n-gram frequency in sliding windows (n = 2–4) | local repetition |
| **Jaccard** | |A∩B| / |A∪B| between response tokens and reference tokens | overlap with the expected answer |
| **Rule checks** | stage-specific: ends with terminal punctuation, non-empty, length band, language | format failures |

Combine into a score in [0, 1], e.g. `score = w1·ttr_norm + w2·(1 − ngram_rep) + w3·jaccard + w4·rules`, with weights in `eval.yaml`. **Fit the weights on the κ-study labeled set** instead of guessing.
Performance: **0.013 ms per probe** (50 probes in 0.64 ms), versus ~2.5 s for a 70B API round trip, which is a > 99.9% latency reduction. Benchmark with `timeit` over many repeats and report median plus p95.

### 9.2 Teacher judge (`eval/teacher_judge.py`)

- Groq 70B, structured JSON rubric, **4-axis scoring** (choose axes that fit your tasks, for example: *correctness/instruction-following*, *fluency/grammar*, *faithfulness/no hallucination*, *format/punctuation*), each 1–5, normalized to [0, 1] and averaged (or weighted).
- **Rubric versions:**
  - **Rubric A:** a plain scoring prompt → agreement with humans κ = **0.66**.
  - **Rubric B:** chain-of-thought plus worked exemplars → κ = **0.77** (+0.11).
- Always store the rubric version string with each judgment so results are comparable.
- Mitigate judge biases: randomize the order of reference/response blocks, ignore length, and require the justification *before* the score in the JSON field order.

### 9.3 Tiered routing (`eval/router.py`)

```
local_score = local_evaluator(probe, response)
if local_score >= 0.70:     verdict = local            # fast pass
else:                       verdict = teacher_judge()  # escalate
```

Log, per run: fraction handled locally, teacher calls made, estimated API time saved, and (on the labeled set) the **false-pass rate**, which is the rate at which the local evaluator passes something the teacher would fail. That false-pass rate is the number that answers RQ4 honestly. If it's high, tighten the 0.70 threshold, add rule checks, or add a hard-fail floor.

### 9.4 Reward scalar (`eval/reward.py`)

Per stage: `reward = mean(probe_scores)` in [0, 1]. Aggregate across stages with the mean, plus `min_stage_reward` to catch a collapsed stage hidden by the average. The loop controller uses both.

### 9.5 Judge–human agreement study (`eval/agreement.py`)

1. Sample ~100 (probe, response) pairs spanning all stages and score ranges (stratified).
2. Have a human (you, ideally plus one more annotator) label pass/fail or a 1–5 score **blind** to the judge's output.
3. Run the judge with Rubric A and Rubric B.
4. Compute Cohen's κ: `κ = (p_o − p_e) / (1 − p_e)`, with `scikit-learn`'s `cohen_kappa_score` (use `weights="quadratic"` for ordinal 1–5 scores). Report a bootstrap 95% CI.
5. Inspect disagreements and write them up. These are where the rubric improvements come from.

Interpretation guide: 0.61–0.80 is conventionally "substantial" agreement, so 0.77 sits in that band. With a small sample, κ has wide uncertainty, so report the sample size and CI.

### 9.6 Reference metrics (`eval/metrics.py`)

ROUGE-L (summarization), BERTScore (semantic similarity), exact match / edit distance (grammar, punctuation). These supplement but do not replace the judge.

### 9.7 Stage eval CLI

```bash
python eval/run_stage_eval.py --stage 5 --checkpoint trainer/checkpoints/checkpoint_best
```

Outputs: `eval_report.json`, `eval_report.md`, and a per-probe CSV.

---

## 10. Testing strategy (95 tests)

**Run:** `pytest tests/ -v --tb=short` (100% pass is the reference state). The suite was built across 4 stress rounds. Suggested layout (adjust the counts to match your real suite):

| Area | What to test |
|---|---|
| Masking | boundary equality, label counts, truncation, pad handling |
| Replay buffer | ratio math, determinism with seed, empty pool, stage ordering |
| Dataset I/O | malformed JSONL, empty lines, unicode, huge samples |
| Sanitizer | nested/unclosed/multiple `<think>` tags |
| Critic parsing | fenced JSON, trailing text, missing fields, retry path |
| Taxonomy/gate | severity × priority matrix, boundary priority = 5 vs. 6 |
| Validator L1–L7 | each layer's pass/fail edges, reason codes, order |
| Validator performance | 500 in < 9 ms, 1,000 in < 250 ms (use generous CI margins) |
| Concurrency | 10 threads, no duplicate acceptance, no deadlock |
| Failover | endpoint down, timeouts, slow responses, recovery |
| Local evaluator | range [0,1], determinism, empty string, long input |
| Router | threshold boundary 0.70, escalation count |
| Agreement | κ on known toy cases (perfect = 1, chance ≈ 0) |
| Probe leakage | no probe appears in training data |

CI: GitHub Actions running `pytest` on every push, with coverage reporting. Mark network tests with `@pytest.mark.network` and mock them by default.

---

## 11. Infrastructure: Docker and Kubernetes

- **Docker multi-stage:** a `builder` stage installs wheels, and a slim `runtime` stage copies only what's needed, runs as non-root, and sets `PYTHONUNBUFFERED=1`. Use a CUDA base image for the training image and a CPU-only image for the eval/validator services.
- **Compose:** services for `trainer`, `ollama`, and `evaluator`, with volumes for `data/`, `logs/`, and `checkpoints/`.
- **Kubernetes:** a `Deployment` for the evaluator/validator service, with an **HPA scaling 1→4 replicas** on CPU utilization; secrets for `GROQ_API_KEY`; `ConfigMap` for YAML configs; readiness probe on a `/healthz` endpoint; resource requests/limits set explicitly. Training stays a single-GPU job (a `Job`, not an autoscaled deployment).
- Be explicit in docs that K8s is for the stateless eval/validation tier, and that the 4GB-GPU trainer is run locally.

---

## 12. Configuration reference

```yaml
# configs/curriculum.yaml
stages:
  - {id: 1, file: stage_1_english.jsonl,       epochs: 3, replay_ratio: 0.0}
  - {id: 2, file: stage_2_summarization.jsonl, epochs: 3, replay_ratio: 0.2}
  - {id: 3, file: stage_3_grammar.jsonl,       epochs: 3, replay_ratio: 0.3}
  - {id: 4, file: stage_4_punctuation.jsonl,   epochs: 3, replay_ratio: 0.3}
  - {id: 5, file: stage_5_conversation.jsonl,  epochs: 3, replay_ratio: 0.4}
max_steps_per_stage: 2000
early_stop: {metric: probe_reward, patience: 3}

# configs/rlaif.yaml
severity_gate: [CRITICAL, HIGH]
priority_max: 5
max_iterations: 200
requests_per_iteration: 40
samples_per_request: 8
min_new_samples: 5
regress_tolerance: 0.05
target_reward: 0.95

# configs/validator.yaml
min_chars: 10
max_chars: 4096
bigram_repetition_max: 0.35
jaccard_circularity_max: 0.85
dedup: sha256

# configs/eval.yaml
local_pass_threshold: 0.70
rubric_version: B
teacher_model: llama-3-70b
```

(The values above marked as defaults beyond the documented ones are starting points. Tune them and record the final values in your run metadata.)

---

## 13. Experiments and ablations to run

1. **Curriculum vs. mixed vs. curriculum+replay** (RQ2): three arms, same compute, 3 seeds each. Plot per-stage probe reward and forgetting.
2. **Masking on/off:** show the memorization symptom (verbatim reproduction rate) and instruction-following with and without masking.
3. **Gate layer ablation** (RQ3): disable one layer per run, run 10+ loop iterations, and track reward, distinct-2, and repetition ratio.
4. **RLAIF vs. random synthetic vs. human-only** (RQ1): equal sample budget.
5. **Judge rubric A vs. B** (RQ4): κ with bootstrap CI.
6. **Local-evaluator threshold sweep:** thresholds 0.5–0.9, plot cost saved vs. false-pass rate.
7. **Model scaling:** GPT-2 → Qwen2.5-1.5B → Phi-3-mini under the same pipeline.

For every experiment, save the config, seed, git SHA, and raw CSVs under `experiments/<name>/`.

---

## 14. Results

| Metric | Baseline | Final | Delta |
|---|---|---|---|
| Training data | 35 samples | 285+ samples | +714% |
| Replay blending speed | ~4.5 s | < 0.05 s | > 90× |
| Evaluation latency | ~2.5 s/probe | 0.013 ms (local) | > 99.9% |
| SHA-256 dedup (500 samples) | unchecked | < 9 ms | quality gate added |
| Endpoint hang time per batch | ~200 s | ~0 s (failover) | eliminated |
| Regression tests | none | 95 tests, 100% pass | n/a |
| RLAIF reward | n/a | 1.000 (all 5 stages) | n/a |
| Judge–human κ | 0.66 (Rubric A) | 0.77 (Rubric B) | +0.11 |
| Memorization run | loss 8.78 → 0.0001 over ~56K steps on 35 samples | fixed via masking + curriculum + replay | n/a |

**To add:** per-stage held-out probe pass-rate table, ablation plots, and confidence intervals.

---

## 15. Failure log and troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss → ~0, output copies training data | Too many steps on too little data | Curriculum, replay, masking, early stopping, more data |
| Model echoes the prompt | No prompt-label masking | Mask prompt tokens with -100 |
| CUDA OOM | Batch/sequence too long, Ollama still holding VRAM | Batch 1 + accumulation, shorter max length, gradient checkpointing, unload Ollama first |
| Critic returns invalid JSON | Fences or prose around the JSON, or `<think>` blocks | Sanitize, strip fences, re-ask once |
| False failure detections | CoT text parsed as findings | `<think>` stripping before parsing |
| Batch hangs ~200 s | No timeouts, dead endpoint | Explicit timeouts, health probe, failover |
| Synthetic data collapses (same sentences) | No dedup or repetition gate | L4, L6, L7 enforced, check per-layer reject counts |
| Reward 1.000 too early | Probe bank too small or leaked into training | Larger held-out set, leakage test |
| Local evaluator passes bad outputs | Weak signals for the stage | Add rule checks, retune weights, lower or raise threshold |
| Groq 429 errors | Rate limits | Token bucket, caching, backoff |

Maintain `docs/FAILURES_AND_OPTIMIZATIONS.md` as a running log (symptom → root cause → fix → measured effect), since it is strong evidence of research process.

---

## 16. Roadmap

- Scale to **Llama 3.1-8B** with **DPO** preference optimization after SFT (use the critic's pairwise judgments to build chosen/rejected pairs; QLoRA still applies, but expect to need a larger GPU or cloud credits).
- Add **GRPO** reward shaping for math/reasoning tasks, with verifiable rewards.
- Publish a **per-stage held-out probe pass-rate benchmark table**.
- Integrate **Weights & Biases** for run tracking and sweeps.
- Optional: multi-teacher ensemble, disagreement-based escalation, and active-learning style probe selection.

---

## 17. Reproducing results

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

## 18. Resume bullets

- Designed a **post-training alignment pipeline** combining **5-stage curriculum SFT**, an **RLAIF-style data flywheel** (**Groq 70B critic** → **severity-gated synthetic data commissioning** → **Ollama generator** → **7-layer quality gate** → **QLoRA retraining**), and an **automated eval harness** (**0.013 ms/probe local evaluator**, **Cohen's κ = 0.77** judge–human agreement), targeting the research stack of Llama post-training and applied fine-tuning teams.
- Fixed **catastrophic memorization** (**~56K steps on 35 samples**, loss **8.78 → 0.0001**) with **prompt-label masking** and **replay-buffer blending** (stage transition **~4.5 s → < 0.05 s, > 90× speedup**); grew the corpus **35 → 285+ samples** via closed-loop **synthetic data generation** with **SHA-256 deduplication** and **Jaccard circularity filtering** (**500 samples in < 9 ms**).
- Built **95 automated regression tests (100% pass)** across 4 stress rounds, **tiered evaluation routing** (local **0.013 ms** vs. **~2.5 s** 70B API, **> 99.9% latency reduction**), and **endpoint failover** eliminating up to **200 s socket hangs**, all on a **4GB consumer GPU**.

---

## 19. Conventions, limitations, and honesty notes

- **Small-scale caveat:** 285+ samples and small probe banks mean results are indicative, not statistically conclusive. State this plainly and add confidence intervals and multiple seeds where possible.
- **Teacher dependence:** RLAIF inherits the teacher's biases, and judge–human κ is measured on a limited sample.
- **Reward = 1.000** should be presented alongside a larger held-out evaluation, to pre-empt the "saturated benchmark" objection.
- **Naming:** keep the one expansion of CIRCLE you use consistent across the repo and your resume.
- **Code style:** type hints, `ruff` + `black`, docstrings on every public function, and no secrets in the repo (`.env` is git-ignored).
- **License:** pick one (MIT or Apache-2.0) and add `LICENSE`.
