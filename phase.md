# phase.md — Build Plan for the Post-Training Alignment Toolkit

The project is split into **3 major phases**. Each phase has: objective, prerequisites, step-by-step tasks with implementation detail, files produced, acceptance criteria, risks, and a time estimate. Phases are sequential: each one's *exit criteria* are the next one's *entry criteria*.

| Phase | Name | Core output | Rough effort |
|---|---|---|---|
| **1** | Foundation: Data + Curriculum SFT | A trained, non-memorizing QLoRA model with logged training and a best checkpoint | 1.5–2.5 weeks |
| **2** | RLAIF Flywheel: Critic → Generator → Quality Gate → Retrain | A closed loop that grows the dataset safely and improves the model per iteration | 2–3 weeks |
| **3** | Evaluation, Hardening, Infra, and Publication | Tiered eval harness, κ study, 95 tests, Docker/K8s, benchmark table, final write-up | 2–3 weeks |

Effort figures are planning estimates for one person working part-time. Adjust to your schedule.

---

# PHASE 1 — Foundation: Data + Curriculum SFT

## 1.1 Objective

Produce a small open-weight model fine-tuned through a **5-stage curriculum** on a 4GB GPU, using **4-bit NF4 QLoRA**, **prompt-label masking**, and an **in-memory replay buffer**, with full logging and best-checkpoint tracking, and **without memorization**.

## 1.2 Entry criteria

- Working Python env with CUDA, PyTorch, Transformers, PEFT, bitsandbytes.
- A base model downloaded (start with GPT-2 117M for speed; then Qwen2.5-1.5B-Instruct; Phi-3-mini-4k is the stretch target).
- Repo skeleton created (see README §4).

## 1.3 Tasks

### Task 1.1 — Repo, environment, and config skeleton
- Create the folder layout from the README. Add `pyproject.toml`, `requirements.txt`, `.env.example`, `.gitignore` (ignore `.env`, `checkpoints/`, `logs/`, `data/rejected/`).
- Create `configs/model.yaml`, `configs/curriculum.yaml` with the fields in README §12.
- Write a `config.py` loader that reads YAML, validates with Pydantic, and exposes a frozen config object.
- **Done when:** `python -c "from config import load; load('configs/curriculum.yaml')"` succeeds.

### Task 1.2 — Build the curriculum datasets
- Author the initial **35 hand-written samples** spread across 5 stages (about 7 each) as `data/curriculum/stage_{n}_*.jsonl`, using the schema in README §6.2.
- Stage content:
  1. **English:** coherent continuation/completion prompts.
  2. **Summarization:** short paragraph → 1–2 sentence summary.
  3. **Grammar:** erroneous sentence → corrected sentence.
  4. **Punctuation:** unpunctuated text → punctuated with `, . ! ?`.
  5. **Conversation:** short multi-turn or instruction-following exchanges.
- Write `trainer/dataset.py::load_jsonl(path)` with strict validation (required keys, non-empty strings, stage matches file).
- Create **held-out probe banks** (`data/probes/stage_{n}.jsonl`), at least 10 per stage, written separately from training data.
- **Done when:** a validation script reports 0 schema errors and a leakage test confirms 0 probe/training overlaps.

### Task 1.3 — Tokenization and prompt-label masking
- Implement `build_example(prompt, target, tokenizer, max_len)` returning `input_ids`, `attention_mask`, `labels`.
- Use an explicit prompt template with a separator (e.g. `"{prompt}\n### Response:\n"`), and append `eos_token` to the target.
- Set `labels[:prompt_len] = -100`.
- Handle: BPE boundary check, prompt-first truncation, dropping samples whose target would be truncated, GPT-2 pad-token workaround (`pad_token = eos_token` while masking pad labels).
- Write a **collator** that pads `input_ids` with pad id, `labels` with -100, and `attention_mask` with 0.
- Tests: label count equals target token count + 1 (EOS); the first non-masked label is the first target token; no masked tokens inside the target span.
- **Done when:** masking tests pass for all 3 tokenizers (GPT-2, Qwen, Phi-3).

### Task 1.4 — Reproduce the memorization failure (on purpose)
This is a research artifact, not a mistake. It gives you a "before" to compare against.
- Train naive mixed-dataset SFT (no masking, no replay) on the 35 samples for a very large step count.
- Log loss per step; confirm the collapse (the reference run: ~56K steps, loss 8.78 → 0.0001).
- Run the **memorization probe:** greedy-decode each training prompt and compute the exact-match rate against the target. Record it.
- Save the CSV and a plot to `docs/figures/memorization_baseline.png`.
- **Done when:** you have a logged baseline showing loss ≈ 0 plus verbatim reproduction.

### Task 1.5 — QLoRA training loop
- Implement `trainer/train_sft.py`:
  - `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=float16)`.
  - `prepare_model_for_kbit_training`, `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=<per model>)`.
  - AdamW lr=2e-4, linear warmup 5–10%, gradient clipping 1.0, gradient accumulation to an effective batch of 8–16.
  - Gradient checkpointing on, `use_cache=False`.
- Implement a custom or HF `Trainer` loop; either works, but you must log per step.
- Add **VRAM guards:** log `torch.cuda.max_memory_allocated()` each N steps; abort cleanly with a useful message on OOM.
- **Done when:** a single-stage run completes on GPT-2 and on Qwen2.5-1.5B within 4GB.

### Task 1.6 — Replay buffer and curriculum scheduler
- Implement `trainer/replay_buffer.py` (README §7.5): pre-tokenized samples held in RAM, per-stage storage, `blend(current_stage, samples, ratio)`, even sampling across earlier stages, seeded RNG.
- Implement the scheduler: for stage *k* in order → load stage data → blend replay using the stage ratio (0.0–0.4) → train for configured epochs → evaluate on probes of stages ≤ *k* → save checkpoint → move on.
- Benchmark blending vs. a disk-reload baseline. Target: ~4.5 s → < 0.05 s.
- Tests: ratio math, empty pool at stage 1, determinism, no stage-*k* samples duplicated into replay.
- **Done when:** the full 5-stage curriculum runs end to end and the timing benchmark is recorded.

### Task 1.7 — Logging, checkpoint tracking, early stopping
- `callbacks.py`: write `train_steps.csv` (step, epoch, stage, lr, loss, grad_norm, tokens_seen, gpu_mem_mb, wall_time_s) and `eval_steps.csv`.
- **Best-checkpoint tracker:** save `checkpoint_best/` only when held-out probe reward improves; keep `checkpoint_last/` for resume.
- Early stopping on held-out reward with patience, plus a `max_steps_per_stage` cap.
- Write `run.json` (git SHA, config hash, seeds, dataset hashes, library versions).
- **Done when:** a killed run can resume from `checkpoint_last/` and the best checkpoint is demonstrably chosen by probe reward, not train loss.

### Task 1.8 — Phase-1 evaluation (lightweight)
- Before the full harness exists, use a simple scorer: exact match/edit distance for grammar/punctuation, ROUGE-L for summarization, rule checks for conversation.
- Compare four arms on the same data/compute: **(a) mixed, no masking; (b) mixed + masking; (c) curriculum, no replay; (d) curriculum + replay + masking.** Run 3 seeds if time allows.
- Record the results in a table in `docs/phase1_results.md`.

## 1.4 Files produced

`configs/*.yaml`, `data/curriculum/*.jsonl`, `data/probes/*.jsonl`, `trainer/{dataset,replay_buffer,train_sft,callbacks}.py`, `logs/*/train_steps.csv`, `docs/figures/*`, `docs/phase1_results.md`, tests for masking/replay/dataset.

## 1.5 Acceptance criteria (exit gate)

- [ ] Full curriculum trains on 4GB VRAM without OOM.
- [ ] No verbatim memorization on probes (exact-match-to-training rate near 0 on held-out prompts, and documented for training prompts).
- [ ] Masking verified by tests; replay blending < 0.05 s per stage transition.
- [ ] Every run produces CSV logs and run metadata; best checkpoint selected by held-out reward.
- [ ] A written comparison of the four arms exists.

## 1.6 Risks and mitigations

| Risk | Mitigation |
|---|---|
| 35 samples are too few to learn anything | That is expected. Phase 2 grows the data. Use Phase 1 to prove *mechanics*, and keep probe banks tiny but clean |
| OOM with Phi-3-mini | Start with GPT-2/Qwen; Phi-3 only at batch 1, max_len ≤ 512 |
| Tokenizer boundary bugs | Boundary-equality assertion in the dataset builder |
| Overfitting hidden by training loss | Select checkpoints on probe reward only |

---

# PHASE 2 — RLAIF Flywheel: Critic → Generator → Quality Gate → Retrain

## 2.1 Objective

Build the **closed loop** in which a 70B teacher critiques the student, failures are converted into **targeted synthetic data requests**, a local generator produces data, a **7-layer gate** filters it, and the student retrains, growing the corpus from 35 to **285+ samples** with no poisoning.

## 2.2 Entry criteria

- Phase 1 exit gate passed; a working `checkpoint_best`.
- Groq API key available; Ollama installed with a generator model pulled.
- Probe banks in place and leak-tested.

## 2.3 Tasks

### Task 2.1 — Probe runner
- `rlaif/probe.py`: load the checkpoint (4-bit base + LoRA adapter), run fixed-decoding generation over the stage probe bank, save `{probe_id, prompt, response, latency}` to `logs/<run>/iter_<n>/probes.jsonl`.
- Free GPU memory afterwards (`del model; gc.collect(); torch.cuda.empty_cache()`).

### Task 2.2 — Failure taxonomy
- `rlaif/taxonomy.py`: define `Severity` (CRITICAL/HIGH/MEDIUM/LOW) and 7 `FailureCategory` values. Suggested set: repetition, off-topic, instruction-ignored, grammar error, punctuation error, hallucination, format violation. Adapt the names to your actual taxonomy.
- Define the mapping from category → stage → "data request template".
- Document every category with a one-line definition and one bad example in `docs/taxonomy.md`.

### Task 2.3 — Teacher critic client
- `rlaif/critic.py`: Groq client for `llama-3-70b` (verify the current model identifier on Groq's model list).
- System prompt = rubric + taxonomy + JSON-only instruction; low temperature.
- Pydantic v2 models: `Failure`, `Critique` (README §8.2).
- Robust parsing pipeline: **sanitize `<think>` → strip code fences → extract the first JSON object → validate**. On a validation error, re-ask once with the error text, then skip and log.
- Retries with exponential backoff (`tenacity`), rate limiter, and a disk cache keyed by a hash of (prompt, response, rubric version).
- **Done when:** 50 probe responses are critiqued with ≥ 95% parse success and 0 `<think>` leakage.

### Task 2.4 — `<think>` sanitization
- `rlaif/sanitize.py` with a regex for complete blocks plus a cleanup for stray/unclosed tags.
- Tests: nested, unclosed, multiple blocks, tags within JSON strings, uppercase tags, and empty think blocks.
- Add a **regression test showing the false-positive**: a critic output whose `<think>` text mentions "repetition" must *not* produce a repetition failure.

### Task 2.5 — Severity gate
- `rlaif/gate.py`: pass only failures with severity ∈ {CRITICAL, HIGH} **and** `priority ≤ 5`.
- Aggregate surviving failures by `(stage, category)`, count occurrences, and keep 2–3 representative bad outputs per group.
- Enforce a per-iteration request budget.
- Tests: boundary priority 5 vs. 6, all-LOW input yields zero requests, and ordering by priority then count.

### Task 2.6 — Targeted prompt builder
- `rlaif/prompt_builder.py`: template with stage task definition, failure pattern description, 2–3 good exemplars, 1–2 bad exemplars (from real failures), required output format (JSON lines with `prompt` and `target`), sample count, and a diversity instruction.
- Guard: assert no probe prompt appears in the builder's exemplars (protects held-out integrity).

### Task 2.7 — Generator with failover
- `rlaif/generator.py`: Ollama client (`qwen2.5-coder:7b`, or a smaller tag if VRAM-bound).
- **Explicit timeouts** (connect ≈ 3 s, read ≈ 60 s) and a health probe via `GET /api/tags` before each batch.
- Failover chain: primary → smaller local model → teacher API (tagged `source=teacher`) → skip and log.
- GPU handoff: unload the generator (`keep_alive: 0`) before training, and free the student's VRAM before generating.
- Tests with a fake server: connection refused, hang (timeout), malformed output, recovery after failure. Target: per-batch hang time ~200 s → ~0 s.

### Task 2.8 — 7-layer validator
- `validator/layers.py` and `validator/pipeline.py`, with reason codes (README §8.7):
  - **L1** schema; **L2** min length 10; **L3** max length 4096; **L4** bigram repetition ratio < 0.35; **L5** identity (case-insensitive input ≡ target); **L6** Jaccard circularity < 0.85; **L7** SHA-256 dedup on normalized `prompt + target`.
- Run cheapest-first with short-circuit. Normalize text before hashing and before Jaccard.
- Thread-safe dedup (`threading.Lock`), and a test with 10 workers.
- Persist rejects to `data/rejected/<iter>.jsonl` with the reason code; emit per-layer counters.
- Performance tests: 500 samples < 9 ms; 1,000 < 250 ms. Use generous margins in CI to avoid flaky timing.

### Task 2.9 — Loop controller
- `rlaif/controller.py` and `scripts/rl_loop.py` implementing the loop in README §8.8: load best checkpoint → probe → critique → gate → build prompts → generate → validate → append → retrain → evaluate → regress check → report.
- Dataset **versioning:** write each iteration's accepted samples to `data/synthetic/iter_<n>.jsonl`; never mutate older files, so a rollback is deleting one file.
- Stop conditions: converged, starved (too few accepted samples), regression streak, max iterations, API budget.
- Per-iteration report (JSON + markdown): failure counts by category, requests made, accepted/rejected per layer, new dataset size, reward before/after, wall time.
- **Done when:** a multi-iteration run completes unattended, and the dataset grows from 35 to 285+ samples.

### Task 2.10 — Verification and baselines
- Verification run: confirm reward **r = 1.000 across all 5 stages** on the stage probe banks. Also record probe-bank sizes.
- Run the **RQ1 comparison:** the targeted flywheel vs. random-prompt synthetic generation, at an equal accepted-sample budget.
- Run the **RQ3 gate ablation:** disable one layer at a time for ≥ 10 iterations. Track distinct-2, repetition ratio, and reward.

## 2.4 Files produced

`rlaif/*.py`, `validator/*.py`, `scripts/rl_loop.py`, `data/synthetic/iter_*.jsonl`, `data/rejected/*.jsonl`, `logs/<run>/iter_*/report.{json,md}`, `docs/taxonomy.md`, `docs/phase2_results.md`.

## 2.5 Acceptance criteria (exit gate)

- [ ] Unattended multi-iteration loop runs with failover tested (kill Ollama mid-run; the loop continues).
- [ ] Critic parse success ≥ 95%, with zero `<think>` leakage in parsed findings.
- [ ] Validator passes its tests, including the concurrency and throughput targets.
- [ ] Dataset grew to 285+ samples; every accepted sample has provenance metadata.
- [ ] Reward trend is logged per iteration; at least one regression rollback has been exercised (even if staged in a test).
- [ ] RQ1 and RQ3 experiments are recorded, including negative findings.

## 2.6 Risks and mitigations

| Risk | Mitigation |
|---|---|
| Data poisoning / mode collapse | All 7 layers on; watch per-layer reject spikes; monitor distinct-2 per iteration |
| Generator too weak (a 7B model on 4GB is slow or heavily quantized) | Smaller model + stronger prompts, or use the teacher for generation with `source=teacher` |
| Teacher API cost/limits | Cache, batch, token-bucket, cap requests per iteration |
| Reward inflation (probe leakage) | Leakage test in CI; fresh probes added each cycle |
| Loop regression | Versioned datasets + automatic rollback on > tolerance drop |
| GPU contention (train vs. generate) | Strict serialization in the controller |

---

# PHASE 3 — Evaluation, Hardening, Infrastructure, and Publication

## 3.1 Objective

Replace ad-hoc scoring with a **tiered evaluation harness**, validate the judge against humans (**Cohen's κ**), lock behavior with a **95-test regression suite**, package everything in **Docker/Kubernetes**, and publish the results in a form a research reviewer can trust.

## 3.2 Entry criteria

- Phase 2 exit gate passed.
- A labeled set can be produced (probe responses from the Phase 2 iterations).

## 3.3 Tasks

### Task 3.1 — Local evaluator
- `eval/local_evaluator.py` with pure-Python signals: **TTR**, **n-gram windowing** (max repeated n-gram frequency for n = 2–4), **Jaccard** vs. reference, and stage-specific rule checks (terminal punctuation, non-empty, length band).
- Combine into a score in [0, 1] using weights from `eval.yaml`. **Fit the weights** on the Task 3.4 labeled set rather than guessing.
- Benchmark with many repeats; target **0.013 ms/probe** (50 probes in ~0.64 ms). Report median and p95.
- Tests: range, determinism, empty input, very long input, unicode.

### Task 3.2 — Teacher judge
- `eval/teacher_judge.py`: Groq 70B, structured JSON rubric, **4 axes** scored 1–5 and normalized.
- Implement **Rubric A** (plain) and **Rubric B** (chain-of-thought + exemplars). Put the justification field *before* the score field in the schema.
- Bias controls: randomize block order, instruct the judge to ignore length, and fix the temperature low.
- Store `rubric_version` with every judgment.

### Task 3.3 — Tiered router and reward scalar
- `eval/router.py`: `local_score ≥ 0.70 → fast pass`, else escalate to the teacher.
- `eval/reward.py`: per-stage reward in [0, 1] = mean probe score; also `min_stage_reward` and the macro mean.
- Log: local-handled fraction, teacher calls, estimated time saved, and (on the labeled set) **false-pass rate**.
- Tests: threshold boundary at exactly 0.70, escalation counting, and teacher-failure fallback (if the API is down, mark the probe "unjudged", never silently pass it).

### Task 3.4 — Judge–human agreement study
1. Build a **stratified sample of ~100** (probe, response) pairs across stages and quality levels.
2. Label them blind, yourself and ideally a second annotator, using a written guideline.
3. Run Rubric A and Rubric B.
4. Compute Cohen's κ (`sklearn.metrics.cohen_kappa_score`; `weights="quadratic"` for ordinal scores) with a bootstrap 95% CI. Reference results: **0.66 (A) → 0.77 (B)**.
5. Review the disagreements and iterate the rubric. Document 5–10 representative cases in `docs/judge_agreement.md`.
6. Also compute local-evaluator vs. teacher correlation (Pearson/Spearman) and the routing threshold sweep (0.5–0.9): cost saved vs. false-pass rate (**RQ4**).

### Task 3.5 — Reference metrics
- `eval/metrics.py`: ROUGE-L (summarization), BERTScore (semantic similarity), exact match/edit distance (grammar/punctuation).
- Use these as supporting evidence next to the judge scores. Do not let them replace the judge.

### Task 3.6 — Stage eval CLI and reports
- `eval/run_stage_eval.py --stage N --checkpoint PATH` → `eval_report.json`, `eval_report.md`, per-probe CSV.
- Add `--compare PATH_A PATH_B` to produce a diff between two checkpoints.

### Task 3.7 — Test suite to 95 tests
- Grow the suite over **4 stress rounds**, each hunting a different class of bug: (1) correctness edges, (2) malformed external data, (3) concurrency and timing, (4) failure injection (network, disk, OOM simulation).
- Use the area table in README §10 as the checklist; mock network by default, mark live-API tests with `@pytest.mark.network`.
- Add coverage reporting and a **probe-leakage test** and a **config-schema test**.
- CI: GitHub Actions workflow running `pytest tests/ -v --tb=short` on push and PR.
- **Done when:** all 95 tests pass (100%) on a clean clone.

### Task 3.8 — Docker and Kubernetes
- **Dockerfile (multi-stage):** builder installs wheels, runtime is slim, runs non-root. Two targets: `trainer` (CUDA base) and `evaluator` (CPU).
- **docker-compose:** `trainer`, `ollama`, `evaluator`, with mounted `data/`, `logs/`, `checkpoints/`.
- **K8s:** `Deployment` for the evaluator/validator service, **HPA 1→4 replicas** (CPU target), `Secret` for the Groq key, `ConfigMap` for configs, readiness/liveness on `/healthz`, explicit resource requests/limits. Training stays a local single-GPU job or a K8s `Job`.
- Smoke-test: `docker compose up` then run the eval CLI inside the container; `kubectl apply -f k8s/` on a local cluster (kind or minikube) and verify scale-out under synthetic load.

### Task 3.9 — Full ablation and benchmark publication
- Run the experiments in README §13 that remain: curriculum vs. mixed vs. curriculum+replay (3 seeds), masking on/off, rubric A vs. B, threshold sweep, and model scaling GPT-2 → Qwen2.5-1.5B → Phi-3-mini.
- Publish a **per-stage held-out probe pass-rate table** (the roadmap item), using a *larger* held-out set than the loop's own probes, so the 1.000 reward is put in context.
- Produce plots: loss curves, per-stage reward vs. iteration, reject counts per validator layer, κ comparison, cost-vs-false-pass curve.

### Task 3.10 — Documentation and write-up
- Finalize `README.md` (results table filled from real logs, with confidence intervals where available).
- Complete `docs/FAILURES_AND_OPTIMIZATIONS.md` (symptom → cause → fix → measured effect) for every item in the README §15 table.
- Write a **2–4 page technical report** answering RQ1–RQ4 with data, including negative or partial findings, and a limitations section.
- Record a short demo (terminal capture) of one loop iteration.
- Update resume bullets with final, verified numbers only.

### Task 3.11 — Forward work (optional stretch, after the core is done)
- **DPO** on Llama 3.1-8B: build chosen/rejected pairs from critic pairwise judgments; train with QLoRA (likely needs cloud GPU).
- **GRPO** reward shaping on math/reasoning tasks with verifiable rewards.
- **Weights & Biases** integration for run tracking and sweeps.

## 3.4 Files produced

`eval/*.py`, `tests/*` (95 tests), `.github/workflows/ci.yml`, `docker/*`, `k8s/*`, `docs/judge_agreement.md`, `docs/FAILURES_AND_OPTIMIZATIONS.md`, `docs/report.pdf` (or `.md`), final plots, final `README.md`.

## 3.5 Acceptance criteria (final definition of done)

- [ ] Local evaluator ≤ ~0.013 ms/probe (median), with a benchmark script checked in.
- [ ] Router false-pass rate measured and reported; the threshold choice is justified by the sweep.
- [ ] Cohen's κ reported with sample size and CI (reference: 0.77 for Rubric B).
- [ ] 95 tests, 100% pass, running in CI from a clean clone.
- [ ] `docker compose up` works; K8s manifests apply cleanly and the HPA scales 1→4 under load.
- [ ] Held-out benchmark table published; RQ1–RQ4 each have a written answer.
- [ ] Every number on the resume traces to a log or test in the repo.

## 3.6 Risks and mitigations

| Risk | Mitigation |
|---|---|
| κ unstable on a small sample | Report CI, add a second annotator, enlarge the sample if time allows |
| Local evaluator passes bad outputs | Add rule checks per stage, tune weights on labeled data, add a hard-fail floor |
| Flaky timing tests in CI | Wide margins; separate benchmark scripts from pass/fail tests |
| Judge model deprecation on Groq | Make `teacher_model` a config value and keep the rubric version-stamped |
| Scope creep (DPO/GRPO) | Treat as post-core stretch; ship Phases 1–3 first |

---

# Cross-Phase Checklist

| Concern | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Reproducibility | seeds, run.json | dataset versioning | pinned deps, Docker, CI |
| Logging | per-step CSV | per-iteration reports | eval reports, plots |
| Held-out integrity | probe banks + leakage test | no probe in generator prompts | larger held-out benchmark |
| Failure handling | OOM guard | failover, rollback | failure injection tests |
| Evidence for claims | loss/forgetting tables | RQ1, RQ3 results | RQ2, RQ4, κ, benchmark table |

# Suggested Timeline (part-time, ~10–15 h/week)

| Week | Focus |
|---|---|
| 1 | Phase 1 tasks 1.1–1.4: skeleton, data, masking, memorization baseline |
| 2 | Phase 1 tasks 1.5–1.8: QLoRA loop, replay buffer, logging, 4-arm comparison |
| 3 | Phase 2 tasks 2.1–2.5: probe runner, taxonomy, critic, sanitizer, gate |
| 4 | Phase 2 tasks 2.6–2.8: prompt builder, generator failover, 7-layer validator |
| 5 | Phase 2 tasks 2.9–2.10: loop controller, verification, RQ1/RQ3 runs |
| 6 | Phase 3 tasks 3.1–3.4: local evaluator, judge, router, κ study |
| 7 | Phase 3 tasks 3.5–3.8: metrics, 95-test suite, CI, Docker/K8s |
| 8 | Phase 3 tasks 3.9–3.10: ablations, benchmark table, write-up, resume update |
