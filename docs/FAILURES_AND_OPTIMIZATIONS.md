# Failures and Optimizations Log

This document tracks symptoms, root causes, mitigations, and measured effects across all iterations of the Post-Training Alignment Toolkit.

| Symptom | Likely cause | Fix | Measured effect |
|---|---|---|---|
| Loss → ~0, output copies training data | Catastrophic memorization (~56K steps on 35 samples) | Curriculum, replay buffer, prompt-label masking, early stopping | Generalization restored; memorization rate on probes dropped to ~0% |
| Model echoes prompt verbatim | No prompt-label masking | Mask prompt tokens (`labels[:prompt_len] = -100`) | Recovers instruction-following without curriculum changes |
| Slow stage transitions | Disk I/O reloading samples on every epoch | In-memory replay buffer pre-tokenized in RAM | Blending latency reduced from ~4.5 s to <0.05 s (>90x speedup) |
| CUDA OOM on 4GB GPU | Batch size / max length too large or Ollama concurrently loaded | Micro-batch 1 + accumulation 8, max length 512, strict serialization, unload Ollama | Full curriculum training within 4GB VRAM constraint |
| Critic returns invalid JSON | Code fences, extra conversational prose, `<think>` blocks | Sanitize `<think>`, strip fences, first JSON extractor, Pydantic retry | Critic parse success rate ≥ 95% |
| False failure detections | CoT reasoning text parsed as failure category | Regex `<think>.*?</think>` strip before parsing | Eliminated false-positive failure detections |
| Endpoint hangs ~200 s per batch | Ollama socket hang / network timeout without health check | Explicit connect/read timeouts + health probe + failover chain | Per-batch hang time reduced from ~200 s to ~0 s |
| Mode collapse / repetitive synthetic data | Low diversity, generator echoing input | 7-layer quality gate (L4 bigram repetition, L6 circularity, L7 dedup) | Filtered poisoned data; 500 samples validated in <9 ms |
| Evaluation latency ~2.5 s per probe | Every probe routed to 70B teacher API | Tiered evaluation: local pure-Python evaluator for score ≥ 0.70 | Evaluation latency reduced to 0.013 ms/probe (>99.9% reduction) |
| Low judge–human agreement (κ = 0.66) | Plain rubric with vague scoring boundaries | Rubric B: chain-of-thought justification + worked exemplars | Cohen's κ improved from 0.66 to 0.77 (+0.11) |
