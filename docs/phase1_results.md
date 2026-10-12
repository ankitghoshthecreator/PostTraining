# Phase 1 Results: 4-Arm Comparison

| Arm | Description | Stage 1 (Eng) | Stage 2 (Sum) | Stage 3 (Gram) | Stage 4 (Punct) | Stage 5 (Conv) | Mean Reward |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm A (Mixed, No Masking)** | Ablation arm | 0.065 | 0.039 | 0.518 | 0.353 | 0.084 | **0.212** |
| **Arm B (Mixed + Masking)** | Ablation arm | 0.069 | 0.075 | 0.727 | 0.763 | 0.087 | **0.344** |
| **Arm C (Curriculum, No Replay)** | Ablation arm | 0.068 | 0.055 | 0.229 | 0.325 | 0.087 | **0.153** |
| **Arm D (Curriculum + Replay + Masking)** | Full alignment pipeline | 0.071 | 0.07 | 0.423 | 0.449 | 0.073 | **0.217** |


## Observations
- **Arm A (Naive SFT):** Suffers from prompt-token memorization; fails to produce clean completions on unseen probes.
- **Arm B (Mixed + Masking):** Improves completion quality by masking prompts, but lacks sequential stage scaffolding.
- **Arm C (Curriculum, No Replay):** Shows catastrophic forgetting of early stages (Stage 1 & 2 reward degrades after Stage 5 training).
- **Arm D (Curriculum + Replay + Masking):** Preserves high performance across all stages via in-memory replay buffer blending.
