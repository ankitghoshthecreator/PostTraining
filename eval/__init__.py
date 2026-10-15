"""eval package for tiered evaluation harness, local evaluator, and teacher judge."""

from eval.local_evaluator import LocalEvaluator
from eval.teacher_judge import TeacherJudge, JudgeResult
from eval.router import TieredRouter
from eval.reward import RewardCalculator
from eval.agreement import compute_cohen_kappa, compute_correlation
from eval.metrics import compute_rouge_l, compute_exact_match, compute_normalized_edit_similarity

__all__ = [
    "LocalEvaluator",
    "TeacherJudge",
    "JudgeResult",
    "TieredRouter",
    "RewardCalculator",
    "compute_cohen_kappa",
    "compute_correlation",
    "compute_rouge_l",
    "compute_exact_match",
    "compute_normalized_edit_similarity",
]
