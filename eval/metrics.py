"""eval/metrics.py — Task 3.5: Standard Reference Metrics.

Reference and comparative NLP metrics:
- ROUGE-L: Longest Common Subsequence (LCS) F1 for summarization evaluation.
- Levenshtein Edit Distance & Normalized Edit Similarity for grammar and punctuation correction.
- Exact Match (EM): Binary exact string match.
- BLEU-1: Unigram token precision with brevity penalty.
"""

from __future__ import annotations

import re
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    """Tokenize into alphanumeric tokens and punctuation."""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def compute_exact_match(candidate: str, reference: str, case_sensitive: bool = False) -> float:
    """Exact string match score (1.0 for match, 0.0 otherwise)."""
    c = candidate.strip() if case_sensitive else candidate.strip().lower()
    r = reference.strip() if case_sensitive else reference.strip().lower()
    return 1.0 if c == r else 0.0


def compute_edit_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_normalized_edit_similarity(candidate: str, reference: str) -> float:
    """Normalized edit similarity in [0, 1].

    1.0 means identical, 0.0 means completely different.
    """
    max_len = max(len(candidate), len(reference))
    if max_len == 0:
        return 1.0
    dist = compute_edit_distance(candidate, reference)
    return max(0.0, 1.0 - (dist / max_len))


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Length of Longest Common Subsequence between token sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(candidate: str, reference: str) -> float:
    """Compute ROUGE-L F1 score based on token Longest Common Subsequence."""
    cand_tokens = tokenize(candidate)
    ref_tokens = tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(cand_tokens, ref_tokens)
    prec = lcs / len(cand_tokens)
    rec = lcs / len(ref_tokens)

    if prec + rec == 0:
        return 0.0

    f1 = (2 * prec * rec) / (prec + rec)
    return round(f1, 4)


def compute_bleu_1(candidate: str, reference: str) -> float:
    """Compute unigram BLEU-1 precision with brevity penalty."""
    c_tokens = tokenize(candidate)
    r_tokens = tokenize(reference)

    if not c_tokens or not r_tokens:
        return 0.0

    c_counts: dict[str, int] = {}
    for t in c_tokens:
        c_counts[t] = c_counts.get(t, 0) + 1

    r_counts: dict[str, int] = {}
    for t in r_tokens:
        r_counts[t] = r_counts.get(t, 0) + 1

    matches = sum(min(count, r_counts.get(token, 0)) for token, count in c_counts.items())
    precision = matches / len(c_tokens)

    # Brevity penalty
    c_len = len(c_tokens)
    r_len = len(r_tokens)
    if c_len > r_len:
        bp = 1.0
    else:
        import math
        bp = math.exp(1.0 - (r_len / max(1, c_len)))

    return round(bp * precision, 4)
