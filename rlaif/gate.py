"""rlaif/gate.py — Task 2.5: Severity gate.

Filters and aggregates failures discovered across probe evaluations:
- Passes only failures with severity in {CRITICAL, HIGH} AND priority <= 5.
- Aggregates surviving failures by (stage, category).
- Selects representative exemplar failures.
- Enforces per-iteration request budgets.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from pydantic import BaseModel

from rlaif.critic import FailureItem
from rlaif.taxonomy import FailureCategory, Severity


class TargetedDataRequest(BaseModel):
    stage: int
    category: FailureCategory
    count_observed: int
    priority: int
    sample_bad_outputs: List[str]
    num_samples_to_generate: int = 5


def filter_and_aggregate_failures(
    stage_failures: List[Tuple[int, FailureItem, str]],
    budget: int = 15,
) -> List[TargetedDataRequest]:
    """Filter failures through severity gate and aggregate by (stage, category).

    Args:
        stage_failures: list of (stage_id, FailureItem, student_bad_output)
        budget: maximum number of distinct targeted requests allowed
    """
    # 1. Severity Gate: Severity in {CRITICAL, HIGH} AND priority <= 5
    surviving = []
    for stage_id, failure, bad_output in stage_failures:
        if failure.severity in (Severity.CRITICAL, Severity.HIGH) and failure.priority <= 5:
            surviving.append((stage_id, failure, bad_output))

    if not surviving:
        return []

    # 2. Group by (stage, category)
    grouped: Dict[Tuple[int, FailureCategory], List[Tuple[int, str]]] = defaultdict(list)
    for stage_id, failure, bad_output in surviving:
        grouped[(stage_id, failure.category)].append((failure.priority, bad_output))

    requests: List[TargetedDataRequest] = []
    for (st, cat), items in grouped.items():
        min_prio = min(item[0] for item in items)
        samples = [item[1] for item in items if item[1]][:3]  # up to 3 exemplars
        requests.append(
            TargetedDataRequest(
                stage=st,
                category=cat,
                count_observed=len(items),
                priority=min_prio,
                sample_bad_outputs=samples,
                num_samples_to_generate=5,
            )
        )

    # 3. Sort by priority (ascending: 1 is highest priority), then count (descending)
    requests.sort(key=lambda r: (r.priority, -r.count_observed))

    return requests[:budget]
