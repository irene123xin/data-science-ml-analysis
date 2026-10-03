"""Ranking metrics used by the MovieLens offline evaluation.

The functions accept ranked item identifiers and a set of relevant items. They
do not fit a recommender or assume a particular relevance threshold.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Hashable

Item = Hashable


def _top_k(recommended: Sequence[Item], k: int) -> list[Item]:
    if k <= 0:
        raise ValueError("k must be a positive integer")
    return list(recommended[:k])


def precision_at_k(recommended: Sequence[Item], relevant: Iterable[Item], k: int) -> float:
    """Return the fraction of the first *k* recommendations that are relevant."""
    ranked = _top_k(recommended, k)
    relevant_set = set(relevant)
    return sum(item in relevant_set for item in ranked) / k


def recall_at_k(recommended: Sequence[Item], relevant: Iterable[Item], k: int) -> float:
    """Return the fraction of relevant items recovered in the first *k* positions."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ranked = _top_k(recommended, k)
    return len(set(ranked).intersection(relevant_set)) / len(relevant_set)


def ndcg_at_k(recommended: Sequence[Item], relevant: Iterable[Item], k: int) -> float:
    """Return binary normalized discounted cumulative gain at *k*."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ranked = _top_k(recommended, k)
    dcg = sum((1.0 / math.log2(rank + 2)) for rank, item in enumerate(ranked) if item in relevant_set)
    ideal_hits = min(len(relevant_set), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal_dcg


def hit_rate_at_k(recommended: Sequence[Item], relevant: Iterable[Item], k: int) -> float:
    """Return 1 when at least one relevant item appears in the first *k*, otherwise 0."""
    relevant_set = set(relevant)
    return float(any(item in relevant_set for item in _top_k(recommended, k)))


def evaluate_rankings(
    recommendations: dict[Item, Sequence[Item]],
    relevant_items: dict[Item, Iterable[Item]],
    k: int,
) -> dict[str, float]:
    """Macro-average ranking metrics across users present in both mappings."""
    users = sorted(set(recommendations).intersection(relevant_items), key=str)
    if not users:
        raise ValueError("no users have both recommendations and relevant items")
    rows = [
        (
            precision_at_k(recommendations[user], relevant_items[user], k),
            recall_at_k(recommendations[user], relevant_items[user], k),
            ndcg_at_k(recommendations[user], relevant_items[user], k),
            hit_rate_at_k(recommendations[user], relevant_items[user], k),
        )
        for user in users
    ]
    return {
        "Precision@K": sum(row[0] for row in rows) / len(rows),
        "Recall@K": sum(row[1] for row in rows) / len(rows),
        "NDCG@K": sum(row[2] for row in rows) / len(rows),
        "HitRate@K": sum(row[3] for row in rows) / len(rows),
        "Eligible users": float(len(users)),
    }
