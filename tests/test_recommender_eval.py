import math

import pytest

from src.recommender_eval import (
    evaluate_rankings,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_binary_ranking_metrics() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert precision_at_k(ranked, relevant, 3) == pytest.approx(1 / 3)
    assert recall_at_k(ranked, relevant, 3) == pytest.approx(1 / 2)
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert hit_rate_at_k(ranked, relevant, 3) == 1.0


def test_empty_relevance_and_invalid_k() -> None:
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert ndcg_at_k([1, 2], set(), 2) == 0.0
    assert hit_rate_at_k([1, 2], set(), 2) == 0.0
    with pytest.raises(ValueError):
        precision_at_k([1], {1}, 0)


def test_evaluate_rankings_macro_average() -> None:
    summary = evaluate_rankings({1: [10, 11], 2: [20, 21]}, {1: {10}, 2: {99}}, k=2)
    assert summary["Precision@K"] == pytest.approx(0.25)
    assert summary["Recall@K"] == pytest.approx(0.5)
    assert summary["HitRate@K"] == pytest.approx(0.5)
    assert summary["Eligible users"] == 2.0
