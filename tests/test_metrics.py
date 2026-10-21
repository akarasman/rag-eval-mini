"""Tests for retrieval metrics, using small rankings whose answers are worked out by hand.

A query with no relevant docs raises ValueError; the caller (evaluate.py) must skip such queries.
Not covered yet: repeated doc IDs in a ranking (behaviour still undecided).
"""
from math import log2

import pytest

from rag_eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


# --- query with no relevant docs (all metrics refuse to score it) -----------

@pytest.mark.parametrize("relevant", [{}, {"a": 0}])
@pytest.mark.parametrize(
    "metric",
    [
        lambda ranked, relevant: recall_at_k(ranked, relevant, 5),
        lambda ranked, relevant: reciprocal_rank(ranked, relevant),
        lambda ranked, relevant: ndcg_at_k(ranked, relevant, 5),
    ],
    ids=["recall", "reciprocal_rank", "ndcg"],
)
def test_metrics_raise_when_query_has_no_relevant_docs(metric, relevant):
    with pytest.raises(ValueError, match="no relevant docs"):
        metric(["a", "b"], relevant)


# --- recall@k ---------------------------------------------------------------

@pytest.mark.parametrize(
    "ranked, relevant, k, expected",
    [
        (["a", "b", "c"], {"a": 1, "c": 1}, 1, 0.5),    # only "a" in top 1 -> 1 of 2
        (["a", "b", "c"], {"a": 1, "c": 1}, 2, 0.5),    # "c" is at rank 3, still outside
        (["a", "b", "c"], {"a": 1, "c": 1}, 3, 1.0),    # both found
        (["x", "y"], {"a": 1}, 5, 0.0),                 # relevant doc never retrieved
        (["a"], {"a": 1}, 10, 1.0),                     # k larger than the ranking
        ([], {"a": 1}, 3, 0.0),                         # empty ranking
        (["a", "b"], {"a": 2, "b": 1}, 1, 0.5),         # graded: counts docs, not grades
        (["a", "b"], {"a": 1, "b": 0}, 2, 1.0),         # grade 0 is not relevant -> 1 of 1
    ],
)
def test_recall_at_k(ranked, relevant, k, expected):
    assert recall_at_k(ranked, relevant, k) == pytest.approx(expected)


def test_recall_never_decreases_as_k_grows():
    ranked = ["x", "a", "y", "b", "z"]
    relevant = {"a": 1, "b": 1}
    values = [recall_at_k(ranked, relevant, k) for k in range(1, 8)]
    assert values == sorted(values)


@pytest.mark.parametrize("k", [0, -1])
def test_recall_rejects_non_positive_k(k):
    with pytest.raises(ValueError, match="k"):
        recall_at_k(["a"], {"a": 1}, k)


# --- reciprocal rank --------------------------------------------------------

@pytest.mark.parametrize(
    "ranked, relevant, k, expected",
    [
        (["a", "x", "y"], {"a": 1}, None, 1.0),         # rank 1
        (["x", "a", "y"], {"a": 1}, None, 0.5),         # rank 2
        (["x", "y", "a"], {"a": 1}, None, 1 / 3),       # rank 3
        (["x", "y"], {"a": 1}, None, 0.0),              # absent
        (["x", "y", "a"], {"a": 1}, 2, 0.0),            # rank 3 but cut off at k=2
        (["x", "y", "a"], {"a": 1}, 3, 1 / 3),          # cutoff exactly at the hit
        (["x", "a", "b"], {"a": 1, "b": 1}, None, 0.5), # first relevant doc counts
        (["a", "b"], {"a": 0, "b": 1}, None, 0.5),      # grade 0 is skipped
        ([], {"a": 1}, None, 0.0),                      # empty ranking
    ],
)
def test_reciprocal_rank(ranked, relevant, k, expected):
    assert reciprocal_rank(ranked, relevant, k) == pytest.approx(expected)


def test_reciprocal_rank_default_k_means_no_cutoff():
    assert reciprocal_rank(["x"] * 9 + ["a"], {"a": 1}) == pytest.approx(0.1)


@pytest.mark.parametrize("k", [0, -1])
def test_reciprocal_rank_rejects_non_positive_k(k):
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank(["a"], {"a": 1}, k)


# --- nDCG@k -----------------------------------------------------------------

def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a"], {"a": 1}, 5) == pytest.approx(1.0)
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2) == pytest.approx(1.0)  # best grade first


def test_ndcg_single_relevant_doc_at_lower_ranks():
    # IDCG = 1 (gain 1 at rank 1), so nDCG = 1 / log2(rank + 1)
    assert ndcg_at_k(["x", "a"], {"a": 1}, 5) == pytest.approx(1 / log2(3))   # ~0.6309
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, 5) == pytest.approx(0.5)      # 1 / log2(4)


def test_ndcg_absent_or_cut_off_is_zero():
    assert ndcg_at_k(["x", "y"], {"a": 1}, 5) == 0.0
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}, 2) == 0.0


def test_ndcg_graded_reversed_ranking():
    # grades: a=2, b=1; ranking [b, a], k=2; gain = 2^rel - 1
    # DCG  = 1/log2(2) + 3/log2(3) = 1 + 1.8928 = 2.8928
    # IDCG = 3/log2(2) + 1/log2(3) = 3 + 0.6309 = 3.6309
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2) == pytest.approx(0.7967, abs=1e-4)


def test_ndcg_two_relevant_docs_with_a_gap():
    # ranking [a, x, b], both relevant (grade 1)
    # DCG  = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    assert ndcg_at_k(["a", "x", "b"], {"a": 1, "b": 1}, 3) == pytest.approx(0.9197, abs=1e-4)


def test_ndcg_ideal_dcg_comes_from_qrels_not_from_the_ranking():
    # Only one of two relevant docs retrieved: DCG = 1, IDCG = 1 + 0.6309, so it must not be 1.0
    assert ndcg_at_k(["a"], {"a": 1, "b": 1}, 5) == pytest.approx(0.6131, abs=1e-4)


def test_ndcg_k_larger_than_ranking():
    assert ndcg_at_k(["a"], {"a": 1}, 100) == pytest.approx(1.0)


@pytest.mark.parametrize("ranked", [["a", "b", "c"], ["c", "b", "a"], ["x", "a"], []])
def test_ndcg_stays_between_zero_and_one(ranked):
    assert 0.0 <= ndcg_at_k(ranked, {"a": 2, "b": 1}, 3) <= 1.0


@pytest.mark.parametrize("k", [0, -1])
def test_ndcg_rejects_non_positive_k(k):
    with pytest.raises(ValueError, match="k"):
        ndcg_at_k(["a"], {"a": 1}, k)
