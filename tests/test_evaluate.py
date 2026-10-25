"""Tests for the evaluation loop, using fake retrievers whose rankings are known in advance."""
from typing import Dict, List

import pytest

import rag_eval.metrics as metrics_module
from rag_eval.evaluate import evaluate
from rag_eval.retrievers.base import Ranking
from rag_eval.retrievers.oracle import OracleRetriever


class _FixedRetriever:
    """Returns a preset ranking per query text and records the k of every search call."""
    name = "fixed"

    def __init__(self, rankings: Dict[str, List[str]]):
        self.rankings = rankings
        self.search_ks: List[int] = []

    def index(self, corpus) -> None:
        pass

    def search(self, query: str, k: int = 10) -> Ranking:
        self.search_ks.append(k)
        return [(doc_id, 1.0 / rank) for rank, doc_id in enumerate(self.rankings[query][:k], start=1)]


QUERIES = {"q1": "first", "q2": "second"}
QRELS = {"q1": {"a": 1, "b": 1}, "q2": {"c": 1}}
RANKINGS = {"first": ["x", "a", "y", "b"], "second": ["c", "x", "y", "z"]}


def test_evaluate_computes_known_values_per_query_and_in_the_mean():
    result = evaluate(_FixedRetriever(RANKINGS), QUERIES, QRELS, ks=(1, 2, 4))

    # q1: relevant docs sit at ranks 2 and 4. q2: the only relevant doc is at rank 1.
    assert result.per_query["q1"]["recall@1"] == 0.0
    assert result.per_query["q1"]["recall@2"] == pytest.approx(0.5)
    assert result.per_query["q1"]["recall@4"] == pytest.approx(1.0)
    assert result.per_query["q1"]["mrr@1"] == 0.0
    assert result.per_query["q1"]["mrr@2"] == pytest.approx(0.5)
    assert result.per_query["q1"]["ndcg@2"] == pytest.approx(0.386853, abs=1e-5)
    assert result.per_query["q2"]["recall@1"] == pytest.approx(1.0)
    assert result.per_query["q2"]["ndcg@4"] == pytest.approx(1.0)

    assert result.mean["recall@1"] == pytest.approx(0.5)
    assert result.mean["recall@2"] == pytest.approx(0.75)
    assert result.mean["mrr@4"] == pytest.approx(0.75)


def test_evaluate_reports_every_metric_for_every_k():
    result = evaluate(_FixedRetriever(RANKINGS), QUERIES, QRELS, ks=(1, 10))

    expected = {f"{metric}@{k}" for metric in ("recall", "mrr", "ndcg") for k in (1, 10)}
    assert set(result.mean) == expected
    assert all(set(scores) == expected for scores in result.per_query.values())


def test_evaluate_mean_is_the_average_of_the_per_query_values():
    result = evaluate(_FixedRetriever(RANKINGS), QUERIES, QRELS, ks=(1, 2, 4))

    for name, mean in result.mean.items():
        assert mean == pytest.approx(sum(scores[name] for scores in result.per_query.values()) / len(QUERIES))


def test_evaluate_searches_once_per_query_with_the_largest_k():
    retriever = _FixedRetriever(RANKINGS)

    evaluate(retriever, QUERIES, QRELS, ks=(10, 1, 5))

    assert retriever.search_ks == [10, 10]


def test_evaluate_counts_a_short_or_empty_ranking_as_misses():
    retriever = _FixedRetriever({"first": [], "second": ["c"]})

    result = evaluate(retriever, QUERIES, QRELS, ks=(1, 10))

    assert result.per_query["q1"] == {name: 0.0 for name in result.per_query["q1"]}
    assert result.per_query["q2"]["recall@10"] == pytest.approx(1.0)  # shorter than k but complete


def test_evaluate_scores_the_oracle_perfectly():
    corpus = {doc_id: {"title": "", "text": ""} for doc_id in "abcxyz"}
    oracle = OracleRetriever(QUERIES, QRELS)
    oracle.index(corpus)

    result = evaluate(oracle, QUERIES, QRELS, ks=(1, 2, 10))

    for k in (1, 2, 10):
        assert result.mean[f"mrr@{k}"] == pytest.approx(1.0)
        assert result.mean[f"ndcg@{k}"] == pytest.approx(1.0)
    assert result.mean["recall@10"] == pytest.approx(1.0)
    # q1 has two relevant docs, so recall@1 is capped at 1/2 for any retriever; q2 reaches 1.
    assert result.mean["recall@1"] == pytest.approx((0.5 + 1.0) / 2)


@pytest.mark.parametrize("qrels", [{"q1": {"a": 1}}, {"q1": {"a": 1}, "q2": {"c": 0}}, {"q1": {"a": 1}, "q2": {}}])
def test_evaluate_raises_for_a_query_without_relevant_docs(qrels):
    with pytest.raises(ValueError, match="q2"):
        evaluate(_FixedRetriever(RANKINGS), QUERIES, qrels)


def test_evaluate_validates_every_query_before_searching_any():
    retriever = _FixedRetriever(RANKINGS)

    with pytest.raises(ValueError):
        evaluate(retriever, QUERIES, {"q1": {"a": 1}})

    assert retriever.search_ks == []


def test_evaluate_raises_without_queries():
    with pytest.raises(ValueError, match="queries"):
        evaluate(_FixedRetriever(RANKINGS), {}, QRELS)


@pytest.mark.parametrize("ks", [(), (0,), (5, -1)])
def test_evaluate_rejects_invalid_ks(ks):
    with pytest.raises(ValueError, match="ks"):
        evaluate(_FixedRetriever(RANKINGS), QUERIES, QRELS, ks=ks)


def test_evaluate_reports_only_the_selected_metrics():
    result = evaluate(_FixedRetriever(RANKINGS), QUERIES, QRELS, ks=(1, 2), metrics=("ndcg",))

    assert set(result.mean) == {"ndcg@1", "ndcg@2"}
    assert all(set(scores) == {"ndcg@1", "ndcg@2"} for scores in result.per_query.values())


def test_evaluate_raises_for_an_unknown_metric_before_searching():
    retriever = _FixedRetriever(RANKINGS)

    with pytest.raises(ValueError, match="nope"):
        evaluate(retriever, QUERIES, QRELS, metrics=("recall", "nope"))

    assert retriever.search_ks == []


def test_evaluate_rejects_empty_metrics():
    with pytest.raises(ValueError, match="metrics"):
        evaluate(_FixedRetriever(RANKINGS), QUERIES, QRELS, metrics=())


def test_evaluate_uses_a_newly_registered_metric(monkeypatch):
    def top1_is_relevant(ranked_ids, relevant, k):
        return 1.0 if ranked_ids[:1] and ranked_ids[0] in relevant else 0.0

    monkeypatch.setitem(metrics_module.METRIC_REGISTRY, "top1", top1_is_relevant)

    result = evaluate(_FixedRetriever(RANKINGS), QUERIES, QRELS, ks=(3,), metrics=("top1",))

    assert result.per_query == {"q1": {"top1@3": 0.0}, "q2": {"top1@3": 1.0}}
