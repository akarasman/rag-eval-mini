"""Tests for the MLflow logging helpers that do not need MLflow itself."""
import csv
import io
from typing import Dict, List

import pytest

from rag_eval.data import Dataset
from rag_eval.evaluate import EvaluationResult
from rag_eval.retrievers.base import Ranking
from rag_eval.tracking import _build_example, _mlflow_metric_name, _per_query_csv, _select_examples


def _per_query(scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {query_id: {"ndcg@10": score} for query_id, score in scores.items()}


# --- metric names -----------------------------------------------------------

@pytest.mark.parametrize("name, expected", [("recall@10", "recall_at_10"), ("ndcg@1", "ndcg_at_1"), ("mrr", "mrr")])
def test_mlflow_metric_name_replaces_the_at_sign(name, expected):
    assert _mlflow_metric_name(name) == expected


# --- _select_examples --------------------------------------------------------

def test_select_examples_returns_the_lowest_and_highest_scoring_queries():
    per_query = _per_query({"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.7, "e": 0.3})

    worst, best = _select_examples(per_query, "ndcg@10", n=2)

    assert worst == ["b", "e"]  # lowest first
    assert best == ["a", "d"]  # highest first


def test_select_examples_breaks_ties_by_query_id_so_it_is_reproducible():
    per_query = _per_query({"q3": 0.0, "q1": 0.0, "q2": 0.0, "q4": 1.0})

    worst, best = _select_examples(per_query, "ndcg@10", n=2)

    assert worst == ["q1", "q2"]
    assert best == ["q4", "q3"]


def test_select_examples_never_lists_a_query_twice():
    worst, best = _select_examples(_per_query({"a": 0.1, "b": 0.5, "c": 0.9}), "ndcg@10", n=2)

    assert worst == ["a", "b"]
    assert best == ["c"]
    assert not set(worst) & set(best)


def test_select_examples_with_zero_returns_nothing():
    assert _select_examples(_per_query({"a": 0.1}), "ndcg@10", n=0) == ([], [])


# --- _build_example ----------------------------------------------------------

class _FixedRetriever:
    name = "fixed"

    def __init__(self, doc_ids: List[str]):
        self.doc_ids = doc_ids
        self.search_ks: List[int] = []

    def index(self, corpus) -> None:
        pass

    def search(self, query: str, k: int = 10) -> Ranking:
        self.search_ks.append(k)
        return [(doc_id, 1.0 / rank) for rank, doc_id in enumerate(self.doc_ids[:k], start=1)]


def _dataset() -> Dataset:
    corpus = {doc_id: {"title": f"title {doc_id}", "text": ""} for doc_id in ["a", "b", "c", "d"]}
    return Dataset(
        name="tiny", split="test", corpus=corpus,
        queries={"q1": "first query"},
        qrels={"q1": {"b": 1, "d": 2}},  # d is more relevant than b
    )


def test_build_example_lists_relevant_docs_with_their_rank_and_the_top_results_with_their_grade():
    retriever = _FixedRetriever(["a", "b", "c", "d"])

    example = _build_example("q1", _dataset(), retriever, {"ndcg@10": 0.4}, search_k=100)

    assert example["query"] == "first query"
    assert example["scores"] == {"ndcg@10": 0.4}
    assert example["relevant"] == [
        {"doc_id": "d", "grade": 2, "title": "title d", "rank": 4},  # most relevant first
        {"doc_id": "b", "grade": 1, "title": "title b", "rank": 2},
    ]
    assert [(doc["rank"], doc["doc_id"], doc["relevance"]) for doc in example["retrieved"]] == [
        (1, "a", 0), (2, "b", 1), (3, "c", 0), (4, "d", 2),
    ]


def test_build_example_marks_a_relevant_doc_that_was_not_retrieved_with_no_rank():
    example = _build_example("q1", _dataset(), _FixedRetriever(["a", "c"]), {}, search_k=100)

    assert [doc["rank"] for doc in example["relevant"]] == [None, None]


def test_build_example_searches_deep_enough_to_find_relevant_docs_but_lists_only_the_top():
    docs = {f"d{i}": {"title": "", "text": ""} for i in range(30)}
    dataset = Dataset(name="t", split="test", corpus=docs, queries={"q": "x"}, qrels={"q": {"d25": 1}})
    retriever = _FixedRetriever(list(docs))

    example = _build_example("q", dataset, retriever, {}, search_k=100)

    assert retriever.search_ks == [100]
    assert example["relevant"][0]["rank"] == 26
    assert len(example["retrieved"]) == 10


# --- _per_query_csv ----------------------------------------------------------

def test_per_query_csv_has_one_row_per_query_with_a_column_per_metric():
    result = EvaluationResult(
        mean={},
        per_query={"q1": {"recall@5": 0.5, "ndcg@5": 1.0}, "q2": {"recall@5": 0.0, "ndcg@5": 0.25}},
    )

    rows = list(csv.reader(io.StringIO(_per_query_csv(result, {"q1": "first, with comma", "q2": "second"}))))

    assert rows[0] == ["query_id", "query", "recall@5", "ndcg@5"]
    assert rows[1] == ["q1", "first, with comma", "0.500000", "1.000000"]  # the comma must not break the row
    assert rows[2] == ["q2", "second", "0.000000", "0.250000"]
