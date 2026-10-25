"""Evaluation loop.

Runs a retriever over every query, scores each ranking with the metric
functions, and averages. Returns per-query scores and their means; the means are
a flat dict of metric name -> value that the experiment script can log.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .metrics import MetricFunction, get_metric
from .retrievers.base import Retriever


@dataclass(frozen=True)
class EvaluationResult:
    """Metric values per query and their macro average.

    Attributes:
        mean: metric name -> mean over queries, e.g. {"recall@10": ..., "mrr@10": ..., "ndcg@10": ...}.
        per_query: query_id -> {metric name -> value}, for error analysis and paired comparisons.
    """
    mean: Dict[str, float]
    per_query: Dict[str, Dict[str, float]]


def _get_per_query_scores(
    ranked_ids: List[str],
    relevant: Dict[str, int],
    ks: Tuple[int, ...],
    metric_fns: Dict[str, MetricFunction],
) -> Dict[str, float]:
    """Score one ranking with every selected metric at every k. Keys look like "recall@10"."""
    return {
        f"{name}@{k}": metric_fn(ranked_ids, relevant, k)
        for name, metric_fn in metric_fns.items()
        for k in ks
    }


def _validate_queries(queries: Dict[str, str], qrels: Dict[str, Dict[str, int]]) -> None:
    for query_id in queries:
        if not any(grade > 0 for grade in qrels.get(query_id, {}).values()):
            raise ValueError(f"query {query_id!r} has no relevant docs in qrels")


def _mean_scores(per_query_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Macro average: every query counts equally."""
    num_queries = len(per_query_scores)
    metric_names = next(iter(per_query_scores.values())).keys()
    return {
        name: sum(scores[name] for scores in per_query_scores.values()) / num_queries
        for name in metric_names
    }


def evaluate(
    retriever: Retriever,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    ks: Tuple[int, ...] = (1, 5, 10, 100),
    metrics: Tuple[str, ...] = ("recall", "mrr", "ndcg"),
) -> EvaluationResult:
    """Score a retriever on every query with the selected metrics at each k in ks.

    Metrics are looked up by name in metrics.METRIC_REGISTRY and reported as "<name>@<k>".
    Each query is searched once with k=max(ks) and the same ranking is scored at every k.
    A ranking shorter than k (or empty) is scored as is, so missing documents count as misses.
    Every query counts equally in the means.

    Raises:
        ValueError: if ks or metrics is empty, if ks has a non-positive value, if a metric name is
            not registered, if there are no queries, or if a query has no qrels or no relevant doc
            (grade > 0). These mean a problem upstream, so it fails loudly instead of silently
            shrinking the query set. Everything is checked before the first search.
    """
    if not ks or any(k <= 0 for k in ks):
        raise ValueError(f"ks must be a non-empty tuple of positive integers, got {ks!r}")
    if not metrics:
        raise ValueError("metrics must not be empty")
    metric_fns = {name: get_metric(name) for name in metrics}  # raises for unknown names
    if not queries:
        raise ValueError("no queries to evaluate")
    _validate_queries(queries, qrels)

    max_k = max(ks)
    per_query_scores: Dict[str, Dict[str, float]] = {}
    for query_id, query_text in queries.items():
        ranked_ids = [doc_id for doc_id, _ in retriever.search(query_text, k=max_k)]
        per_query_scores[query_id] = _get_per_query_scores(ranked_ids, qrels[query_id], ks, metric_fns)

    mean = _mean_scores(per_query_scores)

    logging.info(f"Evaluated {retriever.name} on {len(per_query_scores)} queries: " +
                 ", ".join(f"{name}={value:.4f}" for name, value in mean.items()))
    return EvaluationResult(mean=mean, per_query=per_query_scores)
