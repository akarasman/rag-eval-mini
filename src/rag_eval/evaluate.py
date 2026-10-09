"""Evaluation loop.

Runs a retriever over every query, scores each ranking with the metric
functions, and averages. Returns a flat dict of metric name -> value that the
experiment script can log.
"""
from .retrievers.base import Retriever


def evaluate(
    retriever: Retriever,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    ks: tuple[int, ...] = (1, 5, 10, 100),
) -> dict[str, float]:
    """Return e.g. {"recall@10": ..., "mrr@10": ..., "ndcg@10": ...}."""
    # TODO: iterate over queries that have qrels (decide what to do with the rest)
    # TODO: retrieve once with max(ks), compute each metric per query, average
    # TODO: decide how to treat queries with no relevant docs
    raise NotImplementedError
