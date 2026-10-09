"""Hybrid retriever.

Combines the rankings of two (or more) retrievers into one. The central
question is how to fuse results whose scores live on different scales.
"""
from .base import Ranking, Retriever


def fuse(rankings: list[Ranking], k: int = 10) -> Ranking:
    """Merge several ranked lists into one."""
    # TODO: pick a fusion method (rank-based vs. score-based) and implement it
    raise NotImplementedError


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], candidates_per_retriever: int = 100):
        # TODO: store sub-retrievers and fusion parameters
        raise NotImplementedError

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        # TODO: index sub-retrievers (or accept already-indexed ones; your call)
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> Ranking:
        # TODO: query each retriever, fuse, return top-k
        raise NotImplementedError
