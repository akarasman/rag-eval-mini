"""Lexical retriever (BM25).

Scores documents by weighted term overlap with the query. No embeddings,
no model: a strong baseline that the dense retriever has to beat.
"""
from .base import Ranking


class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        # TODO: store hyperparameters
        raise NotImplementedError

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        # TODO: tokenize docs, build term statistics (or hand them to a library)
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> Ranking:
        # TODO: tokenize query, score all docs, return top-k (doc_id, score)
        raise NotImplementedError
