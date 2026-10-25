"""Lexical retriever (BM25).

Scores documents by weighted term overlap with the query. No embeddings,
no model: a strong baseline that the dense retriever has to beat.

Uses the `bm25s` library (Lucene BM25 variant). Tokenization is ours (text.tokenize),
applied identically to documents and queries.
"""
from typing import Dict, List, Optional

import bm25s
import numpy as np

from ..text import doc_to_text, tokenize
from .base import Ranking


class BM25Retriever:
    """Args:
        k1: term-frequency saturation.
        b: document-length normalization.
        stem: stem tokens. Held here so documents and queries can never be tokenized differently.
    """
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75, stem: bool = False):
        self.k1 = k1
        self.b = b
        self.stem = stem
        self._bm25: Optional[bm25s.BM25] = None
        self._doc_ids: List[str] = []

    def index(self, corpus: Dict[str, Dict[str, str]]) -> None:
        """Tokenize every document and build the BM25 statistics."""
        doc_ids = list(corpus)
        tokenized = [tokenize(doc_to_text(corpus[doc_id]), stem=self.stem) for doc_id in doc_ids]

        bm25 = bm25s.BM25(method="lucene", k1=self.k1, b=self.b)
        bm25.index(tokenized, show_progress=False)

        self._bm25 = bm25 
        self._doc_ids = doc_ids

    def search(self, query: str, k: int = 10) -> Ranking:
        """Top-k documents, best first. Documents sharing no term with the query score 0.0
        and pad the ranking in corpus order, so the result has min(k, corpus size) entries.

        Raises:
            RuntimeError: if index() hasn't been called.
            ValueError: if k <= 0.
        """
        if self._bm25 is None:
            raise RuntimeError("index() must be called before search()")
        if k <= 0:
            raise ValueError("k must be positive")

        query_tokens = tokenize(query, stem=self.stem)
        if query_tokens:
            scores = self._bm25.get_scores(query_tokens)
        else:  # bm25s raises on an empty token list
            scores = np.zeros(len(self._doc_ids))

        # Stable sort: documents with equal scores (e.g. all the zeros) keep corpus order.
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self._doc_ids[i], float(scores[i])) for i in top]
