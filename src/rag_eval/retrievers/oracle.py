"""Oracle retriever (test double).

Cheats by reading the qrels: for each query it returns that query's relevant docs
first (best grade first). Evaluated with the real metrics it should score perfectly,
which makes it a check on evaluate.py and the metrics, and a ceiling for real retrievers.
Not a real retriever: it can only answer queries it was built with.
"""
import logging
from typing import Dict, List, Optional

from .base import Ranking


class OracleRetriever:
    """Args:
        queries: query_id -> query text.
        qrels: query_id -> {doc_id: relevance grade}.
        use_filler_docs: pad rankings with non-relevant docs up to k, like a real retriever
            would. Scores are unaffected either way; this only changes the ranking length.
    """
    name = "oracle"

    def __init__(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], use_filler_docs: bool = True):
        self.queries = queries
        self.qrels = qrels
        self.use_filler_docs = use_filler_docs
        self._index: Optional[Dict[str, List[str]]] = None  # query text -> relevant doc IDs, best grade first
        self._filler_docs: List[str] = []

    def index(self, corpus: Dict[str, Dict[str, str]]) -> None:
        """Build the query text -> relevant doc IDs lookup (docs missing from the corpus are ignored)."""
        index: Dict[str, List[str]] = {}
        for query_id, query_text in self.queries.items():
            if query_text in index:
                raise ValueError(f"Duplicate query text: {query_text}")

            relevant_docs = {
                doc_id: grade for doc_id, grade in self.qrels.get(query_id, {}).items()
                if doc_id in corpus
            }
            index[query_text] = sorted(relevant_docs, key=relevant_docs.get, reverse=True)

        self._index = index  # assigned last, so a failed index() doesn't leave half-built state
        self._filler_docs = list(corpus) if self.use_filler_docs else []

    def _add_filler_docs(self, doc_ids: List[str], k: int) -> List[str]:
        """Pad doc_ids with non-relevant docs up to k. Returns a new list."""
        if len(doc_ids) >= k:
            return doc_ids

        padded = list(doc_ids)
        taken = set(doc_ids)  # filler must not repeat a relevant doc, or recall could exceed 1
        for doc_id in self._filler_docs:
            if len(padded) == k:
                break
            if doc_id not in taken:
                padded.append(doc_id)

        if len(padded) < k:
            logging.warning(f"Not enough filler docs: ranking has {len(padded)} < k={k} docs.")
        return padded

    def search(self, query: str, k: int = 10) -> Ranking:
        """Relevant docs first (best grade first), then optional filler, trimmed to k.

        Raises:
            RuntimeError: if index() hasn't been called.
            KeyError: if the query text is not one the oracle was built with.
            ValueError: if k <= 0.
        """
        if self._index is None:
            raise RuntimeError("index() must be called before search()")
        if k <= 0:
            raise ValueError("k must be positive")
        if query not in self._index:
            raise KeyError(f"Oracle has never seen this query: {query!r}")

        doc_ids = self._index[query][:k]  # slicing copies, so the stored list is never mutated
        if self.use_filler_docs:
            doc_ids = self._add_filler_docs(doc_ids, k)

        # Strictly decreasing scores, so the order is unambiguous even for equal grades.
        return [(doc_id, 1.0 / rank) for rank, doc_id in enumerate(doc_ids, start=1)]
