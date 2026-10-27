"""FAISS vector store: exact ("flat") or approximate ("hnsw") inner-product search.

"flat" checks every stored vector, like the exact store, but with FAISS's optimized kernels. "hnsw"
searches a graph of neighbours instead, which is much faster on large collections but may miss some
true neighbours; ef_search trades speed for accuracy at query time.

FAISS only knows row positions (0..n-1), so the store keeps the list of IDs and maps positions back.
"""
from typing import List, Optional

import faiss
import numpy as np

from .base import SearchResult, validate_build_input, validate_search_input
from .registry import register_vector_store

_INDEX_TYPES = ("flat", "hnsw")


@register_vector_store("faiss")
class FaissVectorStore:
    """Args:
        index_type: "flat" (exact) or "hnsw" (approximate).
        hnsw_m: links per node in the HNSW graph; more is more accurate, bigger and slower to build.
            Ignored for "flat".
        ef_search: candidates HNSW examines per query; more is more accurate and slower. Ignored for "flat".

    Raises:
        ValueError: if index_type is unknown, or hnsw_m or ef_search is not positive.
    """
    name = "faiss"

    def __init__(self, index_type: str = "flat", hnsw_m: int = 32, ef_search: int = 64):
        if index_type not in _INDEX_TYPES:
            raise ValueError(f"index_type must be one of {_INDEX_TYPES}, got {index_type!r}")
        if hnsw_m <= 0 or ef_search <= 0:
            raise ValueError("hnsw_m and ef_search must be positive")
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.ef_search = ef_search
        self._index: Optional[faiss.Index] = None
        self._ids: List[str] = []

    def build(self, ids: List[str], vectors: np.ndarray) -> None:
        """Replace the content with a freshly built index. See VectorStore.build."""
        array = validate_build_input(ids, vectors)  # checked before anything is replaced
        dim = array.shape[1]
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(dim)
        else:
            index = faiss.IndexHNSWFlat(dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = self.ef_search
        index.add(array)

        self._index = index
        self._ids = list(ids)

    def search(self, vector: np.ndarray, k: int) -> SearchResult:
        """Top-k by inner product, best first. Exact for "flat", approximate for "hnsw".

        Raises:
            RuntimeError: if build() hasn't been called.
            ValueError: if k <= 0 or the vector has the wrong shape.
        """
        if self._index is None:
            raise RuntimeError("build() must be called before search()")
        query = validate_search_input(vector, k, self._index.d)

        scores, positions = self._index.search(query.reshape(1, -1), k)
        
        # When k exceeds the stored count, FAISS pads with position -1; those are not results.
        return [(self._ids[pos], float(score)) for score, pos in zip(scores[0], positions[0]) if pos != -1]

    def __len__(self) -> int:
        return len(self._ids)
