"""Exact vector store: a plain numpy matrix and a brute-force inner product.

Checks every stored vector, so the result is always the true top-k. It is the reference that
approximate stores are measured against, and it is fast enough for SciFact (about 5k vectors).
"""
from typing import List, Optional

import numpy as np

from ..utils import top_k_indices
from .base import SearchResult, validate_build_input, validate_search_input
from .registry import register_vector_store


@register_vector_store("exact")
class ExactVectorStore:
    name = "exact"

    def __init__(self) -> None:
        self._vectors: Optional[np.ndarray] = None  # (n, dim), float32
        self._ids: List[str] = []

    def build(self, ids: List[str], vectors: np.ndarray) -> None:
        """Replace the content. See VectorStore.build."""
        array = validate_build_input(ids, vectors)
        self._vectors = array
        self._ids = list(ids)

    def search(self, vector: np.ndarray, k: int) -> SearchResult:
        """Exact top-k by inner product, best first; equal scores keep insertion order.

        Raises:
            RuntimeError: if build() hasn't been called.
            ValueError: if k <= 0 or the vector has the wrong shape.
        """
        if self._vectors is None:
            raise RuntimeError("build() must be called before search()")
        query = validate_search_input(vector, k, self._vectors.shape[1])

        scores = self._vectors @ query
        return [(self._ids[i], float(scores[i])) for i in top_k_indices(scores, k)]

    def __len__(self) -> int:
        return len(self._ids)
