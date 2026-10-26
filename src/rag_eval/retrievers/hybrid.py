"""Hybrid retriever.

Asks several retrievers (e.g. BM25 and a dense model) and merges their rankings with a fusion
strategy chosen by name (see fusion.py; "rrf", Reciprocal Rank Fusion, by default). RRF uses only the
POSITIONS in each ranking, never the scores, so it does not matter that BM25 scores and cosine
similarities live on different scales.
"""
import inspect
from typing import Any, Dict, List, Optional

from .base import Ranking, Retriever
from .fusion import get_fusion
from .registry import build_retriever, register_retriever


@register_retriever("hybrid")
class HybridRetriever:
    """Combines two or more retrievers with a fusion strategy.

    Give the sub-retrievers either by name (`components`, which is what a YAML config uses) or as
    ready-made objects (`retrievers`, handy in tests). Exactly one of the two is required.

    Args:
        components: [{"name": "bm25", "params": {...}}, ...], built through the retriever registry.
        retrievers: already constructed retrievers.
        fusion: name of a registered fusion strategy ("rrf", ...).
        fusion_params: that strategy's named settings, e.g. {"rrf_k": 60}.
        candidates_per_retriever: how many results to take from each sub-retriever before fusing;
            at least k are always taken.

    index() indexes every sub-retriever. Sub-retrievers given by name are built immediately (a dense
    one loads its model only on index()), and the fusion name and settings are checked immediately,
    so a mistake fails before any experiment run starts. The scores of the returned ranking are the
    fusion's scores (for RRF small numbers), not BM25 or cosine scores.

    Raises:
        ValueError: if not exactly one of components / retrievers is given, there are fewer than
            two retrievers, candidates_per_retriever is not positive, or the fusion is unknown or
            its settings are invalid.
    """
    name = "hybrid"

    def __init__(
        self,
        components: Optional[List[Dict[str, Any]]] = None,
        retrievers: Optional[List[Retriever]] = None,
        fusion: str = "rrf",
        fusion_params: Optional[Dict[str, Any]] = None,
        candidates_per_retriever: int = 100,
    ):
        if (components is None) == (retrievers is None):
            raise ValueError("give exactly one of components or retrievers")
        if candidates_per_retriever <= 0:
            raise ValueError("candidates_per_retriever must be positive")

        if retrievers is None:
            retrievers = [build_retriever(c["name"], c.get("params", {})) for c in components]
        if len(retrievers) < 2:
            raise ValueError("a hybrid retriever needs at least two retrievers")

        self._retrievers: List[Retriever] = list(retrievers)
        self.fusion_name = fusion
        self.fusion_params: Dict[str, Any] = dict(fusion_params or {})
        self._fusion = get_fusion(fusion)
        self._validate_fusion_params()
        self.candidates_per_retriever = candidates_per_retriever
        self._indexed = False

    def _validate_fusion_params(self) -> None:
        """Reject unknown or invalid fusion settings now, not at the first search.

        Fusion functions validate their arguments even for empty rankings (see fusion.py), so one
        call on empty rankings, one per retriever, checks the names and the values.
        """
        try:
            inspect.signature(self._fusion).bind_partial(**self.fusion_params)
            self._fusion([[] for _ in self._retrievers], k=1, **self.fusion_params)
        except TypeError as error:  # unknown keyword argument, or one that clashes with rankings / k
            raise ValueError(f"Invalid parameters for fusion {self.fusion_name!r}: {error}") from error

    def index(self, corpus: Dict[str, Dict[str, str]]) -> None:
        """Index every sub-retriever on the corpus."""
        for retriever in self._retrievers:
            retriever.index(corpus)
        self._indexed = True

    def search(self, query: str, k: int = 10) -> Ranking:
        """Search every sub-retriever, fuse their rankings, return the top k.

        Raises:
            RuntimeError: if index() hasn't been called.
            ValueError: if k <= 0.
        """
        if not self._indexed:
            raise RuntimeError("index() must be called before search()")
        if k <= 0:
            raise ValueError("k must be positive")

        depth = max(self.candidates_per_retriever, k)
        rankings = [retriever.search(query, k=depth) for retriever in self._retrievers]
        return self._fusion(rankings, k=k, **self.fusion_params)
