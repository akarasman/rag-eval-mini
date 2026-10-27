"""Fusion functions: ways to merge the rankings of several retrievers into one.

A fusion function has the signature

    fusion(rankings: List[Ranking], k: int, **params) -> Ranking

`rankings` holds one ranking per retriever (each best first), `k` is how many results to return,
and `params` are the strategy's own named settings (e.g. rrf_k). It must

- return at most k unique documents, best first, with deterministic tie-breaking;
- validate its arguments and params even when the rankings are empty. HybridRetriever relies on this
  to reject bad settings when it is built (it calls the function once on empty rankings).

To add a strategy: write such a function and decorate it with @register_fusion("name").
"""
from typing import Callable, Dict, List, Optional

import numpy as np

from ..utils import top_k_indices
from .base import Ranking

FusionFunction = Callable[..., Ranking]  # (rankings, k, **params) -> Ranking; params differ per strategy

FUSION_REGISTRY: Dict[str, FusionFunction] = {}


def register_fusion(name: str) -> Callable[[FusionFunction], FusionFunction]:
    """Decorator that registers a fusion function under `name`, returned unchanged.

    Raises:
        ValueError: if the name is already registered.
    """
    def decorator(fusion_fn: FusionFunction) -> FusionFunction:
        if name in FUSION_REGISTRY:
            raise ValueError(f"Fusion {name!r} is already registered")
        FUSION_REGISTRY[name] = fusion_fn
        return fusion_fn
    return decorator


def available_fusions() -> List[str]:
    return sorted(FUSION_REGISTRY)


def get_fusion(name: str) -> FusionFunction:
    """Look up a registered fusion function by name.

    Raises:
        ValueError: if the name is not registered (the message lists the available ones).
    """
    if name not in FUSION_REGISTRY:
        raise ValueError(f"Unknown fusion {name!r}; available: {available_fusions()}")
    return FUSION_REGISTRY[name]


@register_fusion("weighted")
def weighted_score_fusion(rankings: List[Ranking], k: int = 10, weights: Optional[List[float]] = None) -> Ranking:
    """Merge ranked lists by a weighted sum of min-max normalized scores.

    Each ranking's scores are rescaled to [0, 1] (its best document 1, its worst 0), which puts BM25
    scores and cosine similarities on the same scale. A document gets weight * normalized score from
    every ranking it appears in; a document missing from a ranking gets 0 from it, the same as that
    ranking's worst document. Unlike RRF this uses how far apart the scores are, not only the order.
    Higher scores must mean better. Within one ranking, only the first occurrence of a document counts.

    Args:
        rankings: one ranking per retriever, each best first, with comparable-within-a-ranking scores.
        k: how many fused results to return.
        weights: one non-negative weight per ranking, in the same order; they are rescaled to sum to 1.
            None means equal weights.

    Returns:
        At most k (doc_id, fused score) pairs, best first, with scores in [0, 1]. Documents with equal
        scores keep the order in which they first appeared. A ranking whose scores are all equal
        (e.g. a single document) carries no ordering information, so every document in it gets 1.0.

    Raises:
        ValueError: if rankings is empty, k <= 0, or weights has the wrong length, a negative or
            non-finite entry, or sums to zero. Checked even when the rankings themselves are empty.
    """
    if not rankings:
        raise ValueError("at least one ranking is required")
    if k <= 0:
        raise ValueError("k must be positive")
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError(f"got {len(weights)} weights for {len(rankings)} rankings")
    if any(not np.isfinite(w) or w < 0 for w in weights):
        raise ValueError("weights must be finite and non-negative")
    if sum(weights) <= 0:
        raise ValueError("weights must not all be zero")
    shares = [w / sum(weights) for w in weights]

    fused: Dict[str, float] = {}  # insertion order = order of first appearance
    for ranking, share in zip(rankings, shares):
        unique: Dict[str, float] = {}
        for doc_id, score in ranking:
            unique.setdefault(doc_id, score)
        if not unique:
            continue
        low, high = min(unique.values()), max(unique.values())
        for doc_id, score in unique.items():
            normalized = (score - low) / (high - low) if high > low else 1.0
            fused[doc_id] = fused.get(doc_id, 0.0) + share * normalized

    doc_ids = list(fused)
    scores = np.array([fused[doc_id] for doc_id in doc_ids])
    return [(doc_ids[i], float(scores[i])) for i in top_k_indices(scores, k)] if doc_ids else []


@register_fusion("rrf")
def reciprocal_rank_fusion(rankings: List[Ranking], k: int = 10, rrf_k: int = 60) -> Ranking:
    """Merge ranked lists: a document gets 1 / (rrf_k + rank) points from each list it appears in.

    A document ranked well by several retrievers beats one ranked first by only one. Only positions
    are used, never scores, so rankings on different score scales can be mixed. Within one ranking,
    only the first occurrence of a document counts.

    Args:
        rankings: one ranking per retriever, each best first.
        k: how many fused results to return.
        rrf_k: damping constant. Larger values flatten the difference between ranks; 60 is the
            value from the original RRF paper.

    Returns:
        At most k (doc_id, RRF score) pairs, best first. Documents with equal scores keep the order
        in which they first appeared (earlier ranking first, then better rank).

    Raises:
        ValueError: if rankings is empty, k <= 0 or rrf_k <= 0.
    """
    if not rankings:
        raise ValueError("at least one ranking is required")
    if k <= 0:
        raise ValueError("k must be positive")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")

    points: Dict[str, float] = {}  # insertion order = order of first appearance
    for ranking in rankings:
        seen = set()
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            points[doc_id] = points.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    doc_ids = list(points)
    scores = np.array([points[doc_id] for doc_id in doc_ids])
    return [(doc_ids[i], float(scores[i])) for i in top_k_indices(scores, k)] if doc_ids else []
