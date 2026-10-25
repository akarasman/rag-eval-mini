"""Retrieval metrics.

Pure functions that score ONE ranked list against ONE query's relevance labels.
No dependency on retrievers or datasets, so they can be tested on hand-made
examples. Averaging over queries happens in evaluate.py.
"""


import heapq
from math import log2
from typing import Dict, List, Optional


def _require_relevant(relevant: Dict[str, int]) -> None:
    """Metrics are undefined for a query with no relevant docs; callers must skip such queries."""
    if not any(grade > 0 for grade in relevant.values()):
        raise ValueError("query has no relevant docs (no grade > 0); skip it before scoring")


def recall_at_k(ranked_ids: List[str], relevant: Dict[str, int], k: int) -> float:
    """Fraction of relevant docs found in the top-k.
    
    Args:
        ranked_ids: list of doc ids in ranked order
        relevant: dict of doc ids to relevance grades
        k: number of top documents to consider
    
    Returns:
        Recall at k (float between 0 and 1)

    Raises:
        ValueError: if k <= 0, or if the query has no relevant docs.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    _require_relevant(relevant)

    sum_relevant = sum(1 for rv in relevant.values() if rv > 0)
    retrieved_relevant = sum(1 for rid in ranked_ids[:k] if rid in relevant and relevant[rid] > 0)

    return retrieved_relevant / sum_relevant

def reciprocal_rank(ranked_ids: List[str], relevant: Dict[str, int], k: Optional[int] = None) -> float:
    """1 / rank of the first relevant doc (0 if none).
    
    Args:
        ranked_ids: list of doc ids in ranked order
        relevant: dict of doc ids to relevance grades
        k: number of top documents to consider

    Returns:
        Reciprocal rank (float between 0 and 1)

    Raises:
        ValueError: if k is given and <= 0, or if the query has no relevant docs.
    """

    if k is not None and k <= 0:
        raise ValueError("k must be positive")
    _require_relevant(relevant)

    for i, doc_id in enumerate(ranked_ids, start=1):
        if k is not None and i > k:
            break
        if doc_id in relevant and relevant[doc_id] > 0:
            return 1.0 / i
    
    return 0.0


def ndcg_at_k(ranked_ids: List[str], relevant: Dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain at k.
    
    Args:
        ranked_ids: list of doc ids in ranked order
        relevant: dict of doc ids to relevance grades
        k: number of top documents to consider

    Returns:
        Normalised discounted cumulative gain at k (float between 0 and 1)

    Raises:
        ValueError: if k <= 0, or if the query has no relevant docs.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    _require_relevant(relevant)

    dcgk = sum((2**relevant[doc_id] - 1) / log2(i+2) for i, doc_id in enumerate(ranked_ids[:k]) if doc_id in relevant)
    ideal_ranking = heapq.nlargest(k, relevant.values())
    idcgk = sum((2**rel - 1) / log2(i+2) for i, rel in enumerate(ideal_ranking))

    return dcgk / idcgk
