"""Retrieval metrics.

Pure functions that score ONE ranked list against ONE query's relevance labels.
No dependency on retrievers or datasets, so they can be tested on hand-made
examples. Averaging over queries happens in evaluate.py.
"""


def recall_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """Fraction of relevant docs found in the top-k."""
    # TODO: define "relevant" (grade > 0?) and the empty-relevant-set case
    raise NotImplementedError


def reciprocal_rank(ranked_ids: list[str], relevant: dict[str, int], k: int | None = None) -> float:
    """1 / rank of the first relevant doc (0 if none)."""
    # TODO: decide whether to cut off at k
    raise NotImplementedError


def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """Normalised discounted cumulative gain at k."""
    # TODO: choose gain (rel vs 2^rel - 1) and discount; compute the ideal DCG from the qrels
    # TODO: guard against a zero ideal DCG
    raise NotImplementedError
