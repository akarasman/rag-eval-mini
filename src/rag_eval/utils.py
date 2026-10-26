"""Small helpers shared by several modules."""
import numpy as np


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first; equal scores keep index order.

    Gives the same result as np.argsort(-scores, kind="stable")[:k], but only sorts the k winners
    instead of all n scores: O(n + k log k) instead of O(n log n). Equal scores are ordered by index
    (lowest first), including at the cutoff, so the result is fully deterministic.

    Args:
        scores: 1-D array of scores, higher is better. Must not contain NaN.
        k: how many indices to return. If k >= len(scores), all indices are returned, sorted.

    Returns:
        Integer array of at most k indices into `scores`. The input is not modified.

    Raises:
        ValueError: if k <= 0 or scores is not 1-D.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if scores.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {scores.shape}")

    n = len(scores)
    if k >= n:
        return np.argsort(-scores, kind="stable")

    cutoff = np.partition(scores, n - k)[n - k]  # the k-th largest score
    above = np.flatnonzero(scores > cutoff)  # certainly in the top k, in index order
    ties = np.flatnonzero(scores == cutoff)[: k - len(above)]  # fill the rest from the cutoff ties, lowest index first
    winners = np.concatenate([above, ties])
    return winners[np.argsort(-scores[winners], kind="stable")]
