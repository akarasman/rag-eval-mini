"""Tests for the shared helpers."""
import numpy as np
import pytest

from rag_eval.utils import top_k_indices


def _reference(scores: np.ndarray, k: int) -> list:
    """What top_k_indices must equal: a full stable sort, best first, ties by index."""
    return list(np.argsort(-scores, kind="stable")[:k])


# --- same result as argsort -------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 10, 99, 100, 101, 1000])
@pytest.mark.parametrize("seed", range(5))
def test_matches_a_full_stable_sort_on_random_scores(seed, k):
    scores = np.random.default_rng(seed).normal(size=100)

    assert list(top_k_indices(scores, k)) == _reference(scores, k)


@pytest.mark.parametrize("k", [1, 3, 7, 50, 199, 200, 500])
@pytest.mark.parametrize("distinct_values", [1, 2, 6])
def test_matches_a_full_stable_sort_when_many_scores_are_tied(distinct_values, k):
    scores = np.random.default_rng(1).integers(0, distinct_values, size=200).astype(np.float32)

    assert list(top_k_indices(scores, k)) == _reference(scores, k)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_matches_a_full_stable_sort_for_any_numeric_dtype(dtype):
    scores = np.random.default_rng(2).integers(-20, 20, size=60).astype(dtype)

    assert list(top_k_indices(scores, 7)) == _reference(scores, 7)


# --- behaviour --------------------------------------------------------------

def test_returns_the_best_scores_first():
    scores = np.array([0.1, 0.9, 0.5, 0.7])

    assert list(top_k_indices(scores, 2)) == [1, 3]


def test_ties_are_ordered_by_index_including_at_the_cutoff():
    scores = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    assert list(top_k_indices(scores, 3)) == [0, 1, 2]


def test_ties_at_the_cutoff_prefer_the_lower_index_over_a_higher_one():
    scores = np.array([5.0, 3.0, 9.0, 3.0, 3.0])  # one 9, one 5, then three tied 3s competing for one place

    assert list(top_k_indices(scores, 3)) == [2, 0, 1]


def test_negative_and_infinite_scores_are_handled():
    scores = np.array([-1.0, -np.inf, 0.0, -0.5, np.inf])

    assert list(top_k_indices(scores, 3)) == [4, 2, 3]


def test_returns_every_index_when_k_is_at_least_the_size():
    scores = np.array([0.2, 0.8, 0.5])

    assert list(top_k_indices(scores, 3)) == [1, 2, 0]
    assert list(top_k_indices(scores, 10)) == [1, 2, 0]


def test_a_single_score():
    assert list(top_k_indices(np.array([0.3]), 5)) == [0]


def test_an_empty_array_gives_no_indices():
    assert len(top_k_indices(np.array([]), 3)) == 0


def test_returns_an_integer_index_array_usable_for_indexing():
    scores = np.array([0.1, 0.9, 0.5])

    top = top_k_indices(scores, 2)

    assert isinstance(top, np.ndarray) and np.issubdtype(top.dtype, np.integer)
    assert list(scores[top]) == [0.9, 0.5]


def test_does_not_modify_its_input():
    scores = np.random.default_rng(3).normal(size=50)
    snapshot = scores.copy()

    top_k_indices(scores, 5)

    assert np.array_equal(scores, snapshot)


def test_is_deterministic():
    scores = np.random.default_rng(4).integers(0, 3, size=80).astype(np.float32)

    assert list(top_k_indices(scores, 10)) == list(top_k_indices(scores, 10))


# --- errors -----------------------------------------------------------------

@pytest.mark.parametrize("k", [0, -1])
def test_rejects_non_positive_k(k):
    with pytest.raises(ValueError, match="k"):
        top_k_indices(np.array([1.0, 2.0]), k)


@pytest.mark.parametrize("scores", [np.zeros((2, 2)), np.array(1.0)])
def test_rejects_scores_that_are_not_one_dimensional(scores):
    with pytest.raises(ValueError, match="1-D"):
        top_k_indices(scores, 1)
