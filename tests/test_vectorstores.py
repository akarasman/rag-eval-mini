"""Tests for vector stores.

Contract tests run against every store listed in STORE_FACTORIES (adding a backend there runs them all);
behaviour specific to the exact store (determinism, true top-k) is tested separately.
"""
import numpy as np
import pytest

from rag_eval.vectorstores.exact import ExactVectorStore

STORE_FACTORIES = {"exact": ExactVectorStore}


@pytest.fixture(params=sorted(STORE_FACTORIES))
def make_store(request):
    """A factory for an empty store of the backend under test."""
    return STORE_FACTORIES[request.param]


IDS = ["a", "b", "c", "d"]
VECTORS = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.6, 0.8],
    [-1.0, 0.0],
])


def _built(make_store):
    store = make_store()
    store.build(IDS, VECTORS)
    return store


# --- contract: every store must pass these ----------------------------------

def test_search_returns_the_nearest_vectors_best_first_with_inner_product_scores(make_store):
    result = _built(make_store).search(np.array([1.0, 0.0]), k=3)

    assert [vector_id for vector_id, _ in result] == ["a", "c", "b"]
    assert [score for _, score in result] == pytest.approx([1.0, 0.6, 0.0])


def test_search_trims_to_k_and_returns_everything_when_k_exceeds_the_size(make_store):
    store = _built(make_store)

    assert len(store.search(np.array([1.0, 0.0]), k=2)) == 2
    assert len(store.search(np.array([1.0, 0.0]), k=100)) == len(IDS)


def test_len_counts_the_stored_vectors(make_store):
    store = make_store()
    assert len(store) == 0

    store.build(IDS, VECTORS)

    assert len(store) == 4


def test_build_replaces_the_previous_content(make_store):
    store = _built(make_store)

    store.build(["x"], np.array([[0.0, 1.0]]))

    assert [vector_id for vector_id, _ in store.search(np.array([0.0, 1.0]), k=5)] == ["x"]
    assert len(store) == 1


def test_build_keeps_its_own_copy_of_the_vectors(make_store):
    vectors = VECTORS.copy()
    store = make_store()
    store.build(IDS, vectors)

    vectors[:] = 0.0  # the caller reuses its array

    assert store.search(np.array([1.0, 0.0]), k=1)[0][0] == "a"


def test_search_before_build_raises(make_store):
    with pytest.raises(RuntimeError, match="build"):
        make_store().search(np.array([1.0, 0.0]), k=1)


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(make_store, k):
    with pytest.raises(ValueError, match="k"):
        _built(make_store).search(np.array([1.0, 0.0]), k=k)


@pytest.mark.parametrize("query", [np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.0]]), np.array(1.0)])
def test_search_rejects_a_query_of_the_wrong_shape(make_store, query):
    with pytest.raises(ValueError, match="shape"):
        _built(make_store).search(query, k=1)


@pytest.mark.parametrize("ids, vectors, message", [
    (["a", "b"], np.zeros((3, 2)), "ids"),  # length mismatch
    (["a", "a"], np.zeros((2, 2)), "unique"),
    ([], np.zeros((0, 2)), "non-empty"),
    (["a"], np.zeros(2), "2-D"),
])
def test_build_rejects_bad_input_and_keeps_the_previous_content(make_store, ids, vectors, message):
    store = _built(make_store)

    with pytest.raises(ValueError, match=message):
        store.build(ids, vectors)

    assert len(store) == len(IDS)
    assert store.search(np.array([1.0, 0.0]), k=1)[0][0] == "a"


# --- exact store only -------------------------------------------------------

def test_exact_store_keeps_insertion_order_for_equal_scores():
    store = ExactVectorStore()
    store.build(["z", "a", "m"], np.array([[1.0, 0.0]] * 3))

    assert [vector_id for vector_id, _ in store.search(np.array([1.0, 0.0]), k=3)] == ["z", "a", "m"]


def test_exact_store_finds_the_true_top_k_on_random_data():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(500, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[17]
    store = ExactVectorStore()
    store.build([str(i) for i in range(500)], vectors)

    result = store.search(query, k=10)

    expected = np.argsort(-(vectors @ query), kind="stable")[:10]
    assert [vector_id for vector_id, _ in result] == [str(i) for i in expected]
    assert result[0][0] == "17"  # a vector is its own nearest neighbour
    assert result[0][1] == pytest.approx(1.0, abs=1e-5)


def test_exact_store_breaks_ties_at_the_cutoff_by_insertion_order():
    store = ExactVectorStore()
    store.build(list("abcdef"), np.array([[1.0, 0.0]] * 6))  # six equal scores, only three fit

    assert [vector_id for vector_id, _ in store.search(np.array([1.0, 0.0]), k=3)] == ["a", "b", "c"]


# --- registry ---------------------------------------------------------------

def test_exact_store_is_registered_and_built_by_name():
    from rag_eval.vectorstores.registry import build_vector_store

    assert isinstance(build_vector_store("exact", {}), ExactVectorStore)


def test_build_vector_store_rejects_an_unknown_name_and_lists_the_available_ones():
    from rag_eval.vectorstores.registry import build_vector_store

    with pytest.raises(ValueError, match="exact"):
        build_vector_store("nope", {})


def test_build_vector_store_rejects_parameters_the_store_does_not_take():
    from rag_eval.vectorstores.registry import build_vector_store

    with pytest.raises(ValueError, match="exact.*typo"):
        build_vector_store("exact", {"typo": 1})


def test_register_vector_store_rejects_a_duplicate_name():
    from rag_eval.vectorstores.registry import register_vector_store

    with pytest.raises(ValueError, match="already registered"):
        register_vector_store("exact")(ExactVectorStore)
