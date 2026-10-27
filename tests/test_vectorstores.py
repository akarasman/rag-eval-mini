"""Tests for vector stores.

Contract tests run against every store listed in STORE_FACTORIES (adding a backend there runs them all);
behaviour specific to one store (exact: determinism, true top-k; faiss: agreement with exact) is tested
separately.
"""
from functools import partial

import numpy as np
import pytest

from rag_eval.vectorstores.exact import ExactVectorStore
from rag_eval.vectorstores.faiss_store import FaissVectorStore

STORE_FACTORIES = {
    "exact": ExactVectorStore,
    "faiss_flat": partial(FaissVectorStore, index_type="flat"),
    "faiss_hnsw": partial(FaissVectorStore, index_type="hnsw"),
}


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


# --- faiss store only -------------------------------------------------------
# Tie order is not tested here: FAISS does not guarantee one, so a test of it would pin an accident.

def _random_unit_vectors(n: int, dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [str(i) for i in range(n)], vectors


def _ids_of(result):
    return [vector_id for vector_id, _ in result]


@pytest.mark.parametrize("params", [
    {"index_type": "ivf"},
    {"hnsw_m": 0},
    {"ef_search": -1},
])
def test_faiss_store_rejects_bad_parameters(params):
    with pytest.raises(ValueError, match=next(iter(params))):  # the message names the bad parameter
        FaissVectorStore(**params)


def test_faiss_flat_returns_the_same_top_k_as_the_exact_store():
    ids, vectors = _random_unit_vectors(500, 32)
    exact, flat = ExactVectorStore(), FaissVectorStore(index_type="flat")
    exact.build(ids, vectors)
    flat.build(ids, vectors)

    for query in vectors[:20]:
        expected, result = exact.search(query, k=10), flat.search(query, k=10)

        assert _ids_of(result) == _ids_of(expected)
        assert [score for _, score in result] == pytest.approx([score for _, score in expected], abs=1e-5)


def test_faiss_hnsw_finds_most_of_the_true_neighbours():
    # Approximate, so measure recall against the exact store rather than demanding equality.
    ids, vectors = _random_unit_vectors(500, 32)
    exact, hnsw = ExactVectorStore(), FaissVectorStore(index_type="hnsw")
    exact.build(ids, vectors)
    hnsw.build(ids, vectors)

    queries = _random_unit_vectors(20, 32, seed=1)[1]  # not stored vectors, so the task is not trivial
    recalls = [
        len(set(_ids_of(hnsw.search(q, k=10))) & set(_ids_of(exact.search(q, k=10)))) / 10
        for q in queries
    ]

    assert np.mean(recalls) >= 0.9


def test_faiss_hnsw_builds_an_hnsw_index_with_the_given_settings():
    # Looks at the private index on purpose: the contract tests would also pass if "hnsw" built a flat index.
    import faiss

    store = FaissVectorStore(index_type="hnsw", ef_search=128)
    store.build(IDS, VECTORS)

    assert isinstance(store._index, faiss.IndexHNSWFlat)
    assert store._index.hnsw.efSearch == 128


# --- registry ---------------------------------------------------------------

@pytest.mark.parametrize("name, store_class", [("exact", ExactVectorStore), ("faiss", FaissVectorStore)])
def test_every_store_is_registered_and_built_by_name(name, store_class):
    from rag_eval.vectorstores.registry import build_vector_store

    assert isinstance(build_vector_store(name, {}), store_class)


def test_faiss_settings_from_a_config_reach_the_store():
    from rag_eval.vectorstores.registry import build_vector_store

    store = build_vector_store("faiss", {"index_type": "hnsw", "ef_search": 32})

    assert (store.index_type, store.ef_search) == ("hnsw", 32)


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
