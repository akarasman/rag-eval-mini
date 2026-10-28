"""Tests specific to the Qdrant store. The generic contract is in test_vectorstores.py.

Tests that take `qdrant_store_factory` need a Qdrant server and are skipped without one (see conftest.py);
the others need no server.
"""
import numpy as np
import pytest

from rag_eval.vectorstores.exact import ExactVectorStore
from rag_eval.vectorstores.qdrant_store import QdrantVectorStore
from rag_eval.vectorstores.registry import build_vector_store


def _random_unit_vectors(n: int, dim: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [str(i) for i in range(n)], vectors


def _ids_of(result):
    return [vector_id for vector_id, _ in result]


# --- no server needed ----------------------------------------------------------------

@pytest.mark.parametrize("params", [
    {"collection": ""},
    {"hnsw_m": 0},
    {"ef_construct": -1},
    {"ef_search": 0},
    {"timeout": 0},
])
def test_rejects_bad_parameters(params):
    with pytest.raises(ValueError, match=next(iter(params))):  # the message names the bad parameter
        QdrantVectorStore(**params)


def test_settings_from_a_config_reach_the_store():
    store = build_vector_store("qdrant", {"collection": "mine", "ef_search": 32})

    assert (store.collection, store.ef_search) == ("mine", 32)


def test_an_unreachable_server_is_reported_as_one_clear_error():
    store = QdrantVectorStore(url="http://localhost:6399", timeout=1)  # nothing listens here

    with pytest.raises(RuntimeError, match="cannot reach the Qdrant server at http://localhost:6399"):
        store.build(["a"], np.array([[1.0, 0.0]]))


def test_a_build_that_fails_validation_does_not_contact_the_server():
    store = QdrantVectorStore(url="http://localhost:6399", timeout=1)

    with pytest.raises(ValueError, match="unique"):  # not RuntimeError: the input is checked first
        store.build(["a", "a"], np.zeros((2, 2)))


# --- server needed --------------------------------------------------------------------

def test_ids_that_are_not_valid_qdrant_ids_come_back_unchanged(qdrant_store_factory):
    ids = ["4346436", "PMID:123", "a b/c", "ünïcode-ΩΩ", ""]
    store = qdrant_store_factory()
    store.build(ids, np.eye(5, dtype=np.float32))

    result = store.search(np.array([0, 0, 0, 0, 1.0]), k=5)

    assert result[0][0] == ""
    assert sorted(_ids_of(result)) == sorted(ids)


def test_scores_are_inner_products_even_for_vectors_that_are_not_unit_length(qdrant_store_factory):
    store = qdrant_store_factory()
    store.build(["a", "b"], np.array([[3.0, 0.0], [0.0, 2.0]]))  # Qdrant's cosine would rescale these

    result = store.search(np.array([1.0, 1.0]), k=2)

    assert result == [("a", pytest.approx(3.0)), ("b", pytest.approx(2.0))]


def test_finds_most_of_the_true_neighbours(qdrant_store_factory):
    ids, vectors = _random_unit_vectors(500)
    exact, qdrant = ExactVectorStore(), qdrant_store_factory()
    exact.build(ids, vectors)
    qdrant.build(ids, vectors)

    queries = _random_unit_vectors(20, seed=1)[1]
    recalls = [
        len(set(_ids_of(qdrant.search(q, k=10))) & set(_ids_of(exact.search(q, k=10)))) / 10
        for q in queries
    ]

    assert np.mean(recalls) >= 0.9


def test_a_second_build_may_change_the_dimension(qdrant_store_factory):
    store = qdrant_store_factory()
    store.build(["a"], np.array([[1.0, 0.0]]))

    store.build(["x", "y"], np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    assert _ids_of(store.search(np.array([0.0, 1.0, 0.0]), k=2)) == ["y", "x"]


def test_the_hnsw_settings_are_applied_to_the_collection(qdrant_store_factory, qdrant_client):
    store = qdrant_store_factory(hnsw_m=8, ef_construct=50)
    store.build(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]))

    config = qdrant_client.get_collection(store.collection).config.params.vectors

    assert (config.hnsw_config.m, config.hnsw_config.ef_construct) == (8, 50)
    assert config.distance.value == "Dot"


def test_ef_search_is_sent_with_every_query(qdrant_store_factory, monkeypatch):
    store = qdrant_store_factory(ef_search=77)
    store.build(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    sent = []
    original = store._client.query_points

    def spy(*args, **kwargs):
        sent.append(kwargs["search_params"])
        return original(*args, **kwargs)

    monkeypatch.setattr(store._client, "query_points", spy)

    store.search(np.array([1.0, 0.0]), k=1)

    assert sent[0].hnsw_ef == 77


def test_without_ef_search_the_server_default_is_used(qdrant_store_factory, monkeypatch):
    store = qdrant_store_factory()
    store.build(["a"], np.array([[1.0, 0.0]]))
    sent = []
    original = store._client.query_points

    def spy(*args, **kwargs):
        sent.append(kwargs["search_params"])
        return original(*args, **kwargs)

    monkeypatch.setattr(store._client, "query_points", spy)

    store.search(np.array([1.0, 0.0]), k=1)

    assert sent == [None]


def test_the_data_is_in_the_server_not_in_the_store_object(qdrant_store_factory, qdrant_client):
    store = qdrant_store_factory()
    store.build(["a", "b", "c"], np.eye(3, dtype=np.float32))

    assert qdrant_client.count(store.collection, exact=True).count == 3
