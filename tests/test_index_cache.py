"""Tests for FAISS persistence, the index cache, and the dense retriever's use of it."""
import logging
import zlib
from typing import List

import faiss
import numpy as np
import pytest

from rag_eval.index_cache import IndexCache
from rag_eval.retrievers.dense import DenseRetriever
from rag_eval.vectorstores.base import PersistableVectorStore
from rag_eval.vectorstores.exact import ExactVectorStore
from rag_eval.vectorstores.faiss_store import FaissVectorStore


def _random_unit_vectors(n: int, dim: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [str(i) for i in range(n)], vectors


def _built_faiss(index_type: str = "flat", n: int = 200, **params) -> FaissVectorStore:
    ids, vectors = _random_unit_vectors(n)
    store = FaissVectorStore(index_type=index_type, **params)
    store.build(ids, vectors)
    return store


def _results(store, queries: np.ndarray, k: int = 10):
    return [store.search(query, k) for query in queries]


# --- which stores can persist -------------------------------------------------

def test_only_the_faiss_store_is_persistable():
    assert isinstance(FaissVectorStore(), PersistableVectorStore)
    assert not isinstance(ExactVectorStore(), PersistableVectorStore)


# --- FaissVectorStore.save / load ---------------------------------------------

@pytest.mark.parametrize("index_type", ["flat", "hnsw"])
def test_save_then_load_gives_identical_search_results(tmp_path, index_type):
    original = _built_faiss(index_type)
    original.save(tmp_path / "index.npz")
    loaded = FaissVectorStore(index_type=index_type)

    loaded.load(tmp_path / "index.npz")

    queries = _random_unit_vectors(10, seed=1)[1]
    assert _results(loaded, queries) == _results(original, queries)
    assert len(loaded) == len(original)


def test_save_writes_exactly_the_given_path(tmp_path):
    _built_faiss().save(tmp_path / "index.bin")  # np.savez would have appended ".npz" to a path

    assert [path.name for path in tmp_path.iterdir()] == ["index.bin"]


def test_a_loaded_hnsw_store_uses_its_own_ef_search_not_the_saved_one(tmp_path):
    _built_faiss("hnsw", ef_search=16).save(tmp_path / "index.npz")
    loaded = FaissVectorStore(index_type="hnsw", ef_search=128)

    loaded.load(tmp_path / "index.npz")

    assert loaded._index.hnsw.efSearch == 128  # private on purpose: nothing else shows the setting


def test_save_before_build_raises(tmp_path):
    with pytest.raises(RuntimeError, match="build"):
        FaissVectorStore().save(tmp_path / "index.npz")


def _write_mismatched_file(path):
    # an index of 200 vectors with only 3 ids
    np.savez(path, index=faiss.serialize_index(_built_faiss()._index), ids=np.array(["a", "b", "c"]))


def _write_file_without_ids(path):
    np.savez(path, index=faiss.serialize_index(_built_faiss()._index))


def _write_garbage(path):
    path.write_bytes(b"not a numpy file")


def _write_truncated(path):
    _built_faiss().save(path)
    path.write_bytes(path.read_bytes()[:50])


@pytest.mark.parametrize("write_bad_file", [
    _write_garbage, _write_truncated, _write_mismatched_file, _write_file_without_ids,
])
def test_load_rejects_a_bad_file_and_keeps_the_previous_content(tmp_path, write_bad_file):
    store = _built_faiss(n=50)
    query = _random_unit_vectors(1, seed=2)[1][0]
    before = store.search(query, 5)
    write_bad_file(tmp_path / "bad.npz")

    with pytest.raises(ValueError):
        store.load(tmp_path / "bad.npz")

    assert len(store) == 50
    assert store.search(query, 5) == before


def test_load_of_a_missing_file_is_a_value_error(tmp_path):
    with pytest.raises(ValueError):
        FaissVectorStore().load(tmp_path / "nothing.npz")


# --- IndexCache ------------------------------------------------------------------

CORPUS_KEY = "corpus-key"
STORE_NAME = "faiss"
STORE_PARAMS = {"index_type": "hnsw", "hnsw_m": 16}


def test_load_is_a_miss_when_nothing_was_saved(tmp_path):
    store = FaissVectorStore()

    assert IndexCache(tmp_path).load(store, CORPUS_KEY, STORE_NAME, STORE_PARAMS) is False
    assert len(store) == 0


def test_the_cache_directory_is_created_on_the_first_save_not_before(tmp_path):
    cache = IndexCache(tmp_path / "cache")
    assert not (tmp_path / "cache").exists()

    cache.save(_built_faiss(), CORPUS_KEY, STORE_NAME, STORE_PARAMS)

    assert (tmp_path / "cache").is_dir()


def test_save_then_load_restores_the_store(tmp_path):
    cache = IndexCache(tmp_path)
    original = _built_faiss("hnsw")
    cache.save(original, CORPUS_KEY, STORE_NAME, STORE_PARAMS)
    restored = FaissVectorStore(index_type="hnsw")

    assert cache.load(restored, CORPUS_KEY, STORE_NAME, STORE_PARAMS) is True

    queries = _random_unit_vectors(10, seed=1)[1]
    assert _results(restored, queries) == _results(original, queries)


def test_a_second_save_replaces_the_first(tmp_path):
    cache = IndexCache(tmp_path)
    cache.save(_built_faiss(n=200), CORPUS_KEY, STORE_NAME, STORE_PARAMS)
    cache.save(_built_faiss(n=30), CORPUS_KEY, STORE_NAME, STORE_PARAMS)
    restored = FaissVectorStore()

    cache.load(restored, CORPUS_KEY, STORE_NAME, STORE_PARAMS)

    assert len(restored) == 30
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("corpus_key, store_name, store_params", [
    ("other-corpus", STORE_NAME, STORE_PARAMS),
    (CORPUS_KEY, "other-store", STORE_PARAMS),
    (CORPUS_KEY, STORE_NAME, {"index_type": "hnsw", "hnsw_m": 32}),
    (CORPUS_KEY, STORE_NAME, {"index_type": "hnsw"}),
])
def test_a_different_corpus_store_or_parameter_is_a_miss(tmp_path, corpus_key, store_name, store_params):
    cache = IndexCache(tmp_path)
    cache.save(_built_faiss(), CORPUS_KEY, STORE_NAME, STORE_PARAMS)

    assert cache.load(FaissVectorStore(), corpus_key, store_name, store_params) is False


def test_the_order_of_the_parameters_does_not_matter(tmp_path):
    cache = IndexCache(tmp_path)
    cache.save(_built_faiss(), CORPUS_KEY, STORE_NAME, {"index_type": "hnsw", "hnsw_m": 16})

    assert cache.load(FaissVectorStore(), CORPUS_KEY, STORE_NAME, {"hnsw_m": 16, "index_type": "hnsw"}) is True


def test_a_damaged_cache_file_is_a_logged_miss(tmp_path, caplog):
    cache = IndexCache(tmp_path)
    cache.save(_built_faiss(), CORPUS_KEY, STORE_NAME, STORE_PARAMS)
    (cache_file,) = tmp_path.glob("index-*.npz")
    cache_file.write_bytes(b"junk")
    store = FaissVectorStore()

    with caplog.at_level(logging.WARNING):
        assert cache.load(store, CORPUS_KEY, STORE_NAME, STORE_PARAMS) is False

    assert "Ignoring unusable index cache" in caplog.text
    assert len(store) == 0


def test_save_leaves_no_temporary_file_behind(tmp_path):
    IndexCache(tmp_path).save(_built_faiss(), CORPUS_KEY, STORE_NAME, STORE_PARAMS)

    (only_file,) = tmp_path.iterdir()
    assert only_file.name.startswith("index-") and ".tmp" not in only_file.name


# --- DenseRetriever with the index cache ---------------------------------------

CORPUS = {
    "d1": {"title": "Tumor suppressor", "text": "The p53 protein suppresses tumor growth."},
    "d2": {"title": "Heart disease", "text": "Myocardial infarction damages heart muscle."},
    "d3": {"title": "Diet", "text": "Vitamin D intake and bone density."},
    "d4": {"title": "Gene expression", "text": "Mutations in p53 alter gene expression."},
}


class _FakeEmbedder:
    """Normalized bag-of-words vectors from hashed words. Records what it was asked to embed."""

    fingerprint = "fake-v1"

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.document_calls: List[List[str]] = []

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().replace(".", " ").split():
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        self.document_calls.append(list(texts))
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def _retriever(tmp_path, embedder=None, **kwargs) -> DenseRetriever:
    kwargs.setdefault("cache_dir", str(tmp_path / "cache"))
    kwargs.setdefault("store", "faiss")
    return DenseRetriever(embedder=embedder or _FakeEmbedder(), **kwargs)


def _index_files(tmp_path):
    return sorted((tmp_path / "cache").glob("index-*.npz"))


@pytest.fixture
def build_calls(monkeypatch) -> List[int]:
    """Sizes of every FaissVectorStore.build call made during the test."""
    calls: List[int] = []
    original = FaissVectorStore.build

    def spy(self, ids, vectors):
        calls.append(len(ids))
        return original(self, ids, vectors)

    monkeypatch.setattr(FaissVectorStore, "build", spy)
    return calls


def test_the_first_index_call_saves_the_index_next_to_the_embeddings(tmp_path):
    _retriever(tmp_path).index(CORPUS)

    assert len(_index_files(tmp_path)) == 1
    assert len(list((tmp_path / "cache").glob("embeddings-*.npy"))) == 1


def test_a_second_index_call_skips_embedding_and_build(tmp_path, build_calls):
    first = _retriever(tmp_path)
    first.index(CORPUS)
    embedder = _FakeEmbedder()
    second = _retriever(tmp_path, embedder)

    second.index(CORPUS)

    assert embedder.document_calls == []
    assert build_calls == [len(CORPUS)]  # only the first retriever built an index
    assert second.search("heart muscle", k=4) == first.search("heart muscle", k=4)


def test_changing_a_store_parameter_rebuilds_the_index_but_not_the_embeddings(tmp_path, build_calls):
    _retriever(tmp_path, store_params={"index_type": "hnsw", "hnsw_m": 8}).index(CORPUS)
    embedder = _FakeEmbedder()

    _retriever(tmp_path, embedder, store_params={"index_type": "hnsw", "hnsw_m": 16}).index(CORPUS)

    assert embedder.document_calls == []  # the embeddings came from their own cache
    assert build_calls == [len(CORPUS), len(CORPUS)]
    assert len(_index_files(tmp_path)) == 2


def test_a_damaged_index_file_is_rebuilt_from_the_cached_embeddings(tmp_path, build_calls):
    first = _retriever(tmp_path)
    first.index(CORPUS)
    _index_files(tmp_path)[0].write_bytes(b"junk")
    embedder = _FakeEmbedder()
    second = _retriever(tmp_path, embedder)

    second.index(CORPUS)

    assert embedder.document_calls == []
    assert len(build_calls) == 2
    assert second.search("heart muscle", k=4) == first.search("heart muscle", k=4)


def test_a_store_without_save_and_load_is_indexed_normally(tmp_path):
    retriever = _retriever(tmp_path, store="exact")

    retriever.index(CORPUS)

    assert _index_files(tmp_path) == []
    assert retriever.search("heart muscle", k=1)[0][0] == "d2"


def test_use_cache_false_writes_no_index_file(tmp_path):
    _retriever(tmp_path, use_cache=False).index(CORPUS)

    assert not (tmp_path / "cache").exists()


def test_rebuild_index_builds_again_but_reuses_the_embeddings(tmp_path, build_calls):
    _retriever(tmp_path).index(CORPUS)
    embedder = _FakeEmbedder()

    _retriever(tmp_path, embedder, rebuild_index=True).index(CORPUS)

    assert embedder.document_calls == []
    assert build_calls == [len(CORPUS), len(CORPUS)]
    assert len(_index_files(tmp_path)) == 1  # replaced, not added


def test_rebuild_embeddings_embeds_again_and_rebuilds_the_index(tmp_path, build_calls):
    _retriever(tmp_path).index(CORPUS)
    embedder = _FakeEmbedder()

    _retriever(tmp_path, embedder, rebuild_embeddings=True).index(CORPUS)

    assert len(embedder.document_calls) == 1
    assert build_calls == [len(CORPUS), len(CORPUS)]


def test_after_a_rebuild_the_next_normal_run_uses_the_refreshed_cache(tmp_path, build_calls):
    _retriever(tmp_path).index(CORPUS)
    _retriever(tmp_path, rebuild_embeddings=True).index(CORPUS)
    embedder = _FakeEmbedder()

    _retriever(tmp_path, embedder).index(CORPUS)

    assert embedder.document_calls == []
    assert len(build_calls) == 2  # the third run built nothing


def test_the_rebuild_flags_do_nothing_with_use_cache_off(tmp_path):
    retriever = _retriever(tmp_path, use_cache=False, rebuild_index=True, rebuild_embeddings=True)

    retriever.index(CORPUS)

    assert not (tmp_path / "cache").exists()
    assert retriever.search("heart muscle", k=1)[0][0] == "d2"
