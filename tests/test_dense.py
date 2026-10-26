"""Tests for the dense retriever, using a fake embedder (no neural network is loaded)."""
import zlib
from typing import List

import numpy as np
import pytest

from rag_eval.retrievers.dense import DenseRetriever
from rag_eval.retrievers.registry import build_retriever

CORPUS = {
    "d1": {"title": "Tumor suppressor", "text": "The p53 protein suppresses tumor growth."},
    "d2": {"title": "Heart disease", "text": "Myocardial infarction damages heart muscle."},
    "d3": {"title": "Diet", "text": "Vitamin D intake and bone density."},
    "d4": {"title": "Gene expression", "text": "Mutations in p53 alter gene expression."},
}


class _FakeEmbedder:
    """Bag-of-words vectors from hashed words, normalized. Records what it was asked to embed."""

    def __init__(self, fingerprint: str = "fake-v1", dim: int = 64, normalize: bool = True):
        self.fingerprint = fingerprint
        self.dim = dim
        self.normalize = normalize
        self.document_calls: List[List[str]] = []
        self.query_calls: List[str] = []

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().replace(".", " ").split():
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if self.normalize and norm > 0 else vector

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        self.document_calls.append(list(texts))
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls.append(text)
        return self._vector(text)


def _retriever(tmp_path, embedder=None, **kwargs) -> DenseRetriever:
    kwargs.setdefault("cache_dir", str(tmp_path / "cache"))
    return DenseRetriever(embedder=embedder or _FakeEmbedder(), **kwargs)


def _ids(ranking):
    return [doc_id for doc_id, _ in ranking]


# --- searching --------------------------------------------------------------

def test_ranks_the_topically_matching_document_first(tmp_path):
    retriever = _retriever(tmp_path)
    retriever.index(CORPUS)

    assert _ids(retriever.search("heart muscle damage", k=4))[0] == "d2"
    assert _ids(retriever.search("vitamin bone", k=4))[0] == "d3"


def test_returns_scores_in_descending_order_as_cosine_similarities(tmp_path):
    retriever = _retriever(tmp_path)
    retriever.index(CORPUS)

    scores = [score for _, score in retriever.search("p53 tumor", k=4)]

    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 - 1e-5 <= score <= 1.0 + 1e-5 for score in scores)


def test_always_returns_k_results_when_the_corpus_is_large_enough(tmp_path):
    retriever = _retriever(tmp_path)
    retriever.index(CORPUS)

    assert len(retriever.search("unknownword", k=3)) == 3  # unlike BM25, every document has a similarity
    assert len(retriever.search("p53", k=100)) == len(CORPUS)


def test_indexes_the_title_together_with_the_text(tmp_path):
    embedder = _FakeEmbedder()
    retriever = _retriever(tmp_path, embedder)

    retriever.index({"d": {"title": "A Title", "text": "the body"}})

    assert embedder.document_calls == [["A Title. the body"]]


def test_a_document_is_its_own_best_match(tmp_path):
    retriever = _retriever(tmp_path)
    retriever.index(CORPUS)
    text = "The p53 protein suppresses tumor growth."

    top_id, top_score = retriever.search(f"Tumor suppressor. {text}", k=1)[0]

    assert top_id == "d1"
    assert top_score == pytest.approx(1.0, abs=1e-5)


def test_reindexing_replaces_the_previous_corpus(tmp_path):
    retriever = _retriever(tmp_path)
    retriever.index(CORPUS)

    retriever.index({"x": {"title": "", "text": "heart"}})

    assert _ids(retriever.search("heart", k=5)) == ["x"]


# --- errors -----------------------------------------------------------------

def test_search_before_index_raises(tmp_path):
    with pytest.raises(RuntimeError, match="index"):
        _retriever(tmp_path).search("p53")


@pytest.mark.parametrize("k", [0, -1])
def test_rejects_non_positive_k_without_embedding_the_query(tmp_path, k):
    embedder = _FakeEmbedder()
    retriever = _retriever(tmp_path, embedder)
    retriever.index(CORPUS)

    with pytest.raises(ValueError, match="k"):
        retriever.search("p53", k=k)

    assert embedder.query_calls == []


def test_an_embedder_that_does_not_normalize_is_rejected(tmp_path):
    retriever = _retriever(tmp_path, _FakeEmbedder(normalize=False))

    with pytest.raises(ValueError, match="unit-length"):
        retriever.index(CORPUS)


def test_a_failed_index_leaves_the_previous_index_in_place(tmp_path):
    retriever = _retriever(tmp_path)
    retriever.index(CORPUS)

    with pytest.raises(ValueError):
        retriever.index({})  # an empty corpus cannot be indexed

    assert _ids(retriever.search("heart muscle", k=1)) == ["d2"]


def test_unknown_store_and_bad_store_params_fail_when_the_retriever_is_built(tmp_path):
    with pytest.raises(ValueError, match="nope"):
        _retriever(tmp_path, store="nope")
    with pytest.raises(ValueError, match="typo"):
        _retriever(tmp_path, store="exact", store_params={"typo": 1})


# --- embedding cache --------------------------------------------------------

def test_a_second_index_of_the_same_corpus_loads_embeddings_from_the_cache(tmp_path):
    first = _FakeEmbedder()
    _retriever(tmp_path, first).index(CORPUS)
    second = _FakeEmbedder()
    retriever = _retriever(tmp_path, second)

    retriever.index(CORPUS)

    assert len(first.document_calls) == 1
    assert second.document_calls == []  # nothing was embedded
    assert _ids(retriever.search("heart muscle", k=1)) == ["d2"]  # and it still works


def test_cached_vectors_give_the_same_ranking_as_fresh_ones(tmp_path):
    fresh = _retriever(tmp_path)
    fresh.index(CORPUS)
    cached = _retriever(tmp_path)
    cached.index(CORPUS)

    assert cached.search("p53 gene", k=4) == fresh.search("p53 gene", k=4)


def test_a_changed_document_text_is_not_served_from_the_cache(tmp_path):
    _retriever(tmp_path).index(CORPUS)
    embedder = _FakeEmbedder()
    changed = {**CORPUS, "d3": {"title": "Diet", "text": "Something else entirely."}}

    _retriever(tmp_path, embedder).index(changed)

    assert len(embedder.document_calls) == 1


def test_a_different_embedder_fingerprint_is_not_served_from_the_cache(tmp_path):
    _retriever(tmp_path, _FakeEmbedder(fingerprint="model-a")).index(CORPUS)
    other = _FakeEmbedder(fingerprint="model-b")

    _retriever(tmp_path, other).index(CORPUS)

    assert len(other.document_calls) == 1


def test_the_cache_does_not_depend_on_the_vector_store(tmp_path):
    _retriever(tmp_path, store="exact").index(CORPUS)
    embedder = _FakeEmbedder()

    _retriever(tmp_path, embedder, store="exact", store_params={}).index(CORPUS)

    assert embedder.document_calls == []


def test_use_cache_false_neither_reads_nor_writes_the_cache(tmp_path):
    embedder = _FakeEmbedder()
    retriever = _retriever(tmp_path, embedder, use_cache=False)

    retriever.index(CORPUS)
    retriever.index(CORPUS)

    assert len(embedder.document_calls) == 2
    assert not (tmp_path / "cache").exists()


def test_a_damaged_cache_file_is_recomputed_not_trusted(tmp_path):
    _retriever(tmp_path).index(CORPUS)
    (cache_file,) = (tmp_path / "cache").glob("embeddings-*.npy")
    cache_file.write_bytes(b"not a numpy file")
    embedder = _FakeEmbedder()

    retriever = _retriever(tmp_path, embedder)
    retriever.index(CORPUS)

    assert len(embedder.document_calls) == 1
    assert _ids(retriever.search("heart muscle", k=1)) == ["d2"]


# --- construction and registry ----------------------------------------------

def test_the_embedding_model_is_not_loaded_when_the_retriever_is_built():
    retriever = build_retriever("dense", {"store": "exact", "model_name": "no/such-model"})

    assert retriever._embedder is None  # would have failed here if it tried to load a model


def test_dense_is_registered_with_its_parameters():
    retriever = build_retriever("dense", {"store": "exact", "query_prefix": "query: ", "use_cache": False})

    assert retriever.name == "dense"
    assert retriever.use_cache is False
