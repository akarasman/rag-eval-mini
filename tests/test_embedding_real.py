"""Slow test with the real embedding model (downloads it on first use). Run with: pytest -m slow"""
import numpy as np
import pytest

from rag_eval.retrievers.dense import DenseRetriever

pytest.importorskip("sentence_transformers")
pytestmark = pytest.mark.slow

CORPUS = {
    "heart": {"title": "Myocardial infarction", "text": "A heart attack occurs when blood flow to the heart muscle is blocked."},
    "bone": {"title": "Osteoporosis", "text": "Low bone density increases fracture risk, and vitamin D supports bone health."},
    "tumor": {"title": "p53", "text": "The p53 protein suppresses tumor growth and is often mutated in cancer."},
}


def test_the_real_model_matches_meaning_not_just_words(tmp_path):
    retriever = DenseRetriever(cache_dir=str(tmp_path / "cache"))
    retriever.index(CORPUS)

    # No word in the query appears in the matching document: BM25 would score it zero.
    top_id, top_score = retriever.search("cardiac arrest caused by a blocked artery", k=1)[0]

    assert top_id == "heart"
    assert 0.0 < top_score <= 1.0 + 1e-5


def test_the_real_embedder_returns_unit_length_float32_vectors():
    from rag_eval.embedding import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    vectors = embedder.embed_documents(["first text", "second text"])
    query = embedder.embed_query("a question")

    assert vectors.dtype == np.float32 and vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert query.shape == (384,) and np.isclose(np.linalg.norm(query), 1.0, atol=1e-4)


def test_the_fingerprint_changes_with_prefixes_and_length_limit():
    from rag_eval.embedding import SentenceTransformerEmbedder

    base = SentenceTransformerEmbedder().fingerprint

    assert SentenceTransformerEmbedder(query_prefix="query: ").fingerprint != base
    assert SentenceTransformerEmbedder(max_seq_length=64).fingerprint != base
