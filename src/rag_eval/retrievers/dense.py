"""Dense retriever (embeddings + vector store).

Embeds documents and queries with a sentence-embedding model and ranks by inner product of the
(unit-length) vectors, i.e. cosine similarity. The model sits behind the Embedder interface and the
search behind the VectorStore interface, so either can be swapped from the config. Embedding the corpus
is the slow part on CPU, so the vectors are cached on disk (see embedding_cache.py).
"""
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional, List

import numpy as np

from ..embedding import Embedder, SentenceTransformerEmbedder
from ..embedding_cache import EmbeddingCache
from ..text import doc_to_text
from ..vectorstores.base import VectorStore
from ..vectorstores.registry import build_vector_store
from .base import Ranking
from .registry import register_retriever

_UNIT_NORM_TOLERANCE = 1e-3


@register_retriever("dense")
class DenseRetriever:
    """Args:
        model_name: sentence-transformers model id (ignored if `embedder` is given).
        store: name of a registered vector store ("exact", ...).
        store_params: keyword parameters for that store.
        query_prefix: text put in front of every query (some retrieval models expect one).
        document_prefix: text put in front of every document.
        batch_size: documents embedded per forward pass.
        max_seq_length: truncate inputs to this many tokens; None keeps the model's own limit.
        use_cache: reuse and save corpus embeddings on disk.
        cache_dir: where the embedding cache lives.
        embedder: use this instead of building a SentenceTransformerEmbedder (tests inject a fake).

    The embedding model is loaded on the first index() call, not when the retriever is built, so
    constructing one (e.g. from a config) is cheap. The vector store is built immediately, so an
    unknown store name or bad store parameters fail early.
    """
    name = "dense"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        store: str = "exact",
        store_params: Optional[Dict[str, Any]] = None,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 64,
        max_seq_length: Optional[int] = None,
        use_cache: bool = True,
        cache_dir: str = "cache",
        embedder: Optional[Embedder] = None,
    ):
        self.model_name = model_name
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self._cache: Optional[EmbeddingCache] = EmbeddingCache(self.cache_dir) if use_cache else None
        self._embedder_settings = {
            "model_name": model_name,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
            "batch_size": batch_size,
            "max_seq_length": max_seq_length,
        }
        self._embedder: Optional[Embedder] = embedder
        self._store: VectorStore = build_vector_store(store, store_params or {})
        self._doc_ids: list = []
        self._indexed = False

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(**self._embedder_settings)
        return self._embedder

    def _build_vectors(self, embedder: Embedder, texts: List[str], doc_ids: List[str]) -> np.ndarray:
        start = perf_counter()
        vectors = embedder.embed_documents(texts)
        logging.info(f"Embedded {len(doc_ids)} documents in {perf_counter() - start:.1f}s")
        _check_unit_length(vectors)
        if self._cache is not None:
            self._cache.save(embedder.fingerprint, doc_ids, texts, vectors)

        return vectors

    def _load_cached_vectors(self, embedder: Embedder, texts: List[str], doc_ids: List[str]) -> Optional[np.ndarray]:
        vectors = self._cache.load(embedder.fingerprint, doc_ids, texts)
        if vectors is not None:
            logging.info(f"Loaded {len(doc_ids)} cached embeddings from {self.cache_dir}")
        return vectors

    def index(self, corpus: Dict[str, Dict[str, str]]) -> None:
        """Embed every document (or load the cached embeddings) and build the vector store.

        Raises:
            ValueError: if the corpus is empty, or the embedder does not return unit-length vectors.
        """
        embedder = self._get_embedder()
        doc_ids = list(corpus)
        texts = [doc_to_text(corpus[doc_id]) for doc_id in doc_ids]

        vectors = None
        if self._cache is not None:
            vectors = self._load_cached_vectors(embedder, texts, doc_ids)
        if vectors is None:
            vectors = self._build_vectors(embedder, texts, doc_ids)

        self._store.build(doc_ids, vectors)
        self._doc_ids = doc_ids
        self._indexed = True

    def search(self, query: str, k: int = 10) -> Ranking:
        """Top-k documents by cosine similarity, best first.

        Raises:
            RuntimeError: if index() hasn't been called.
            ValueError: if k <= 0.
        """
        if not self._indexed:
            raise RuntimeError("index() must be called before search()")
        if k <= 0:
            raise ValueError("k must be positive")

        query_vector = self._get_embedder().embed_query(query)
        _check_unit_length(query_vector[np.newaxis, :])
        return self._store.search(query_vector, k)


def _check_unit_length(vectors: np.ndarray) -> None:
    """The stores rank by inner product, which is cosine similarity only for unit vectors."""
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=_UNIT_NORM_TOLERANCE):
        raise ValueError(
            f"embedder must return unit-length vectors, got norms between {norms.min():.4f} and {norms.max():.4f}"
        )
