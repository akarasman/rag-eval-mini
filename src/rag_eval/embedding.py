"""Turning text into vectors.

An Embedder maps texts to L2-normalized float32 vectors. The dense retriever depends only on this
interface, so tests can use a fake and never load a neural network. sentence-transformers is
imported lazily, so this module can be imported without PyTorch.
"""
import json
from typing import List, Optional, Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def fingerprint(self) -> str:
        """A string identifying everything that changes the vectors (model, prefixes, length limit).

        Embeddings are cached under a key that includes it, so two embedders that produce different
        vectors must never share a fingerprint.
        """
        ...

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """Embed documents. Returns an (n, dim) float32 array of unit-length rows."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query. Returns a (dim,) float32 array of unit length."""
        ...


class SentenceTransformerEmbedder:
    """Embeddings from a sentence-transformers model.

    Args:
        model_name: Hugging Face model id; the weights are downloaded on first use.
        query_prefix: text put in front of every query (some retrieval models expect one).
        document_prefix: text put in front of every document.
        batch_size: documents embedded per forward pass.
        max_seq_length: truncate inputs to this many tokens; None keeps the model's own limit.
        device: "cpu", "cuda", ...; None lets sentence-transformers choose.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 64,
        max_seq_length: Optional[int] = None,
        device: Optional[str] = None,
    ):
        from sentence_transformers import SentenceTransformer  # heavy import, kept out of module load

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length

    @property
    def fingerprint(self) -> str:
        return json.dumps(
            {
                "model": self.model_name,
                "query_prefix": self.query_prefix,
                "document_prefix": self.document_prefix,
                "max_seq_length": self._model.max_seq_length,
                "normalized": True,
            },
            sort_keys=True,
        )

    def _encode(self, texts: List[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32, copy=False)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]
