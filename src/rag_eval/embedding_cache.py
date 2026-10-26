"""On-disk cache for corpus embeddings.

Embedding the corpus is by far the slowest step of dense retrieval, so the vectors are saved and
reused. They are stored under a key that is a hash of everything that determines the vectors: the
embedder's fingerprint (model and settings) and the ordered document IDs and texts. A different model,
setting, document or document order therefore gets a different file, and stale vectors can never be
picked up.

The only public name is EmbeddingCache; the key and the file layout are private.
"""
import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np


def _cache_key(fingerprint: str, ids: List[str], texts: List[str]) -> str:
    """Hash of the embedder fingerprint and the ordered (id, text) pairs.

    Args:
        fingerprint: Embedder.fingerprint.
        ids: document IDs, in the order they are embedded.
        texts: the text embedded for each ID.

    Returns:
        A hex digest. Changing the fingerprint, any ID or text, or the order changes it. Every part
        is length-prefixed, so moving characters between an ID and its text cannot give the same hash.
    """
    digest = hashlib.sha256()
    for part in (fingerprint, str(len(ids))):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    for doc_id, text in zip(ids, texts):
        for part in (doc_id, text):
            encoded = part.encode("utf-8")
            digest.update(str(len(encoded)).encode("ascii") + b":" + encoded)
    return digest.hexdigest()


class EmbeddingCache:
    """Saves and loads the embeddings of a whole corpus in a directory.

    Args:
        cache_dir: where the cache files live. It is created on the first save, not before.
    """

    def __init__(self, cache_dir: Path):
        self._cache_dir = Path(cache_dir)

    def _path(self, fingerprint: str, ids: List[str], texts: List[str]) -> Path:
        return self._cache_dir / f"embeddings-{_cache_key(fingerprint, ids, texts)[:32]}.npy"

    def load(self, fingerprint: str, ids: List[str], texts: List[str]) -> Optional[np.ndarray]:
        """Look up the embeddings of exactly this corpus under exactly this embedder.

        An unreadable file, or one whose number of rows does not match the number of documents, counts
        as a miss and is logged, so a damaged cache costs a recomputation and never a wrong result.

        Args:
            fingerprint: Embedder.fingerprint.
            ids: document IDs, in the order they would be embedded.
            texts: the text embedded for each ID.

        Returns:
            The cached (len(ids), dim) array, or None if there is no usable cache for this input.
        """
        path = self._path(fingerprint, ids, texts)
        if not path.exists():
            return None
        try:
            vectors = np.load(path, allow_pickle=False)
        except (OSError, ValueError) as error:
            logging.warning(f"Ignoring unreadable embedding cache {path}: {error}")
            return None
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            logging.warning(f"Ignoring embedding cache {path}: shape {vectors.shape}, expected {len(ids)} rows")
            return None
        return vectors

    def save(self, fingerprint: str, ids: List[str], texts: List[str], vectors: np.ndarray) -> None:
        """Store the embeddings of this corpus under this embedder, replacing any earlier ones.

        The file is written to a temporary name first, so an interrupted run cannot leave a
        half-written cache file behind.

        Args:
            fingerprint: Embedder.fingerprint.
            ids: document IDs, in the order they were embedded.
            texts: the text embedded for each ID.
            vectors: the (len(ids), dim) array to store, one row per document.

        Raises:
            ValueError: if vectors does not have one row per document.
        """
        if vectors.ndim != 2 or vectors.shape[0] != len(ids):
            raise ValueError(f"expected one vector per document ({len(ids)}), got shape {vectors.shape}")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(fingerprint, ids, texts)
        temporary = path.with_suffix(".tmp.npy")
        np.save(temporary, vectors, allow_pickle=False)
        os.replace(temporary, path)
