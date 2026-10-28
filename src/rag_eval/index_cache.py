"""On-disk cache for built vector-store indexes.

Sits next to the embedding cache (same folder). The file name is a hash of the corpus key (see
embedding_cache.embedding_key) plus the store's name and parameters, so a different corpus, model or
store setting gets a different file and a stale index is never picked up. Changing store parameters
leaves the embedding cache untouched.

The store does the serialization (PersistableVectorStore.save / load); this module decides where the
file goes, writes it safely, and turns a missing or damaged file into a cache miss.

The only public name is IndexCache.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from .vectorstores.base import PersistableVectorStore


def _index_key(corpus_key: str, store_name: str, store_params: Dict[str, Any]) -> str:
    """Hash of the corpus key and the store's name and parameters.

    Args:
        corpus_key: embedding_key(...) of the corpus and embedder.
        store_name: the registered store name, e.g. "faiss".
        store_params: the store's constructor parameters; key order must not matter.

    Returns:
        A hex digest.
    """
    digest = hashlib.sha256()
    digest.update(corpus_key.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(store_name.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(json.dumps(store_params, sort_keys=True).encode("utf-8"))
    digest.update(b"\x00")

    return digest.hexdigest()


class IndexCache:
    """Saves and loads built indexes in a directory.

    Args:
        cache_dir: where the cache files live. It is created on the first save, not before.
    """

    def __init__(self, cache_dir: Path):
        self._cache_dir = Path(cache_dir)

    def _path(self, corpus_key: str, store_name: str, store_params: Dict[str, Any]) -> Path:
        return self._cache_dir / f"index-{_index_key(corpus_key, store_name, store_params)[:32]}.npz"

    def load(
        self, store: PersistableVectorStore, corpus_key: str, store_name: str, store_params: Dict[str, Any]
    ) -> bool:
        """Fill `store` from the cached index for exactly this corpus and store settings.

        A missing file is a plain miss. An unreadable or inconsistent file (store.load raises
        ValueError) is a miss that is also logged, so a damaged cache costs a rebuild, never a wrong result.

        Args:
            store: the store to fill.
            corpus_key: embedding_key(...) of the corpus and embedder.
            store_name: the registered store name.
            store_params: the store's constructor parameters.

        Returns:
            True if the store was filled, False on a miss.
        """
        path = self._path(corpus_key, store_name, store_params)
        if not path.exists():
            return False
        try:
            store.load(path)
        except ValueError as error:
            logging.warning(f"Ignoring unusable index cache {path}: {error}")
            return False
        return True

    def save(
        self, store: PersistableVectorStore, corpus_key: str, store_name: str, store_params: Dict[str, Any]
    ) -> None:
        """Store the built index of `store`, replacing any earlier one for the same key.

        Written to a temporary name first, then os.replace, so an interrupted run cannot leave a
        half-written file behind.

        Args:
            store: a built store.
            corpus_key: embedding_key(...) of the corpus and embedder.
            store_name: the registered store name.
            store_params: the store's constructor parameters.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._path(corpus_key, store_name, store_params)
        temporary = path.with_suffix(".tmp.npz")
        store.save(temporary)
        os.replace(temporary, path)
