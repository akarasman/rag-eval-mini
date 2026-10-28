"""Common vector store interface.

A vector store holds document vectors and answers "which stored vectors are most similar to this one?".
It knows nothing about text or models: the dense retriever embeds text and hands vectors over.

Contract shared by every store (exact, FAISS, Qdrant, ...):
- Similarity is the INNER PRODUCT, higher is better. With unit-length vectors that is cosine similarity,
  so normalizing is the caller's job (the dense retriever does it); stores never rescale vectors.
- build() replaces the whole content, like Retriever.index(); there are no incremental updates.
- search() returns at most k (id, score) pairs, best first, and fewer only if the store holds fewer
  than k vectors. Approximate stores may miss true neighbours; the exact store never does.
"""
from pathlib import Path
from typing import List, Protocol, Tuple, runtime_checkable

import numpy as np

SearchResult = List[Tuple[str, float]]  # [(vector_id, inner product), ...] best first


class VectorStore(Protocol):
    name: str

    def build(self, ids: List[str], vectors: np.ndarray) -> None:
        """Replace the store's content.

        Args:
            ids: unique identifiers, one per row of `vectors`.
            vectors: 2-D array (n, dim) of float vectors, normally L2-normalized.
        """
        ...

    def search(self, vector: np.ndarray, k: int) -> SearchResult:
        """Return the k stored vectors with the highest inner product with `vector`, best first.

        Args:
            vector: the query, a 1-D array with the same dimension as the stored vectors.
            k: the maximum number of results; must be positive.
        """
        ...

    def __len__(self) -> int:
        """Number of stored vectors (0 before build)."""
        ...


@runtime_checkable
class PersistableVectorStore(VectorStore, Protocol):
    """A store that can write its built index to a file and read it back (FAISS, yes; a server-side
    store like Qdrant keeps its own data, so it does not implement this).

    runtime_checkable: the retriever asks isinstance(store, PersistableVectorStore) to decide
    whether an index cache applies. isinstance only checks that save and load exist.
    """

    def save(self, path: Path) -> None:
        """Write the built index to `path`.

        Args:
            path: the file to write.

        Raises:
            RuntimeError: if build() hasn't been called.
        """
        ...

    def load(self, path: Path) -> None:
        """Replace the store's content with the index saved at `path`.

        Args:
            path: a file written by save() of a store with the same settings.

        Raises:
            ValueError: if the file is unreadable or inconsistent. The store keeps its previous content.
        """
        ...


def validate_build_input(ids: List[str], vectors: np.ndarray) -> np.ndarray:
    """Check what build() received and return the vectors as a private float32 (n, dim) copy.

    Shared by all stores so they reject the same bad input the same way.

    Args:
        ids: the ids passed to build().
        vectors: the vectors passed to build().

    Raises:
        ValueError: if vectors is not a non-empty 2-D array, if ids and vectors differ in length,
            or if ids contain duplicates.
    """
    array = np.array(vectors, dtype=np.float32)  # np.array copies, so later changes by the caller cannot leak in
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"vectors must be a non-empty 2-D array, got shape {array.shape}")
    if len(ids) != array.shape[0]:
        raise ValueError(f"got {len(ids)} ids for {array.shape[0]} vectors")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    return array


def validate_search_input(vector: np.ndarray, k: int, dim: int) -> np.ndarray:
    """Check what search() received and return the query as a float32 1-D array.

    Args:
        vector: the query vector passed to search().
        k: the k passed to search().
        dim: the dimension of the stored vectors.

    Raises:
        ValueError: if k <= 0, or the vector is not 1-D with `dim` entries.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    query = np.asarray(vector, dtype=np.float32)
    if query.shape != (dim,):
        raise ValueError(f"query vector must have shape ({dim},), got {query.shape}")
    return query
