"""Qdrant vector store: a client for a running Qdrant server (HNSW index, inner-product search).

Unlike the FAISS store, the index lives in the server process, not in this one, so start the server
first, e.g. `docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant`.

Qdrant point IDs must be integers or UUIDs, while ours are arbitrary strings. Each vector therefore gets
its row position as the point ID, and the real ID is kept in the payload and read back with the results.
The server persists the collection itself, so this store has no save/load (it is not a
PersistableVectorStore).
"""
from contextlib import contextmanager
from typing import Iterator, List, Optional

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException

from .base import SearchResult, validate_build_input, validate_search_input
from .registry import register_vector_store

_DOC_ID = "doc_id"  # payload key that holds the real ID
_UPLOAD_BATCH_SIZE = 256


@register_vector_store("qdrant")
class QdrantVectorStore:
    """Args:
        collection: name of the Qdrant collection this store owns. build() deletes and recreates it,
            so give each store its own name.
        url: address of the Qdrant server.
        hnsw_m: links per node in the HNSW graph (Qdrant's own default is 16).
        ef_construct: candidates examined while building the graph (Qdrant's default is 100).
        ef_search: candidates examined per query; None uses the server's default.
        timeout: seconds to wait for the server per request.

    Constructing the store does not contact the server; the first build() does.

    Raises:
        ValueError: if collection is empty, or hnsw_m, ef_construct, ef_search or timeout is not positive.
    """
    name = "qdrant"

    def __init__(
        self,
        collection: str = "rag_eval",
        url: str = "http://localhost:6333",
        hnsw_m: int = 16,
        ef_construct: int = 100,
        ef_search: Optional[int] = None,
        timeout: int = 30,
    ):
        if not collection:
            raise ValueError("collection must not be empty")
        if hnsw_m <= 0 or ef_construct <= 0 or timeout <= 0:
            raise ValueError("hnsw_m, ef_construct and timeout must be positive")
        if ef_search is not None and ef_search <= 0:
            raise ValueError("ef_search must be positive")
        self.collection = collection
        self.url = url
        self.hnsw_m = hnsw_m
        self.ef_construct = ef_construct
        self.ef_search = ef_search
        # check_compatibility=False: the client would otherwise call the server while being constructed.
        self._client = QdrantClient(url=url, timeout=timeout, check_compatibility=False)
        self._dim: Optional[int] = None
        self._size = 0

    @contextmanager
    def _server_errors(self) -> Iterator[None]:
        """Report an unreachable server as one clear error."""
        try:
            yield
        except ResponseHandlingException as error:
            raise RuntimeError(f"cannot reach the Qdrant server at {self.url}: {error}") from error

    def build(self, ids: List[str], vectors: np.ndarray) -> None:
        """Replace the collection's content. See VectorStore.build.

        The input is checked before the collection is touched, so bad input leaves the previous content
        in place. A failure while uploading (e.g. the server goes away) leaves the collection incomplete,
        and the store then requires another build() before it can search.

        Raises:
            ValueError: if ids and vectors are inconsistent (see validate_build_input).
            RuntimeError: if the server cannot be reached.
        """
        array = validate_build_input(ids, vectors)
        self._dim = None  # until the upload has finished, the collection must not be searched
        with self._server_errors():
            if self._client.collection_exists(self.collection):
                self._client.delete_collection(self.collection)
            self._client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(
                    size=array.shape[1],
                    distance=models.Distance.DOT,  # inner product, like every store; Qdrant never rescales it
                    hnsw_config=models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.ef_construct),
                ),
            )
            self._client.upload_collection(
                self.collection,
                vectors=array,
                ids=range(len(ids)),
                payload=[{_DOC_ID: doc_id} for doc_id in ids],
                batch_size=_UPLOAD_BATCH_SIZE,
                wait=True,  # the default is to return before the points are searchable
            )
        self._dim = array.shape[1]
        self._size = len(ids)

    def search(self, vector: np.ndarray, k: int) -> SearchResult:
        """Top-k by inner product, best first, approximate. See VectorStore.search.

        Raises:
            RuntimeError: if build() hasn't been called, or the server cannot be reached.
            ValueError: if k <= 0 or the vector has the wrong shape.
        """
        if self._dim is None:
            raise RuntimeError("build() must be called before search()")
        query = validate_search_input(vector, k, self._dim)
        search_params = models.SearchParams(hnsw_ef=self.ef_search) if self.ef_search is not None else None

        with self._server_errors():
            response = self._client.query_points(
                self.collection,
                query=query.tolist(),
                limit=k,
                with_payload=[_DOC_ID],
                search_params=search_params,
            )
        return [(point.payload[_DOC_ID], float(point.score)) for point in response.points]

    def __len__(self) -> int:
        return self._size
