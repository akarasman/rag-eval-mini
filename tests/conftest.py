"""Shared fixtures.

The Qdrant fixtures need a server on localhost:6333, e.g.
    docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
and skip every test that uses them when it cannot be reached.
"""
import uuid
from typing import Callable, Iterator, List

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from rag_eval.vectorstores.qdrant_store import QdrantVectorStore

QDRANT_URL = "http://localhost:6333"


@pytest.fixture(scope="session")
def qdrant_client() -> QdrantClient:
    """A client for the test server; skips the requesting test if there is none."""
    client = QdrantClient(url=QDRANT_URL, timeout=2, check_compatibility=False)
    try:
        client.get_collections()
    except ResponseHandlingException:
        pytest.skip(f"no Qdrant server at {QDRANT_URL}")
    return client


@pytest.fixture
def qdrant_store_factory(qdrant_client: QdrantClient) -> Iterator[Callable[..., QdrantVectorStore]]:
    """Makes QdrantVectorStores on collections with unique names, and deletes them after the test."""
    created: List[str] = []

    def make(**params) -> QdrantVectorStore:
        name = f"test_{uuid.uuid4().hex}"
        created.append(name)
        return QdrantVectorStore(collection=name, url=QDRANT_URL, **params)

    yield make

    for name in created:
        if qdrant_client.collection_exists(name):
            qdrant_client.delete_collection(name)
