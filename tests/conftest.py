"""Shared fixtures.

The Qdrant fixtures need a server on localhost:6333, e.g.
    docker run -d --name qdrant -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
and skip every test that uses them when it cannot be reached.

`scripted_llm` is a fake LLM for the generation tests: no network and no model.
"""
import uuid
from typing import Callable, Iterator, List, Optional, Tuple, Type, Union

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


class ScriptedLLM:
    """A fake LLM that returns canned replies in order and records every call.

    A reply that is an Exception instance is raised instead of returned, to simulate a failing backend.
    Asking for more replies than were scripted fails the test.
    """
    name = "scripted"

    def __init__(self, replies: List[Union[str, Exception]]):
        self._replies = list(replies)
        self.calls: List[Tuple[str, Optional[str]]] = []  # (prompt, system) of every generate() call

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        self.calls.append((prompt, system))
        if len(self.calls) > len(self._replies):
            raise AssertionError(f"ScriptedLLM got call {len(self.calls)} but only {len(self._replies)} replies")
        reply = self._replies[len(self.calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def scripted_llm() -> Type[ScriptedLLM]:
    """The ScriptedLLM class: `scripted_llm(["reply 1", "reply 2"])`."""
    return ScriptedLLM
