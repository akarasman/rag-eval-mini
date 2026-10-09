"""Dense retriever (embeddings + vector index).

Embeds documents and queries with a sentence-embedding model and ranks by
vector similarity. Embedding the corpus is the slow part on CPU, so think about
caching it to disk.
"""
from .base import Ranking


class DenseRetriever:
    name = "dense"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # TODO: load the embedding model
        raise NotImplementedError

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        # TODO: embed all docs (batched), normalise if needed, build the index
        # TODO: cache embeddings on disk, keyed so a different model/corpus doesn't reuse them
        raise NotImplementedError

    def search(self, query: str, k: int = 10) -> Ranking:
        # TODO: embed query the same way as docs, search, map row ids back to doc_ids
        raise NotImplementedError
