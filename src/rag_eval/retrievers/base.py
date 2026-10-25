"""Common retriever interface.

Every retriever takes a query string and returns a ranked list of
(doc_id, score) pairs, best first. The evaluation loop and the hybrid retriever
rely only on this interface.
"""
from typing import Protocol

Ranking = list[tuple[str, float]]  # [(doc_id, score), ...] best first


class Retriever(Protocol):
    name: str

    def index(self, corpus: dict[str, dict[str, str]]) -> None:
        """Build whatever structure is needed to search the corpus."""
        ...

    def search(self, query: str, k: int = 10) -> Ranking:
        """Return the top-k documents for the query, best first.

        The ranking has at most k entries and may be shorter, or empty: a retriever returns
        only documents it has evidence for and never pads with filler. Callers must not
        assume len(ranking) == k.
        """
        ...

# TODO: decide whether Protocol, an ABC, or plain duck typing suits you
# TODO: decide whether scores must be comparable across retrievers (matters for fusion)
