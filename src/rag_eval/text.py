"""Text preparation shared by the retrievers.

Responsible for turning a document dict (title + text) into the string that gets
indexed, and for tokenizing text for the lexical retriever.
(Named text.py rather than tokenize.py to avoid shadowing the stdlib module.)
"""


def doc_to_text(doc: dict[str, str]) -> str:
    """Combine title and body into the single string that gets indexed."""
    # TODO: decide how to combine title + text (and whether to include the title at all)
    raise NotImplementedError


def tokenize(text: str) -> list[str]:
    """Split text into tokens for BM25."""
    # TODO: lowercase, split, optional stopword removal / stemming
    # TODO: the SAME function must be applied to documents and queries
    raise NotImplementedError
