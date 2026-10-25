"""Text preparation shared by the retrievers.

Responsible for turning a document dict (title + text) into the string that gets
indexed, and for tokenizing text for the lexical retriever.
(Named text.py rather than tokenize.py to avoid shadowing the stdlib module.)
"""
import re
from nltk.stem import PorterStemmer
from typing import Optional

_ALNUM = r"[^\W_]+"
_TOKEN_PATTERN = re.compile(rf"{_ALNUM}(?:(?:-|(?<=\d)\.(?=\d)){_ALNUM})*")
_PORTER_STEMMER: PorterStemmer = PorterStemmer() 

def doc_to_text(doc: dict[str, str]) -> str:
    """Combine title and body into the single string that gets indexed."""
    
    title = doc.get("title", '')
    text = doc.get("text", '')
    return title + ('. ' if title != '' else '')  + text


def tokenize(text: str, stem: bool = False) -> list[str]:
    """Split text into tokens for BM25."""

    words = _TOKEN_PATTERN.findall(text.lower())
    if stem:
        return [ _PORTER_STEMMER.stem(word) for word in words ]
    return words
