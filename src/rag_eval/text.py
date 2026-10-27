"""Text preparation shared by the retrievers.

Responsible for turning a document dict (title + text) into the string that gets
indexed, and for tokenizing text for the lexical retriever.
(Named text.py rather than tokenize.py to avoid shadowing the stdlib module.)
"""
import re
from nltk.stem import PorterStemmer
from typing import Dict, List

_ALNUM = r"[^\W_]+"
_TOKEN_PATTERN = re.compile(rf"{_ALNUM}(?:(?:-|(?<=\d)\.(?=\d)){_ALNUM})*")
_PORTER_STEMMER: PorterStemmer = PorterStemmer() 

def doc_to_text(doc: Dict[str, str]) -> str:
    """Combine title and body into the single string that gets indexed.

    Args:
        doc: a corpus entry with "title" and "text"; a missing field counts as "".
    """
    
    title = doc.get("title", '')
    text = doc.get("text", '')
    return title + ('. ' if title != '' else '')  + text


def tokenize(text: str, stem: bool = False) -> List[str]:
    """Split text into tokens for BM25.

    Args:
        text: the text to split; it is lowercased first.
        stem: reduce each token to its Porter stem. Documents and queries must use the same setting.
    """

    words = _TOKEN_PATTERN.findall(text.lower())
    if stem:
        return [ _PORTER_STEMMER.stem(word) for word in words ]
    return words
