"""Tests for text.py."""
import copy
from typing import List

import pytest
import rag_eval.text as text_module
from rag_eval.text import doc_to_text, tokenize


# --- doc_to_text ------------------------------------------------------------
# Pure function: assert the exact output string, not just "it didn't crash".

def test_joins_title_and_text_with_period_separator():
    assert doc_to_text({"title": "T", "text": "body"}) == "T. body"


@pytest.mark.parametrize(
    "doc, expected",
    [
        pytest.param({"title": "", "text": "body"}, "body", id="empty-title"),
        pytest.param({"text": "body"}, "body", id="missing-title"),
        pytest.param({"title": "T", "text": ""}, "T. ", id="empty-text"),
        pytest.param({"title": "T"}, "T. ", id="missing-text"),
        pytest.param({}, "", id="empty-doc"),
    ],
)
def test_doc_to_text_handles_missing_or_empty_fields(doc, expected):
    assert doc_to_text(doc) == expected


def test_doc_to_text_does_not_mutate_its_input():
    doc = {"title": "T", "text": "body"}
    snapshot = copy.deepcopy(doc)

    doc_to_text(doc)

    assert doc == snapshot


# --- tokenize ---------------------------------------------------------------

@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("Cancer", ["cancer"], id="lowercases"),
        pytest.param("cells, tissues.", ["cells", "tissues"], id="drops-punctuation"),
        pytest.param("(p53)", ["p53"], id="keeps-identifiers"),
        pytest.param("IL-6 well-known", ["il-6", "well-known"], id="keeps-hyphenated-compounds"),
        pytest.param("3.5", ["3.5"], id="keeps-decimals"),
        pytest.param("mg/kg", ["mg", "kg"], id="slash-splits"),
        pytest.param("a a b", ["a", "a", "b"], id="keeps-repeats"),
        pytest.param("snake_case", ["snake", "case"], id="underscore-splits"),
        pytest.param("", [], id="empty-string"),
        pytest.param("  ,. ", [], id="only-punctuation"),
    ],
)
def test_tokenize_rules(text, expected):
    assert tokenize(text) == expected


def test_tokenize_stem_collapses_inflections():
    assert tokenize("mutation mutations", stem=True) == ["mutat", "mutat"]


class _FakeStemmer:
    """Records every word it is asked to stem."""

    def __init__(self):
        self.stemmed_words: List[str] = []

    def stem(self, word: str) -> str:
        self.stemmed_words.append(word)
        return word


def test_tokenize_does_not_touch_the_stemmer_when_stem_is_false(monkeypatch):
    stemmer = _FakeStemmer()
    monkeypatch.setattr(text_module, "_PORTER_STEMMER", stemmer)

    tokenize("Cool text to stem.", stem=False)

    assert stemmer.stemmed_words == []


def test_tokenize_applies_stemmer_to_every_token(monkeypatch):
    stemmer = _FakeStemmer()
    monkeypatch.setattr(text_module, "_PORTER_STEMMER", stemmer)

    tokenize("a b c d", stem=True)

    assert stemmer.stemmed_words == ["a", "b", "c", "d"]
