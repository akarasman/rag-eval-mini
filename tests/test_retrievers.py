"""Tests for retrievers. Covers the oracle and BM25.

TODO: dense and hybrid tests on a tiny hand-made corpus where the right top result is obvious.
"""
import logging

import pytest

from rag_eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from rag_eval.retrievers.bm25 import BM25Retriever
from rag_eval.retrievers.oracle import OracleRetriever
from rag_eval.retrievers.registry import build_retriever, register_retriever


# --- oracle -----------------------------------------------------------------

CORPUS = {str(i): {"title": "", "text": ""} for i in range(1, 7)}
QUERIES = {"q1": "first", "q2": "second"}
QRELS = {
    "q1": {"1": 1},
    "q2": {"3": 1, "2": 2, "9": 1},  # grade-1 listed before grade-2; "9" is not in the corpus
}


def _oracle(use_filler_docs=True, corpus=CORPUS) -> OracleRetriever:
    oracle = OracleRetriever(QUERIES, QRELS, use_filler_docs=use_filler_docs)
    oracle.index(corpus)
    return oracle


def _ids(ranking):
    return [doc_id for doc_id, _ in ranking]


def test_oracle_returns_relevant_docs_best_grade_first():
    assert _ids(_oracle(use_filler_docs=False).search("second", k=10)) == ["2", "3"]


def test_oracle_ignores_relevant_docs_missing_from_the_corpus():
    assert "9" not in _ids(_oracle().search("second", k=10))


def test_oracle_trims_to_k():
    assert _ids(_oracle(use_filler_docs=False).search("second", k=1)) == ["2"]
    assert _ids(_oracle().search("second", k=1)) == ["2"]


def test_oracle_without_filler_may_return_fewer_than_k():
    assert len(_oracle(use_filler_docs=False).search("first", k=5)) == 1


def test_oracle_filler_pads_to_k_without_repeating_relevant_docs():
    ids = _ids(_oracle().search("second", k=5))

    assert len(ids) == 5
    assert ids[:2] == ["2", "3"]
    assert len(set(ids)) == 5  # a duplicate here would let recall exceed 1


def test_oracle_filler_runs_out_gracefully_and_warns(caplog):
    small_corpus = {"1": {"title": "", "text": ""}, "2": {"title": "", "text": ""}}
    with caplog.at_level(logging.WARNING):
        ids = _ids(_oracle(corpus=small_corpus).search("first", k=10))

    assert sorted(ids) == ["1", "2"]
    assert "filler" in caplog.text


def test_oracle_scores_are_strictly_decreasing():
    scores = [score for _, score in _oracle().search("second", k=5)]

    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_oracle_search_is_repeatable_and_does_not_corrupt_its_index():
    oracle = _oracle()

    first = oracle.search("first", k=5)
    oracle.search("first", k=10)  # a longer search in between must not leak filler into the index
    assert oracle.search("first", k=5) == first


def test_oracle_raises_for_unseen_query_text():
    with pytest.raises(KeyError, match="never seen"):
        _oracle().search("something else")


def test_oracle_raises_if_searched_before_indexing():
    with pytest.raises(RuntimeError, match="index"):
        OracleRetriever(QUERIES, QRELS).search("first")


@pytest.mark.parametrize("k", [0, -1])
def test_oracle_rejects_non_positive_k(k):
    with pytest.raises(ValueError, match="k"):
        _oracle().search("first", k=k)


def test_oracle_rejects_duplicate_query_text_and_stays_unindexed():
    oracle = OracleRetriever({"q1": "same", "q2": "same"}, QRELS)

    with pytest.raises(ValueError, match="Duplicate"):
        oracle.index(CORPUS)
    with pytest.raises(RuntimeError):
        oracle.search("same")


@pytest.mark.parametrize("use_filler_docs", [True, False])
@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("query_id", ["q1", "q2"])
def test_oracle_scores_perfectly_with_the_real_metrics(query_id, k, use_filler_docs):
    oracle = _oracle(use_filler_docs=use_filler_docs)
    relevant = {doc_id: grade for doc_id, grade in QRELS[query_id].items() if doc_id in CORPUS}

    ids = _ids(oracle.search(QUERIES[query_id], k=k))

    assert reciprocal_rank(ids, relevant, k) == pytest.approx(1.0)
    assert ndcg_at_k(ids, relevant, k) == pytest.approx(1.0)
    # recall can only reach min(k, |relevant|) / |relevant|, for any retriever
    assert recall_at_k(ids, relevant, k) == pytest.approx(min(k, len(relevant)) / len(relevant))


# --- bm25 -------------------------------------------------------------------

BM25_CORPUS = {
    "d1": {"title": "Tumor suppressor", "text": "The p53 protein suppresses tumor growth."},
    "d2": {"title": "Heart disease", "text": "Myocardial infarction damages heart muscle."},
    "d3": {"title": "Diet", "text": "Vitamin D intake and bone density."},
    "d4": {"title": "Gene expression", "text": "Mutations in p53 alter gene expression."},
}


def _bm25(corpus=BM25_CORPUS, **kwargs) -> BM25Retriever:
    retriever = BM25Retriever(**kwargs)
    retriever.index(corpus)
    return retriever


def test_bm25_ranks_the_matching_document_first():
    assert _ids(_bm25().search("heart muscle", k=4))[0] == "d2"


def test_bm25_scores_are_descending_and_positive():
    ranking = _bm25().search("p53 gene", k=4)
    scores = [score for _, score in ranking]

    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_bm25_returns_only_documents_that_share_a_term_with_the_query():
    assert set(_ids(_bm25().search("p53", k=4))) == {"d1", "d4"}
    assert _ids(_bm25().search("vitamin", k=4)) == ["d3"]


def test_bm25_returns_at_most_the_corpus_size():
    assert len(_bm25().search("p53 heart vitamin gene", k=100)) == len(BM25_CORPUS)


def test_bm25_trims_to_k():
    assert len(_bm25().search("p53", k=1)) == 1


def test_bm25_equal_scores_keep_corpus_order():
    corpus = {key: {"title": "", "text": "heart"} for key in ("z", "a", "m")}

    assert _ids(_bm25(corpus).search("heart", k=3)) == ["z", "a", "m"]


@pytest.mark.parametrize("query", ["", "...", "zzzunknown"])
def test_bm25_query_without_known_terms_returns_nothing(query):
    assert _bm25().search(query, k=4) == []


def test_bm25_matching_is_case_insensitive():
    assert _bm25().search("HEART", k=4) == _bm25().search("heart", k=4)


def test_bm25_stemming_matches_inflections_only_when_enabled():
    corpus = {"a": {"title": "", "text": "mutations"}, "b": {"title": "", "text": "unrelated"}}

    assert _bm25(corpus).search("mutation", k=2) == []
    top_id, top_score = _bm25(corpus, stem=True).search("mutation", k=2)[0]
    assert top_id == "a"
    assert top_score > 0


def test_bm25_title_is_searchable():
    assert _ids(_bm25().search("diet", k=1)) == ["d3"]


def test_bm25_raises_if_searched_before_indexing():
    with pytest.raises(RuntimeError, match="index"):
        BM25Retriever().search("p53")


@pytest.mark.parametrize("k", [0, -1])
def test_bm25_rejects_non_positive_k(k):
    with pytest.raises(ValueError, match="k"):
        _bm25().search("p53", k=k)


def test_bm25_reindexing_replaces_the_previous_corpus():
    retriever = _bm25()
    retriever.index({"x": {"title": "", "text": "heart"}})

    assert _ids(retriever.search("heart", k=5)) == ["x"]


# --- registry ---------------------------------------------------------------

def test_bm25_is_registered_and_built_from_params():
    retriever = build_retriever("bm25", {"k1": 1.2, "stem": True})

    assert isinstance(retriever, BM25Retriever)
    assert (retriever.k1, retriever.b, retriever.stem) == (1.2, 0.75, True)


def test_build_retriever_rejects_an_unknown_name_and_lists_the_available_ones():
    with pytest.raises(ValueError, match="bm25"):
        build_retriever("nope", {})


def test_build_retriever_rejects_parameters_the_retriever_does_not_take():
    with pytest.raises(ValueError, match="bm25.*typo"):
        build_retriever("bm25", {"typo": 1})


def test_register_retriever_rejects_a_duplicate_name():
    with pytest.raises(ValueError, match="already registered"):
        register_retriever("bm25")(BM25Retriever)
