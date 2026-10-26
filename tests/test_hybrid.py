"""Tests for the hybrid retriever (the fusion strategies are tested in test_fusion.py)."""
from typing import Dict, List

import pytest

from rag_eval.config import ExperimentConfig
from rag_eval.retrievers.base import Ranking
from rag_eval.retrievers.hybrid import HybridRetriever
from rag_eval.retrievers.registry import build_retriever


def _ranking(*doc_ids: str) -> Ranking:
    """A ranking with arbitrary, decreasing scores."""
    return [(doc_id, 100.0 - i) for i, doc_id in enumerate(doc_ids)]


def _ids(ranking: Ranking) -> List[str]:
    return [doc_id for doc_id, _ in ranking]


# --- HybridRetriever --------------------------------------------------------

class _FakeRetriever:
    """Returns a preset ranking and records how it was called."""

    def __init__(self, name: str, doc_ids: List[str]):
        self.name = name
        self.doc_ids = doc_ids
        self.indexed_with: List[Dict] = []
        self.search_ks: List[int] = []

    def index(self, corpus) -> None:
        self.indexed_with.append(corpus)

    def search(self, query: str, k: int = 10) -> Ranking:
        self.search_ks.append(k)
        return _ranking(*self.doc_ids[:k])


def _hybrid(**kwargs) -> HybridRetriever:
    retrievers = [_FakeRetriever("first", ["A", "B", "C"]), _FakeRetriever("second", ["B", "C", "D"])]
    return HybridRetriever(retrievers=retrievers, **kwargs)


def test_index_indexes_every_sub_retriever_on_the_same_corpus():
    first, second = _FakeRetriever("first", ["A"]), _FakeRetriever("second", ["B"])
    corpus = {"A": {"title": "", "text": ""}}

    HybridRetriever(retrievers=[first, second]).index(corpus)

    assert first.indexed_with == [corpus] and second.indexed_with == [corpus]


def test_search_fuses_the_rankings_of_the_sub_retrievers():
    hybrid = _hybrid()
    hybrid.index({})

    assert _ids(hybrid.search("anything", k=4)) == ["B", "C", "A", "D"]


def test_search_asks_each_retriever_for_the_candidate_depth_not_just_k():
    first, second = _FakeRetriever("first", ["A"]), _FakeRetriever("second", ["B"])
    hybrid = HybridRetriever(retrievers=[first, second], candidates_per_retriever=50)
    hybrid.index({})

    hybrid.search("q", k=10)
    hybrid.search("q", k=80)  # more than the candidate depth: k wins

    assert first.search_ks == [50, 80] and second.search_ks == [50, 80]


def test_search_trims_to_k():
    hybrid = _hybrid()
    hybrid.index({})

    assert len(hybrid.search("q", k=2)) == 2


def test_search_before_index_raises():
    with pytest.raises(RuntimeError, match="index"):
        _hybrid().search("q")


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k_without_searching(k):
    first = _FakeRetriever("first", ["A"])
    hybrid = HybridRetriever(retrievers=[first, _FakeRetriever("second", ["B"])])
    hybrid.index({})

    with pytest.raises(ValueError, match="k"):
        hybrid.search("q", k=k)

    assert first.search_ks == []


# --- construction and registry ----------------------------------------------

def test_exactly_one_of_components_or_retrievers_is_required():
    with pytest.raises(ValueError, match="exactly one"):
        HybridRetriever()
    with pytest.raises(ValueError, match="exactly one"):
        HybridRetriever(components=[{"name": "bm25"}, {"name": "bm25"}], retrievers=[object(), object()])


def test_at_least_two_retrievers_are_required():
    with pytest.raises(ValueError, match="at least two"):
        HybridRetriever(components=[{"name": "bm25"}])


@pytest.mark.parametrize("kwargs", [{"fusion_params": {"rrf_k": 0}}, {"candidates_per_retriever": 0}])
def test_non_positive_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _hybrid(**kwargs)


def test_components_are_built_from_their_registered_names():
    hybrid = HybridRetriever(components=[{"name": "bm25"}, {"name": "bm25", "params": {"stem": True}}])

    assert [type(r).__name__ for r in hybrid._retrievers] == ["BM25Retriever", "BM25Retriever"]
    assert [r.stem for r in hybrid._retrievers] == [False, True]


def test_an_unknown_component_name_or_bad_component_params_fail_at_construction():
    with pytest.raises(ValueError, match="nope"):
        HybridRetriever(components=[{"name": "bm25"}, {"name": "nope"}])
    with pytest.raises(ValueError, match="typo"):
        HybridRetriever(components=[{"name": "bm25"}, {"name": "bm25", "params": {"typo": 1}}])


def test_end_to_end_on_a_tiny_corpus_with_two_real_bm25_retrievers():
    corpus = {
        "d1": {"title": "Tumor suppressor", "text": "The p53 protein suppresses tumor growth."},
        "d2": {"title": "Heart disease", "text": "Myocardial infarction damages heart muscle."},
        "d3": {"title": "Diet", "text": "Vitamin D intake and bone density."},
    }
    hybrid = build_retriever("hybrid", {"components": [{"name": "bm25"}, {"name": "bm25", "params": {"stem": True}}]})
    hybrid.index(corpus)

    assert _ids(hybrid.search("heart muscle", k=3))[0] == "d2"


def test_hybrid_is_registered_and_accepted_in_a_config():
    config = ExperimentConfig.model_validate({
        "experiment": "e",
        "retriever": {"name": "hybrid", "params": {"components": [{"name": "bm25"}, {"name": "dense"}]}},
        "mlflow": {"examples": {"n": 0}},
    })

    assert config.retriever.name == "hybrid"


# --- choosing the fusion ----------------------------------------------------

def test_the_default_fusion_is_rrf():
    hybrid = _hybrid()

    assert hybrid.fusion_name == "rrf" and hybrid.fusion_params == {}


def test_the_fusion_is_looked_up_by_name_and_gets_its_params(monkeypatch):
    calls = []

    def recording_fusion(rankings, k=10, scale=1):
        calls.append((len(rankings), k, scale))
        return rankings[0][:k]

    monkeypatch.setattr("rag_eval.retrievers.fusion.FUSION_REGISTRY", {"recording": recording_fusion})
    hybrid = _hybrid(fusion="recording", fusion_params={"scale": 7})
    hybrid.index({})

    result = hybrid.search("q", k=2)

    assert result == _ranking("A", "B")  # the registered function decided the result
    assert calls[-1] == (2, 2, 7)


def test_an_unknown_fusion_fails_when_the_retriever_is_built():
    with pytest.raises(ValueError, match="nope"):
        _hybrid(fusion="nope")


@pytest.mark.parametrize("fusion_params", [{"typo": 1}, {"k": 5}, {"rankings": []}])
def test_unknown_or_clashing_fusion_params_fail_when_the_retriever_is_built(fusion_params):
    with pytest.raises(ValueError, match="Invalid parameters for fusion 'rrf'"):
        _hybrid(fusion_params=fusion_params)


def test_invalid_fusion_param_values_fail_when_the_retriever_is_built():
    with pytest.raises(ValueError, match="rrf_k"):
        _hybrid(fusion_params={"rrf_k": -5})


def test_fusion_settings_given_in_a_config_style_dict_are_applied():
    hybrid = build_retriever("hybrid", {
        "components": [{"name": "bm25"}, {"name": "bm25"}],
        "fusion": "rrf",
        "fusion_params": {"rrf_k": 10},
    })

    assert hybrid.fusion_params == {"rrf_k": 10}
