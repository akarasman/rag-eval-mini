"""Tests for the fusion strategies and their registry."""
from typing import List

import pytest

from rag_eval.retrievers.base import Ranking
from rag_eval.retrievers.fusion import (
    FUSION_REGISTRY,
    available_fusions,
    get_fusion,
    reciprocal_rank_fusion,
    register_fusion,
    weighted_score_fusion,
)


def _ranking(*doc_ids: str) -> Ranking:
    """A ranking with arbitrary, decreasing scores (rank-based fusions must not look at them)."""
    return [(doc_id, 100.0 - i) for i, doc_id in enumerate(doc_ids)]


def _ids(ranking: Ranking) -> List[str]:
    return [doc_id for doc_id, _ in ranking]


# --- registry ---------------------------------------------------------------

def test_the_strategies_are_registered_under_their_names():
    assert get_fusion("rrf") is reciprocal_rank_fusion
    assert get_fusion("weighted") is weighted_score_fusion
    assert {"rrf", "weighted"} <= set(available_fusions())


def test_get_fusion_rejects_an_unknown_name_and_lists_the_available_ones():
    with pytest.raises(ValueError, match="rrf"):
        get_fusion("nope")


def test_register_fusion_adds_the_function_unchanged(monkeypatch):
    monkeypatch.setattr("rag_eval.retrievers.fusion.FUSION_REGISTRY", dict(FUSION_REGISTRY))

    @register_fusion("first-list-only")
    def first_list_only(rankings, k=10):
        return rankings[0][:k]

    assert get_fusion("first-list-only") is first_list_only
    assert first_list_only([_ranking("A", "B")], 1) == _ranking("A")[:1]


def test_register_fusion_rejects_a_duplicate_name():
    with pytest.raises(ValueError, match="already registered"):
        register_fusion("rrf")(lambda rankings, k=10: [])


# --- contract: every registered fusion must pass these ----------------------
# Settings each strategy needs to run (rrf has defaults); add an entry when a strategy needs params.

FUSION_PARAMS = {"rrf": {}, "weighted": {}}


@pytest.fixture(params=sorted(FUSION_REGISTRY))
def fuse(request):
    fusion_fn, params = get_fusion(request.param), FUSION_PARAMS[request.param]
    return lambda rankings, k=10: fusion_fn(rankings, k=k, **params)


def test_every_fusion_returns_unique_documents_best_first_trimmed_to_k(fuse):
    fused = fuse([_ranking("A", "B", "C", "D"), _ranking("C", "E", "A", "F")], k=4)

    assert len(fused) == 4
    assert len(set(_ids(fused))) == 4
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_every_fusion_includes_documents_found_by_only_one_retriever(fuse):
    assert set(_ids(fuse([_ranking("A"), _ranking("B")]))) == {"A", "B"}


def test_every_fusion_is_deterministic(fuse):
    rankings = [_ranking("A", "B", "C", "D"), _ranking("D", "C", "B", "A")]

    assert fuse(rankings) == fuse(rankings)


def test_every_fusion_handles_empty_rankings_and_still_validates_k(fuse):
    assert fuse([[], []], k=1) == []
    with pytest.raises(ValueError):
        fuse([[], []], k=0)


# --- reciprocal_rank_fusion -------------------------------------------------

def test_rrf_gives_each_document_one_over_rrf_k_plus_rank_from_every_list_it_is_in():
    fused = dict(reciprocal_rank_fusion([_ranking("A", "B", "C"), _ranking("B", "C", "D")], k=10, rrf_k=60))

    assert fused["A"] == pytest.approx(1 / 61)
    assert fused["B"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["C"] == pytest.approx(1 / 63 + 1 / 62)
    assert fused["D"] == pytest.approx(1 / 63)


def test_rrf_orders_by_total_points_best_first():
    fused = reciprocal_rank_fusion([_ranking("A", "B", "C"), _ranking("B", "C", "D")])

    assert _ids(fused) == ["B", "C", "A", "D"]
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_a_document_ranked_well_by_both_beats_one_ranked_first_by_only_one():
    fused = reciprocal_rank_fusion([_ranking("X", "S"), _ranking("Y", "S")])

    assert _ids(fused)[0] == "S"  # rank 2 in both lists: 2/62 > 1/61


def test_a_document_found_by_only_one_retriever_is_still_included():
    fused = reciprocal_rank_fusion([_ranking("A"), _ranking("B")])

    assert set(_ids(fused)) == {"A", "B"}


def test_the_original_scores_are_ignored():
    low = [("A", 0.001), ("B", 0.0009)]
    huge = [("A", 1e9), ("B", 1.0)]

    assert reciprocal_rank_fusion([low, low]) == reciprocal_rank_fusion([huge, huge])


def test_equal_scores_keep_the_order_of_first_appearance():
    assert _ids(reciprocal_rank_fusion([_ranking("A"), _ranking("B")])) == ["A", "B"]
    assert _ids(reciprocal_rank_fusion([_ranking("B"), _ranking("A")])) == ["B", "A"]


def test_the_result_is_trimmed_to_k():
    fused = reciprocal_rank_fusion([_ranking("A", "B", "C"), _ranking("D", "E", "F")], k=2)

    assert len(fused) == 2


def test_fewer_documents_than_k_returns_them_all():
    assert len(reciprocal_rank_fusion([_ranking("A", "B")], k=10)) == 2


def test_a_smaller_rrf_k_rewards_top_ranks_more():
    # P is first in one list only; Q is fourth in both. With rrf_k=60 agreement wins (2/64 > 1/61);
    # with rrf_k=1 the top rank wins (1/2 > 2/5).
    rankings = [_ranking("P", "a", "b", "Q"), _ranking("c", "d", "e", "Q")]

    assert _ids(reciprocal_rank_fusion(rankings, rrf_k=60))[0] == "Q"
    assert _ids(reciprocal_rank_fusion(rankings, rrf_k=1))[0] == "P"


def test_only_the_first_occurrence_of_a_document_within_a_ranking_counts():
    once = reciprocal_rank_fusion([_ranking("A", "B")])
    repeated = reciprocal_rank_fusion([[("A", 3.0), ("B", 2.0), ("A", 1.0)]])

    assert dict(repeated)["A"] == pytest.approx(dict(once)["A"])


def test_empty_rankings_give_an_empty_result():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_is_deterministic():
    rankings = [_ranking("A", "B", "C", "D"), _ranking("D", "C", "B", "A")]

    assert reciprocal_rank_fusion(rankings) == reciprocal_rank_fusion(rankings)


@pytest.mark.parametrize("kwargs, message", [
    ({"rankings": []}, "at least one"),
    ({"rankings": [_ranking("A")], "k": 0}, "k must"),
    ({"rankings": [_ranking("A")], "rrf_k": 0}, "rrf_k"),
])
def test_rrf_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion(**kwargs)


# --- weighted score fusion --------------------------------------------------

R1 = [("A", 10.0), ("B", 5.0), ("C", 0.0)]  # normalized: A 1.0, B 0.5, C 0.0
R2 = [("B", 0.9), ("C", 0.6), ("D", 0.3)]  # normalized: B 1.0, C 0.5, D 0.0


def _scores(ranking: Ranking) -> dict:
    return dict(ranking)


def test_weighted_fusion_adds_weighted_min_max_normalized_scores():
    fused = _scores(weighted_score_fusion([R1, R2], weights=[0.7, 0.3]))

    assert fused["A"] == pytest.approx(0.7 * 1.0)  # only in the first list
    assert fused["B"] == pytest.approx(0.7 * 0.5 + 0.3 * 1.0)
    assert fused["C"] == pytest.approx(0.7 * 0.0 + 0.3 * 0.5)
    assert fused["D"] == pytest.approx(0.0)  # the worst document of its only list


def test_the_weights_decide_who_wins():
    assert _ids(weighted_score_fusion([R1, R2], weights=[0.7, 0.3])) == ["A", "B", "C", "D"]
    assert _ids(weighted_score_fusion([R1, R2], weights=[0.3, 0.7])) == ["B", "C", "A", "D"]


def test_no_weights_means_equal_weights():
    assert weighted_score_fusion([R1, R2]) == weighted_score_fusion([R1, R2], weights=[1, 1])
    assert _ids(weighted_score_fusion([R1, R2])) == ["B", "A", "C", "D"]


def test_weights_are_rescaled_to_sum_to_one():
    scaled = _scores(weighted_score_fusion([R1, R2], weights=[7, 3]))
    unit = _scores(weighted_score_fusion([R1, R2], weights=[0.7, 0.3]))

    assert scaled == pytest.approx(unit)


def test_scores_stay_between_zero_and_one():
    scores = [score for _, score in weighted_score_fusion([R1, R2], weights=[3, 9])]

    assert all(0.0 <= score <= 1.0 + 1e-12 for score in scores)


def test_rescaling_one_retrievers_scores_does_not_change_the_result():
    # The point of min-max: BM25 scores times ten and shifted, or any other scale, fuse the same way.
    rescaled = [(doc_id, score * 10 + 3) for doc_id, score in R1]

    fused = weighted_score_fusion([rescaled, R2], weights=[0.4, 0.6])
    original = weighted_score_fusion([R1, R2], weights=[0.4, 0.6])

    assert _ids(fused) == _ids(original)
    assert _scores(fused) == pytest.approx(_scores(original))


def test_unlike_rrf_the_size_of_the_score_gaps_matters():
    # Both lists put A above B in `close` and `far`, only the gap between A and B differs, so a
    # rank-based fusion cannot tell the two cases apart. The weighted one can.
    other = [("B", 0.9), ("A", 0.8), ("Z", 0.0)]
    close = [("A", 1.00), ("B", 0.99), ("C", 0.0)]  # A and B almost tied
    far = [("A", 1.00), ("B", 0.10), ("C", 0.0)]  # B far behind A

    assert _ids(reciprocal_rank_fusion([close, other])) == _ids(reciprocal_rank_fusion([far, other]))
    assert _ids(weighted_score_fusion([close, other]))[0] == "B"
    assert _ids(weighted_score_fusion([far, other]))[0] == "A"


def test_a_zero_weight_removes_a_retrievers_influence_on_the_order():
    fused = weighted_score_fusion([R1, R2], weights=[1, 0])

    assert _ids(fused)[:3] == ["A", "B", "C"]  # the first list's order, untouched by the second


def test_a_list_with_all_equal_scores_gives_every_document_the_full_normalized_score():
    fused = _scores(weighted_score_fusion([[("A", 3.0), ("B", 3.0)], [("C", 7.0)]], weights=[0.5, 0.5]))

    assert fused == pytest.approx({"A": 0.5, "B": 0.5, "C": 0.5})


def test_weighted_fusion_ties_keep_the_order_of_first_appearance():
    fused = weighted_score_fusion([[("B", 1.0), ("A", 1.0)]], weights=[1])

    assert _ids(fused) == ["B", "A"]


def test_only_the_first_occurrence_of_a_document_counts_in_weighted_fusion():
    once = weighted_score_fusion([[("A", 2.0), ("B", 1.0)]], weights=[1])
    repeated = weighted_score_fusion([[("A", 2.0), ("B", 1.0), ("A", 0.0)]], weights=[1])

    assert repeated == once


@pytest.mark.parametrize("weights, message", [
    ([1.0], "weights for 2 rankings"),
    ([1.0, 1.0, 1.0], "weights for 2 rankings"),
    ([1.0, -0.5], "non-negative"),
    ([1.0, float("nan")], "finite"),
    ([1.0, float("inf")], "finite"),
    ([0.0, 0.0], "all be zero"),
])
def test_invalid_weights_are_rejected_even_for_empty_rankings(weights, message):
    with pytest.raises(ValueError, match=message):
        weighted_score_fusion([[], []], k=1, weights=weights)


@pytest.mark.parametrize("kwargs, message", [
    ({"rankings": []}, "at least one"),
    ({"rankings": [R1], "k": 0}, "k must"),
])
def test_weighted_fusion_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        weighted_score_fusion(**kwargs)
