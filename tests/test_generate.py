"""Tests for generation: prompt building, reply parsing, citation resolution and the status logic.

A scripted fake LLM (conftest.py) stands in for the model, so nothing here needs a network or a model.
"""
import logging

import pytest
from pydantic import ValidationError

from rag_eval.generate import (
    SYSTEM_PROMPT,
    AnswerStatus,
    GenerationResult,
    answer_question,
    build_prompt,
    parse_response,
    resolve_citations,
)
from rag_eval.llm.base import LLMError

PASSAGES = [
    ("d1", "The p53 protein suppresses tumor growth."),
    ("d2", "Aspirin reduces the risk of a second heart attack."),
    ("d3", "Vitamin D supports bone density."),
]
QUESTION = "What does aspirin do?"


def _reply(answerable=True, answer="Aspirin lowers heart attack risk.", citations=(2,)):
    import json
    return json.dumps({"answerable": answerable, "answer": answer, "citations": list(citations)})


# --- build_prompt ----------------------------------------------------------------

def test_the_prompt_has_every_passage_with_its_one_based_label_and_the_question():
    prompt = build_prompt(QUESTION, PASSAGES)

    for label, (_, text) in enumerate(PASSAGES, start=1):
        assert f"[{label}] {text}" in prompt
    assert QUESTION in prompt
    assert prompt.index("[1]") < prompt.index("[2]") < prompt.index("[3]") < prompt.index(QUESTION)


def test_the_prompt_does_not_contain_the_doc_ids():
    assert "d1" not in build_prompt(QUESTION, PASSAGES)  # the model cites numbers, never ids


def test_long_passages_are_cut_with_an_ellipsis_and_short_ones_are_left_alone():
    passages = [("a", "x" * 100), ("b", "short")]

    prompt = build_prompt(QUESTION, passages, max_passage_chars=10)

    assert "[1] " + "x" * 10 + "…" in prompt
    assert "x" * 11 not in prompt
    assert "[2] short" in prompt and "short…" not in prompt


def test_no_limit_keeps_the_whole_passage():
    assert "x" * 5000 in build_prompt(QUESTION, [("a", "x" * 5000)], max_passage_chars=None)


@pytest.mark.parametrize("limit", [0, -5])
def test_a_non_positive_passage_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="max_passage_chars"):
        build_prompt(QUESTION, PASSAGES, max_passage_chars=limit)


def test_the_system_prompt_states_the_rules_the_parser_relies_on():
    for phrase in ["only", "not instructions", "answerable", "citations", "JSON"]:
        assert phrase in SYSTEM_PROMPT


# --- parse_response ---------------------------------------------------------------

def test_parses_clean_json():
    response = parse_response('{"answerable": true, "answer": "Yes.", "citations": [1, 3]}')

    assert (response.answerable, response.answer, response.citations) == (True, "Yes.", [1, 3])


@pytest.mark.parametrize("raw", [
    '```json\n{"answerable": true, "answer": "Yes.", "citations": [1]}\n```',
    '```\n{"answerable": true, "answer": "Yes.", "citations": [1]}\n```',
    '  ```JSON {"answerable": true, "answer": "Yes.", "citations": [1]}```  ',
])
def test_parses_json_in_a_code_fence(raw):
    assert parse_response(raw).citations == [1]


@pytest.mark.parametrize("raw", [
    'Sure! Here is the answer: {"answerable": true, "answer": "Yes.", "citations": [1]} Hope that helps.',
    'Here you go:\n```json\n{"answerable": true, "answer": "Yes.", "citations": [1]}\n```\nAnything else?',
])
def test_parses_json_surrounded_by_prose(raw):
    assert parse_response(raw).answer == "Yes."


def test_braces_and_quotes_inside_strings_do_not_confuse_the_object_search():
    raw = 'Answer: {"answerable": true, "answer": "Use {curly} braces and \\"quotes\\" and a } here.", "citations": [1]} done'

    assert parse_response(raw).answer == 'Use {curly} braces and "quotes" and a } here.'


def test_numeric_string_citations_are_accepted():
    assert parse_response('{"answerable": true, "answer": "a", "citations": ["1", 2]}').citations == [1, 2]


def test_extra_keys_are_ignored_and_missing_optional_keys_default():
    response = parse_response('{"answerable": false, "reasoning": "none of the passages mention it"}')

    assert (response.answerable, response.answer, response.citations) == (False, "", [])


def test_null_answer_and_null_citations_are_read_as_empty():
    response = parse_response('{"answerable": false, "answer": null, "citations": null}')

    assert (response.answer, response.citations) == ("", [])


@pytest.mark.parametrize("raw", [
    "",
    "I do not know.",
    "{answerable: true}",  # not valid JSON
    '{"answerable": true, "answer": "cut off',  # the reply ended early
    '{"answer": "x", "citations": [1]}',  # answerable is missing
    '{"answerable": "maybe", "answer": "x"}',
    '{"answerable": true, "answer": 5}',
    '{"answerable": true, "answer": "x", "citations": "1"}',
    '{"answerable": true, "answer": "x", "citations": [1, "two"]}',
    '{"answerable": true, "answer": "x", "citations": [true]}',
    '{"answerable": true, "answer": "x", "citations": [1.5]}',
    "[1, 2, 3]",
    '"just a string"',
])
def test_rejects_replies_that_are_not_the_expected_json(raw):
    with pytest.raises(ValueError):
        parse_response(raw)


def test_the_error_says_what_is_wrong_in_a_short_message():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_response("plain text")
    with pytest.raises(ValueError, match="answerable"):
        parse_response('{"answer": "x"}')


# --- resolve_citations ------------------------------------------------------------

def test_valid_labels_become_doc_ids_in_order_of_first_citation():
    assert resolve_citations([3, 1], PASSAGES) == (["d3", "d1"], [])


def test_duplicate_labels_are_cited_once():
    assert resolve_citations([2, 2, 1, 2], PASSAGES) == (["d2", "d1"], [])


@pytest.mark.parametrize("label", [0, -1, 4, 99])
def test_labels_outside_the_passage_range_are_invalid(label):
    assert resolve_citations([label], PASSAGES) == ([], [label])


def test_valid_and_invalid_labels_are_separated_and_invalid_ones_are_deduplicated_in_order():
    assert resolve_citations([9, 1, 0, 9, 2], PASSAGES) == (["d1", "d2"], [9, 0])


def test_no_labels_gives_nothing():
    assert resolve_citations([], PASSAGES) == ([], [])


def test_two_labels_for_one_document_give_one_doc_id():
    passages = [("same", "first text"), ("same", "second text")]

    assert resolve_citations([1, 2], passages) == (["same"], [])


# --- answer_question: every status --------------------------------------------------

def test_answered_with_the_cited_doc_ids(scripted_llm):
    llm = scripted_llm([_reply(citations=(2, 1))])

    result = answer_question(QUESTION, PASSAGES, llm)

    assert result.status is AnswerStatus.ANSWERED
    assert result.answer == "Aspirin lowers heart attack risk."
    assert result.citations == ["d2", "d1"]
    assert result.invalid_citations == []
    assert result.passages == ["d1", "d2", "d3"]
    assert result.attempts == 1
    assert result.raw_output == llm._replies[0]


def test_the_llm_gets_the_built_prompt_and_the_system_prompt(scripted_llm):
    llm = scripted_llm([_reply()])

    answer_question(QUESTION, PASSAGES, llm, max_passage_chars=20)

    assert llm.calls == [(build_prompt(QUESTION, PASSAGES, 20), SYSTEM_PROMPT)]


def test_abstained_when_the_model_says_the_passages_do_not_answer(scripted_llm):
    llm = scripted_llm([_reply(answerable=False, answer="", citations=())])

    result = answer_question(QUESTION, PASSAGES, llm)

    assert result.status is AnswerStatus.ABSTAINED
    assert (result.answer, result.citations, result.invalid_citations) == ("", [], [])


def test_abstained_ignores_the_citations_the_model_gave_anyway(scripted_llm):
    llm = scripted_llm([_reply(answerable=False, answer="", citations=(1, 99))])

    result = answer_question(QUESTION, PASSAGES, llm)

    assert result.status is AnswerStatus.ABSTAINED
    assert (result.citations, result.invalid_citations) == ([], [])


def test_abstained_keeps_the_text_the_model_wrote(scripted_llm):
    llm = scripted_llm([_reply(answerable=False, answer="Not covered.", citations=())])

    assert answer_question(QUESTION, PASSAGES, llm).answer == "Not covered."


def test_no_passages_never_calls_the_llm(scripted_llm):
    llm = scripted_llm([])

    result = answer_question(QUESTION, [], llm)

    assert result.status is AnswerStatus.NO_PASSAGES
    assert (result.answer, result.citations, result.passages, result.attempts) == ("", [], [], 0)
    assert result.raw_output is None
    assert llm.calls == []


def test_ungrounded_when_an_answer_has_no_citations(scripted_llm):
    llm = scripted_llm([_reply(citations=())])

    result = answer_question(QUESTION, PASSAGES, llm)

    assert result.status is AnswerStatus.UNGROUNDED
    assert result.answer == "Aspirin lowers heart attack risk."  # kept, so the user can see what was claimed
    assert result.citations == []


def test_ungrounded_when_every_citation_is_invalid(scripted_llm):
    llm = scripted_llm([_reply(citations=(7, 0))])

    result = answer_question(QUESTION, PASSAGES, llm)

    assert result.status is AnswerStatus.UNGROUNDED
    assert (result.citations, result.invalid_citations) == ([], [7, 0])


def test_ungrounded_when_the_answer_is_empty_even_with_a_valid_citation(scripted_llm):
    llm = scripted_llm([_reply(answer="   ", citations=(1,))])

    result = answer_question(QUESTION, PASSAGES, llm)

    assert result.status is AnswerStatus.UNGROUNDED
    assert result.citations == []


def test_invalid_output_after_max_attempts_calls_keeps_the_last_raw_text(scripted_llm):
    llm = scripted_llm(["not json", "still not json", "nope"])

    result = answer_question(QUESTION, PASSAGES, llm, max_attempts=3)

    assert result.status is AnswerStatus.INVALID_OUTPUT
    assert (result.answer, result.citations) == ("", [])
    assert result.attempts == 3 and len(llm.calls) == 3
    assert result.raw_output == "nope"
    assert result.passages == ["d1", "d2", "d3"]


def test_one_attempt_means_no_retry(scripted_llm):
    llm = scripted_llm(["not json"])

    result = answer_question(QUESTION, PASSAGES, llm, max_attempts=1)

    assert result.status is AnswerStatus.INVALID_OUTPUT
    assert len(llm.calls) == 1


def test_a_second_attempt_can_succeed_and_gets_a_correction_appended(scripted_llm):
    llm = scripted_llm(["Sorry, I cannot do JSON.", _reply()])

    result = answer_question(QUESTION, PASSAGES, llm)

    assert result.status is AnswerStatus.ANSWERED
    assert result.attempts == 2
    first_prompt, second_prompt = llm.calls[0][0], llm.calls[1][0]
    assert "previous reply" not in first_prompt
    assert second_prompt.startswith(first_prompt)
    assert "previous reply was not valid JSON" in second_prompt
    assert llm.calls[1][1] == SYSTEM_PROMPT


def test_retries_restate_the_prompt_instead_of_piling_up_corrections(scripted_llm):
    llm = scripted_llm(["bad", "bad", _reply()])

    answer_question(QUESTION, PASSAGES, llm, max_attempts=3)

    assert llm.calls[1][0] == llm.calls[2][0]
    assert llm.calls[2][0].count("previous reply") == 1


# --- invalid citations: report vs ignore ------------------------------------------

def test_a_mix_of_valid_and_invalid_citations_is_answered_and_reports_the_invalid_ones(scripted_llm, caplog):
    llm = scripted_llm([_reply(citations=(2, 9))])

    with caplog.at_level(logging.WARNING):
        result = answer_question(QUESTION, PASSAGES, llm, on_invalid_citation="report")

    assert result.status is AnswerStatus.ANSWERED
    assert result.citations == ["d2"]
    assert result.invalid_citations == [9]
    assert len([record for record in caplog.records if record.levelno == logging.WARNING]) == 1


def test_ignore_drops_invalid_citations_without_a_trace(scripted_llm, caplog):
    llm = scripted_llm([_reply(citations=(2, 9))])

    with caplog.at_level(logging.WARNING):
        result = answer_question(QUESTION, PASSAGES, llm, on_invalid_citation="ignore")

    assert result.status is AnswerStatus.ANSWERED
    assert result.citations == ["d2"]
    assert result.invalid_citations == []
    assert caplog.records == []


def test_ignore_still_downgrades_an_answer_whose_citations_are_all_invalid(scripted_llm):
    llm = scripted_llm([_reply(citations=(9,))])

    result = answer_question(QUESTION, PASSAGES, llm, on_invalid_citation="ignore")

    assert result.status is AnswerStatus.UNGROUNDED
    assert result.invalid_citations == []


def test_no_warning_is_logged_when_every_citation_is_valid(scripted_llm, caplog):
    llm = scripted_llm([_reply(citations=(1,))])

    with caplog.at_level(logging.WARNING):
        answer_question(QUESTION, PASSAGES, llm)

    assert caplog.records == []


# --- errors and edges ---------------------------------------------------------------

def test_a_backend_failure_propagates_and_is_not_retried(scripted_llm):
    llm = scripted_llm([LLMError("connection refused"), _reply()])

    with pytest.raises(LLMError, match="connection refused"):
        answer_question(QUESTION, PASSAGES, llm, max_attempts=3)

    assert len(llm.calls) == 1


@pytest.mark.parametrize("question", ["", "   \n"])
def test_a_blank_question_is_rejected_before_anything_runs(scripted_llm, question):
    llm = scripted_llm([])

    with pytest.raises(ValueError, match="question"):
        answer_question(question, PASSAGES, llm)

    assert llm.calls == []


def test_bad_settings_are_rejected_before_the_llm_is_called(scripted_llm):
    llm = scripted_llm([])

    with pytest.raises(ValueError, match="max_attempts"):
        answer_question(QUESTION, PASSAGES, llm, max_attempts=0)
    with pytest.raises(ValueError, match="on_invalid_citation"):
        answer_question(QUESTION, PASSAGES, llm, on_invalid_citation="raise")
    with pytest.raises(ValueError, match="max_passage_chars"):
        answer_question(QUESTION, PASSAGES, llm, max_passage_chars=0)

    assert llm.calls == []


def test_bad_settings_are_rejected_even_when_there_are_no_passages(scripted_llm):
    with pytest.raises(ValueError, match="max_attempts"):
        answer_question(QUESTION, [], scripted_llm([]), max_attempts=0)


def test_the_result_is_frozen(scripted_llm):
    result = answer_question(QUESTION, PASSAGES, scripted_llm([_reply()]))

    with pytest.raises(ValidationError):
        result.answer = "changed"


def test_the_result_serialises_to_plain_json_values(scripted_llm):
    result = answer_question(QUESTION, PASSAGES, scripted_llm([_reply()]))

    dumped = result.model_dump(mode="json")

    assert dumped["status"] == "answered"
    assert isinstance(result, GenerationResult)
