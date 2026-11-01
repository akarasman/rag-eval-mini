"""Tests for the ask CLI: real BM25 retrieval over a tiny corpus, and a scripted fake LLM (no network)."""
import json

import pytest
import yaml

from rag_eval import ask
from rag_eval.ask import format_result, main, retrieve_passages
from rag_eval.generate import AnswerStatus, GenerationResult
from rag_eval.llm.base import LLMError
from rag_eval.llm.registry import LLM_REGISTRY

CORPUS = {
    "d1": {"title": "Aspirin", "text": "Aspirin reduces the risk of a second heart attack."},
    "d2": {"title": "Vitamin D", "text": "Vitamin D supports bone density in older adults."},
    "d3": {"title": "p53", "text": "The p53 protein suppresses tumor growth."},
}
QUESTION = "aspirin heart attack"
WIDE_QUESTION = "aspirin bone protein"  # matches all three documents


def _reply(answerable=True, answer="Aspirin lowers the risk of another heart attack.", citations=(1,)):
    return json.dumps({"answerable": answerable, "answer": answer, "citations": list(citations)})


@pytest.fixture
def setup(tmp_path, monkeypatch, scripted_llm):
    """Writes the corpus and registers a scripted LLM as "scripted"; returns a function that runs main()."""
    data = tmp_path / "data" / "scifact"
    data.mkdir(parents=True)
    (data / "corpus.jsonl").write_text(
        "\n".join(json.dumps({"_id": doc_id, **doc}) for doc_id, doc in CORPUS.items()), encoding="utf-8"
    )

    def run(replies, question=QUESTION, generation=None, extra_args=(), config_overrides=None):
        llm = scripted_llm(replies)
        monkeypatch.setitem(LLM_REGISTRY, "scripted", lambda **params: llm)
        config = {
            "dataset": {"name": "scifact", "data_dir": str(tmp_path / "data")},
            "retriever": {"name": "bm25"},
            "llm": {"name": "scripted"},
        }
        if generation is not None:
            config["generation"] = generation
        config.update(config_overrides or {})
        config_path = tmp_path / "ask.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        main(["--config", str(config_path), *extra_args, question])
        return llm

    return run


# --- the statuses, end to end -------------------------------------------------------------

def test_an_answered_question_prints_the_answer_and_its_numbered_sources(setup, capsys):
    llm = setup([_reply(citations=(1,))])

    output = capsys.readouterr().out

    assert "Status: answered" in output
    assert "Aspirin lowers the risk of another heart attack." in output
    assert "Sources:" in output and "1. d1 - Aspirin" in output
    assert "[1] Aspirin. Aspirin reduces the risk of a second heart attack." in llm.calls[0][0]  # doc_to_text


def test_only_the_cited_documents_are_listed_as_sources(setup, capsys):
    setup([_reply(citations=(2,))], question=WIDE_QUESTION)

    output = capsys.readouterr().out
    sources = output.split("Sources:")[1]

    assert sources.count(". d") == 1  # one numbered source line


def test_an_abstention_says_the_passages_do_not_answer(setup, capsys):
    setup([_reply(answerable=False, answer="", citations=())])

    output = capsys.readouterr().out

    assert "Status: abstained" in output
    assert "The retrieved passages do not answer this question." in output
    assert "Sources:" not in output


def test_a_question_with_no_matching_documents_never_calls_the_llm(setup, capsys):
    llm = setup([], question="zzzz qqqq")

    output = capsys.readouterr().out

    assert "Status: no_passages" in output and "No passages were retrieved" in output
    assert llm.calls == []


def test_an_answer_without_a_valid_citation_is_marked_unverified(setup, capsys):
    setup([_reply(citations=())])

    output = capsys.readouterr().out

    assert "Status: ungrounded" in output
    assert "without a valid citation" in output
    assert "Unverified answer: Aspirin lowers the risk of another heart attack." in output
    assert "Sources:" not in output


def test_replies_that_are_never_valid_json_are_reported_with_the_attempt_count(setup, capsys):
    setup(["not json", "still not json"])

    output = capsys.readouterr().out

    assert "Status: invalid_output" in output
    assert "2 attempt(s)" in output and "--json" in output


def test_citations_to_passages_that_do_not_exist_are_reported(setup, capsys):
    setup([_reply(citations=(1, 9))])

    output = capsys.readouterr().out

    assert "Status: answered" in output
    assert "do not exist: [9]" in output


def test_ignoring_invalid_citations_hides_the_note(setup, capsys):
    setup([_reply(citations=(1, 9))], generation={"on_invalid_citation": "ignore"})

    assert "do not exist" not in capsys.readouterr().out


# --- options ---------------------------------------------------------------------------------

def test_json_output_is_the_whole_result_as_parseable_json(setup, capsys):
    setup([_reply(citations=(1,))], extra_args=["--json"])

    data = json.loads(capsys.readouterr().out)

    assert data["status"] == "answered"
    assert data["citations"] == ["d1"]
    assert data["passages"] == ["d1"]
    assert data["attempts"] == 1
    assert "Aspirin lowers" in data["answer"] and data["raw_output"]


def test_k_comes_from_the_config_and_the_command_line_overrides_it(setup):
    from_config = setup([_reply()], question=WIDE_QUESTION, generation={"k": 1})
    overridden = setup([_reply()], question=WIDE_QUESTION, generation={"k": 1}, extra_args=["--k", "3"])

    assert "[2]" not in from_config.calls[0][0]
    assert "[3]" in overridden.calls[0][0]


def test_the_default_k_is_five(setup):
    llm = setup([_reply()], question=WIDE_QUESTION)

    assert "[3]" in llm.calls[0][0]  # all three documents match, and 5 is more than 3


def test_min_score_can_empty_the_passage_list(setup, capsys):
    llm = setup([], generation={"min_score": 1e9})

    assert "Status: no_passages" in capsys.readouterr().out
    assert llm.calls == []


def test_max_passage_chars_reaches_the_prompt(setup):
    llm = setup([_reply()], generation={"max_passage_chars": 20})

    assert "[1] Aspirin. Aspirin red…" in llm.calls[0][0]
    assert "second heart attack" not in llm.calls[0][0]


def test_the_retriever_can_come_from_an_experiment_config(setup, tmp_path, capsys):
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump({"experiment": "x", "retriever": {"name": "bm25"}}), encoding="utf-8")

    setup([_reply()], config_overrides={"retriever": None, "retriever_from": str(experiment)})

    assert "Status: answered" in capsys.readouterr().out


# --- errors ----------------------------------------------------------------------------------

def test_a_bad_config_exits_with_status_1_and_the_message_on_stderr(setup, capsys):
    with pytest.raises(SystemExit) as exit_info:
        setup([], config_overrides={"llm": {"name": "gpt-nine"}})

    captured = capsys.readouterr()
    assert exit_info.value.code == 1
    assert captured.err.startswith("error:") and "gpt-nine" in captured.err
    assert captured.out == ""


def test_a_failing_backend_exits_with_its_message(setup, capsys):
    with pytest.raises(SystemExit) as exit_info:
        setup([LLMError("is the Ollama server running?")])

    assert exit_info.value.code == 1
    assert "is the Ollama server running?" in capsys.readouterr().err


def test_verbose_shows_the_traceback_instead_of_hiding_it(setup):
    with pytest.raises(LLMError, match="boom"):
        setup([LLMError("boom")], extra_args=["-v"])


def test_a_backend_that_cannot_be_built_stops_the_run_before_the_corpus_is_loaded(setup, monkeypatch, capsys):
    monkeypatch.delenv("NO_SUCH_API_KEY", raising=False)
    monkeypatch.setattr(ask, "download_dataset", lambda *a, **k: pytest.fail("the corpus must not be loaded"))
    llm_config = {"name": "openai_compatible", "params": {
        "model": "m", "base_url": "https://x.example/v1", "api_key_env": "NO_SUCH_API_KEY"}}

    with pytest.raises(SystemExit) as exit_info:
        setup([], config_overrides={"llm": llm_config})

    assert exit_info.value.code == 1
    assert "NO_SUCH_API_KEY" in capsys.readouterr().err


def test_a_blank_question_is_rejected_before_the_config_is_even_read(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--config", "does-not-exist.yaml", "   "])

    assert exit_info.value.code == 2  # argparse's usage-error status
    assert "question" in capsys.readouterr().err


def test_the_config_option_is_required(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["a question"])

    assert exit_info.value.code == 2


@pytest.mark.parametrize("k", ["0", "-2", "many"])
def test_k_must_be_a_positive_integer(k, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--config", "x.yaml", "--k", k, "a question"])

    assert exit_info.value.code == 2


# --- retrieve_passages ---------------------------------------------------------------------------

class _FixedRetriever:
    name = "fixed"

    def __init__(self, ranking):
        self.ranking = ranking
        self.asked = []

    def index(self, corpus):
        pass

    def search(self, query, k=10):
        self.asked.append((query, k))
        return self.ranking


def test_retrieve_passages_returns_doc_ids_with_their_indexed_text_in_rank_order():
    retriever = _FixedRetriever([("d2", 3.0), ("d1", 2.0)])

    passages = retrieve_passages(retriever, CORPUS, "q", k=4)

    assert passages == [
        ("d2", "Vitamin D. Vitamin D supports bone density in older adults."),
        ("d1", "Aspirin. Aspirin reduces the risk of a second heart attack."),
    ]
    assert retriever.asked == [("q", 4)]


def test_min_score_drops_weaker_passages_and_keeps_those_that_equal_it():
    retriever = _FixedRetriever([("d1", 3.0), ("d2", 2.0), ("d3", 1.0)])

    passages = retrieve_passages(retriever, CORPUS, "q", k=3, min_score=2.0)

    assert [doc_id for doc_id, _ in passages] == ["d1", "d2"]


def test_no_min_score_keeps_everything_even_a_zero_score():
    retriever = _FixedRetriever([("d1", 0.0), ("d2", -1.0)])

    assert len(retrieve_passages(retriever, CORPUS, "q", k=2)) == 2


def test_an_empty_ranking_gives_no_passages():
    assert retrieve_passages(_FixedRetriever([]), CORPUS, "q", k=3) == []


# --- format_result ---------------------------------------------------------------------------------

def _result(status, **fields):
    defaults = dict(answer="", citations=[], invalid_citations=[], passages=["d1"], attempts=1, raw_output="raw")
    defaults.update(fields)
    return GenerationResult(status=status, **defaults)


def test_sources_are_numbered_in_citation_order_with_titles():
    result = _result(AnswerStatus.ANSWERED, answer="A.", citations=["d3", "d1"])

    text = format_result(result, CORPUS)

    assert text.splitlines()[-2:] == ["  1. d3 - p53", "  2. d1 - Aspirin"]


def test_a_source_without_a_title_is_listed_by_id_only():
    corpus = {"d9": {"title": "", "text": "body"}}

    text = format_result(_result(AnswerStatus.ANSWERED, answer="A.", citations=["d9"]), corpus)

    assert text.splitlines()[-1] == "  1. d9"


def test_an_unverified_answer_without_text_prints_no_answer_line():
    text = format_result(_result(AnswerStatus.UNGROUNDED, answer=""), CORPUS)

    assert "without a valid citation" in text and "Unverified answer" not in text
