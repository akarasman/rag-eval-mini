"""Tests for the ask config. Loading a config never builds an LLM, reads an API key or contacts a server."""
from pathlib import Path

import pytest
import yaml

from rag_eval.ask_config import AskConfig, GenerationSettings, load_ask_config

EXAMPLE_CONFIGS = sorted((Path(__file__).parent.parent / "configs" / "ask").glob("*.yaml"))

MINIMAL = {
    "retriever": {"name": "bm25"},
    "llm": {"name": "ollama", "params": {"model": "llama3.2"}},
}


def _write(tmp_path, data, name="ask.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _experiment_file(tmp_path, retriever=None, name="experiment.yaml") -> Path:
    return _write(tmp_path, {
        "experiment": "scifact-baselines",
        "retriever": retriever or {"name": "bm25", "params": {"k1": 1.2}},
    }, name)


# --- loading ----------------------------------------------------------------------------

def test_a_minimal_config_loads_with_defaults(tmp_path):
    config = load_ask_config(_write(tmp_path, MINIMAL))

    assert config.dataset.name == "scifact"
    assert (config.retriever.name, config.llm.name) == ("bm25", "ollama")
    assert config.generation == GenerationSettings()
    assert (config.generation.k, config.generation.max_attempts) == (5, 2)
    assert config.generation.on_invalid_citation == "report"
    assert config.generation.max_passage_chars is None and config.generation.min_score is None


def test_every_setting_can_be_given(tmp_path):
    data = {
        **MINIMAL,
        "dataset": {"name": "scifact", "data_dir": "elsewhere"},
        "generation": {"k": 3, "max_attempts": 1, "on_invalid_citation": "ignore",
                       "max_passage_chars": 800, "min_score": 1.5},
    }

    config = load_ask_config(_write(tmp_path, data))

    assert config.dataset.data_dir == Path("elsewhere")
    assert config.generation == GenerationSettings(
        k=3, max_attempts=1, on_invalid_citation="ignore", max_passage_chars=800, min_score=1.5
    )


@pytest.mark.parametrize("path", EXAMPLE_CONFIGS, ids=[path.stem for path in EXAMPLE_CONFIGS])
def test_every_shipped_example_config_is_valid(path):
    config = load_ask_config(path)

    assert config.resolved_retriever.name  # the retriever resolves too


def test_there_is_an_example_for_every_backend():
    assert {path.stem for path in EXAMPLE_CONFIGS} >= {"ollama", "transformers", "openai_compatible"}


def test_a_config_that_is_not_a_mapping_is_rejected(tmp_path):
    path = tmp_path / "ask.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_ask_config(path)


def test_loading_does_not_build_the_llm_or_need_its_key(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_UNSET_KEY", raising=False)
    data = {**MINIMAL, "llm": {"name": "openai_compatible", "params": {
        "model": "m", "base_url": "https://x.example/v1", "api_key_env": "SOME_UNSET_KEY"}}}

    assert load_ask_config(_write(tmp_path, data)).llm.name == "openai_compatible"


# --- strictness -------------------------------------------------------------------------

def test_an_unknown_llm_name_lists_the_available_ones(tmp_path):
    data = {**MINIMAL, "llm": {"name": "gpt-nine"}}

    with pytest.raises(ValueError, match=r"gpt-nine.*ollama.*openai_compatible.*transformers"):
        load_ask_config(_write(tmp_path, data))


def test_an_unknown_retriever_name_is_rejected(tmp_path):
    data = {**MINIMAL, "retriever": {"name": "nope"}}

    with pytest.raises(ValueError, match="nope"):
        load_ask_config(_write(tmp_path, data))


@pytest.mark.parametrize("data", [
    {**MINIMAL, "typo": 1},
    {**MINIMAL, "generation": {"kk": 3}},
    {**MINIMAL, "llm": {"name": "ollama", "paramz": {}}},
    {**MINIMAL, "dataset": {"nmae": "scifact"}},
])
def test_unknown_keys_are_rejected_at_every_level(tmp_path, data):
    with pytest.raises(ValueError, match="Extra inputs"):
        load_ask_config(_write(tmp_path, data))


def test_the_llm_section_is_required(tmp_path):
    with pytest.raises(ValueError, match="llm"):
        load_ask_config(_write(tmp_path, {"retriever": {"name": "bm25"}}))


@pytest.mark.parametrize("generation, field", [
    ({"k": 0}, "k"),
    ({"k": -3}, "k"),
    ({"max_attempts": 0}, "max_attempts"),
    ({"max_passage_chars": 0}, "max_passage_chars"),
    ({"max_passage_chars": -1}, "max_passage_chars"),
    ({"on_invalid_citation": "raise"}, "on_invalid_citation"),
    ({"min_score": "high"}, "min_score"),
])
def test_generation_settings_are_range_checked(tmp_path, generation, field):
    with pytest.raises(ValueError, match=field):
        load_ask_config(_write(tmp_path, {**MINIMAL, "generation": generation}))


def test_a_config_is_frozen(tmp_path):
    config = load_ask_config(_write(tmp_path, MINIMAL))

    with pytest.raises(ValueError):
        config.generation = GenerationSettings(k=9)


# --- where the retriever comes from ------------------------------------------------------

def test_exactly_one_of_retriever_and_retriever_from_is_required(tmp_path):
    experiment = _experiment_file(tmp_path)

    with pytest.raises(ValueError, match="exactly one"):
        load_ask_config(_write(tmp_path, {"llm": MINIMAL["llm"]}))  # neither
    with pytest.raises(ValueError, match="exactly one"):
        load_ask_config(_write(tmp_path, {**MINIMAL, "retriever_from": str(experiment)}))  # both


def test_retriever_from_reuses_the_retriever_of_an_experiment_config(tmp_path):
    experiment = _experiment_file(tmp_path, {"name": "bm25", "params": {"k1": 1.2, "stem": True}})
    data = {"retriever_from": str(experiment), "llm": MINIMAL["llm"]}

    config = load_ask_config(_write(tmp_path, data))

    assert config.retriever is None
    assert config.resolved_retriever.name == "bm25"
    assert config.resolved_retriever.params == {"k1": 1.2, "stem": True}


def test_a_hybrid_retriever_can_be_reused_from_an_experiment_config(tmp_path):
    hybrid = {"name": "hybrid", "params": {"components": [{"name": "bm25"}, {"name": "bm25", "params": {"stem": True}}]}}
    data = {"retriever_from": str(_experiment_file(tmp_path, hybrid)), "llm": MINIMAL["llm"]}

    assert load_ask_config(_write(tmp_path, data)).resolved_retriever.name == "hybrid"


EXPERIMENT_CONFIGS = sorted((Path(__file__).parent.parent / "experiments" / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", EXPERIMENT_CONFIGS, ids=[path.stem for path in EXPERIMENT_CONFIGS])
def test_every_experiment_config_of_the_repo_can_be_reused_as_retriever_from(path):
    config = AskConfig.model_validate({"retriever_from": str(path), "llm": MINIMAL["llm"]})

    assert config.resolved_retriever.name in {"bm25", "dense", "hybrid"}


def test_the_inline_retriever_is_what_resolved_retriever_returns(tmp_path):
    config = load_ask_config(_write(tmp_path, {**MINIMAL, "retriever": {"name": "bm25", "params": {"b": 0.5}}}))

    assert config.resolved_retriever is config.retriever


def test_a_relative_retriever_from_is_resolved_against_the_working_directory(tmp_path, monkeypatch):
    _experiment_file(tmp_path, name="experiment.yaml")
    data = {"retriever_from": "experiment.yaml", "llm": MINIMAL["llm"]}
    ask_file = _write(tmp_path, data)
    monkeypatch.chdir(tmp_path)

    assert load_ask_config(ask_file).resolved_retriever.name == "bm25"


def test_a_missing_retriever_from_file_fails_when_the_config_is_loaded(tmp_path):
    data = {"retriever_from": str(tmp_path / "nothing.yaml"), "llm": MINIMAL["llm"]}

    with pytest.raises(ValueError, match="retriever_from.*cannot read"):
        load_ask_config(_write(tmp_path, data))


def test_an_invalid_experiment_file_fails_when_the_config_is_loaded(tmp_path):
    broken = _write(tmp_path, {"experiment": "x", "retriever": {"name": "nope"}}, "broken.yaml")
    data = {"retriever_from": str(broken), "llm": MINIMAL["llm"]}

    with pytest.raises(ValueError, match="retriever_from.*not a valid experiment config"):
        load_ask_config(_write(tmp_path, data))


def test_the_ask_config_class_can_be_built_directly():
    config = AskConfig.model_validate(MINIMAL)

    assert config.resolved_retriever.name == "bm25"
