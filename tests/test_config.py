"""Tests for experiment configs: loading, defaults, and failing early on bad input."""
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml
from pydantic import ValidationError

from rag_eval.config import ExperimentConfig, load_config

EXAMPLE_CONFIG = Path(__file__).parent.parent / "experiments" / "configs" / "bm25_baseline.yaml"

MINIMAL: Dict[str, Any] = {"experiment": "exp", "retriever": {"name": "bm25"}}


def _config(**overrides) -> ExperimentConfig:
    return ExperimentConfig.model_validate({**MINIMAL, **overrides})


def _write_yaml(tmp_path, data) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


# --- defaults and loading ---------------------------------------------------

def test_minimal_config_gets_sensible_defaults():
    config = _config()

    assert config.dataset.name == "scifact" and config.dataset.split == "test"
    assert config.dataset.path == Path("data") / "scifact"
    assert config.metrics.names == ("recall", "mrr", "ndcg")
    assert config.metrics.ks == (1, 5, 10, 100)
    assert config.mlflow.tracking_uri is None
    assert config.mlflow.examples.n == 10


def test_the_shipped_example_config_is_valid():
    config = load_config(EXAMPLE_CONFIG)

    assert config.experiment == "scifact-baselines"
    assert config.retriever.name == "bm25"


def test_load_config_reads_yaml_and_turns_lists_into_tuples(tmp_path):
    metrics = {"names": ["ndcg"], "ks": [3, 7]}
    path = _write_yaml(tmp_path, {**MINIMAL, "metrics": metrics, "mlflow": {"examples": {"metric": "ndcg@7"}}})

    config = load_config(path)

    assert config.metrics.names == ("ndcg",)
    assert config.metrics.ks == (3, 7)


@pytest.mark.parametrize("content", ["", "- just\n- a list\n", "42"])
def test_load_config_rejects_a_file_that_is_not_a_mapping(tmp_path, content):
    with pytest.raises(ValueError, match="mapping"):
        load_config(_write_yaml(tmp_path, content))


def test_config_is_immutable():
    with pytest.raises(ValidationError):
        _config().experiment = "other"


# --- validation fails early -------------------------------------------------

def test_unknown_retriever_is_rejected_and_the_available_ones_are_listed():
    with pytest.raises(ValidationError, match="bm25"):
        _config(retriever={"name": "nope"})


def test_unknown_metric_is_rejected():
    with pytest.raises(ValidationError, match="nope"):
        _config(metrics={"names": ["recall", "nope"]})


@pytest.mark.parametrize("ks", [[], [0], [5, -1]])
def test_invalid_ks_are_rejected(ks):
    with pytest.raises(ValidationError, match="ks"):
        _config(metrics={"ks": ks})


def test_empty_metric_names_are_rejected():
    with pytest.raises(ValidationError, match="at least one"):
        _config(metrics={"names": []})


@pytest.mark.parametrize("data", [
    {**MINIMAL, "experimnt": "typo"},
    {**MINIMAL, "dataset": {"nme": "scifact"}},
    {**MINIMAL, "mlflow": {"tags": {}, "extra": 1}},
])
def test_unknown_keys_are_rejected_so_typos_do_not_pass_silently(data):
    with pytest.raises(ValidationError, match="Extra inputs"):
        ExperimentConfig.model_validate(data)


def test_experiment_name_is_required_and_non_empty():
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate({"retriever": {"name": "bm25"}})
    with pytest.raises(ValidationError):
        _config(experiment="")


def test_example_metric_must_be_one_that_is_computed():
    with pytest.raises(ValidationError, match="not computed"):
        _config(metrics={"names": ["recall"], "ks": [5]}, mlflow={"examples": {"n": 3, "metric": "ndcg@10"}})


def test_example_metric_is_not_checked_when_examples_are_disabled():
    config = _config(metrics={"names": ["recall"], "ks": [5]}, mlflow={"examples": {"n": 0, "metric": "ndcg@10"}})

    assert config.mlflow.examples.n == 0


def test_negative_example_count_is_rejected():
    with pytest.raises(ValidationError):
        _config(mlflow={"examples": {"n": -1}})


# --- derived values ---------------------------------------------------------

def test_result_names_cover_every_metric_at_every_k():
    config = _config(metrics={"names": ["recall", "ndcg"], "ks": [1, 10]})

    assert config.metrics.result_names == ("recall@1", "recall@10", "ndcg@1", "ndcg@10")


def test_run_name_is_derived_from_the_retriever_unless_configured():
    assert _config().run_name == "bm25"
    assert _config(retriever={"name": "bm25", "params": {"k1": 1.2, "stem": True}}).run_name == "bm25 k1=1.2 stem=True"
    assert _config(mlflow={"run_name": "my run"}).run_name == "my run"


def test_flat_params_describe_the_whole_run():
    config = _config(retriever={"name": "bm25", "params": {"k1": 1.2}}, metrics={"names": ["ndcg"], "ks": [5, 10]})

    assert config.flat_params() == {
        "dataset.name": "scifact",
        "dataset.split": "test",
        "retriever.name": "bm25",
        "metrics.names": "ndcg",
        "metrics.ks": "5,10",
        "retriever.k1": 1.2,
    }
