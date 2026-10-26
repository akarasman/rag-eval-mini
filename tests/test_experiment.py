"""End-to-end test of run_experiment: a real BM25 retriever on a tiny dataset, logged to a temporary MLflow database."""
import json

import mlflow
import pytest
from mlflow import MlflowClient

from rag_eval.config import ExperimentConfig
from rag_eval.data import Dataset
from rag_eval.experiment import run_experiment


@pytest.fixture
def tracking_uri(tmp_path):
    """A throwaway SQLite MLflow backend; the global tracking URI is restored afterwards."""
    previous = mlflow.get_tracking_uri()
    yield f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(previous)


def _dataset() -> Dataset:
    corpus = {
        "d1": {"title": "Tumor suppressor", "text": "The p53 protein suppresses tumor growth."},
        "d2": {"title": "Heart disease", "text": "Myocardial infarction damages heart muscle."},
        "d3": {"title": "Diet", "text": "Vitamin D intake and bone density."},
        "d4": {"title": "Gene expression", "text": "Mutations in p53 alter gene expression."},
    }
    queries = {"q1": "p53 tumor", "q2": "heart muscle damage", "q3": "unknownword"}
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}}  # q3 cannot be found
    return Dataset(name="tiny", split="test", corpus=corpus, queries=queries, qrels=qrels)


def _config(tracking_uri: str, **mlflow_options) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "experiment": "tiny-exp",
        "dataset": {"name": "tiny"},  # matches the injected dataset; real runs load it from this section
        "retriever": {"name": "bm25", "params": {"k1": 1.2}},
        "metrics": {"names": ["recall", "ndcg"], "ks": [1, 5]},
        "mlflow": {
            "tracking_uri": tracking_uri,
            "tags": {"note": "test"},
            "examples": {"n": 1, "metric": "ndcg@5"},
            **mlflow_options,
        },
    })


def _only_run(experiment: str):
    """Used to check specific experiment runs only once."""
    client = MlflowClient()
    runs = client.search_runs([client.get_experiment_by_name(experiment).experiment_id])
    assert len(runs) == 1
    return client, runs[0]


def test_run_experiment_returns_the_evaluation_result(tracking_uri):
    result = run_experiment(_config(tracking_uri), _dataset())

    assert result.mean["recall@5"] == pytest.approx(2 / 3)  # q3 is never found
    assert set(result.per_query) == {"q1", "q2", "q3"}


def test_run_experiment_logs_params_metrics_and_tags(tracking_uri):
    run_experiment(_config(tracking_uri), _dataset())
    _, run = _only_run("tiny-exp")

    assert run.info.run_name == "bm25 k1=1.2"
    assert run.data.params["retriever.name"] == "bm25"
    assert run.data.params["retriever.k1"] == "1.2"
    assert run.data.params["num_docs"] == "4" and run.data.params["num_queries"] == "3"
    assert run.data.tags["dataset"] == "tiny" and run.data.tags["retriever"] == "bm25"
    assert run.data.tags["note"] == "test"
    assert run.data.metrics["recall_at_5"] == pytest.approx(2 / 3)  # "@" is not allowed in MLflow names
    assert {"ndcg_at_1", "ndcg_at_5", "recall_at_1", "index_seconds", "eval_seconds"} <= set(run.data.metrics)


def test_run_experiment_logs_the_config_per_query_scores_and_best_and_worst_examples(tracking_uri):
    run_experiment(_config(tracking_uri), _dataset())
    client, run = _only_run("tiny-exp")

    artifacts = {a.path for a in client.list_artifacts(run.info.run_id)}
    assert {"config.yaml", "per_query.csv", "examples"} <= artifacts

    worst = json.loads(mlflow.artifacts.load_text(f"{run.info.artifact_uri}/examples/worst.json"))
    best = json.loads(mlflow.artifacts.load_text(f"{run.info.artifact_uri}/examples/best.json"))
    assert worst["ranked_by"] == "ndcg@5"
    assert [q["query_id"] for q in worst["queries"]] == ["q3"]  # nothing matches "unknownword"
    assert best["queries"][0]["query_id"] in {"q1", "q2"}
    assert worst["queries"][0]["relevant"][0]["rank"] is None


def test_run_experiment_can_skip_per_query_scores_and_examples(tracking_uri):
    config = _config(tracking_uri, log_per_query=False, examples={"n": 0})

    run_experiment(config, _dataset())
    client, run = _only_run("tiny-exp")

    assert {a.path for a in client.list_artifacts(run.info.run_id)} == {"config.yaml"}


def test_run_experiment_fails_before_starting_a_run_for_unusable_retriever_params(tracking_uri):
    config = ExperimentConfig.model_validate({
        "experiment": "tiny-exp",
        "retriever": {"name": "bm25", "params": {"typo": 1}},
        "mlflow": {"tracking_uri": tracking_uri, "examples": {"n": 0}},
    })

    with pytest.raises(ValueError, match="typo"):
        run_experiment(config, _dataset())

    client = MlflowClient()
    experiment = client.get_experiment_by_name("tiny-exp")
    assert experiment is None or client.search_runs([experiment.experiment_id]) == []
