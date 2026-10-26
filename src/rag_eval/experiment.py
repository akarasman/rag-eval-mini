"""Run one experiment from a config: load data, build and index the retriever, evaluate, log to MLflow."""
import logging
from time import perf_counter
from typing import Optional

import mlflow

from .config import ExperimentConfig
from .data import Dataset
from .evaluate import EvaluationResult, evaluate
from .retrievers.registry import build_retriever
from .tracking import log_run


def run_experiment(config: ExperimentConfig, dataset: Optional[Dataset] = None) -> EvaluationResult:
    """Evaluate the configured retriever on the configured dataset and log the run.

    Args:
        dataset: use this instead of loading config.dataset (lets tests run on a tiny in-memory dataset).

    The dataset and retriever are built before the MLflow run starts, so a bad config or missing data
    does not leave an empty failed run behind.
    """
    if dataset is None:
        dataset = Dataset.load(config.dataset.name, config.dataset.split, config.dataset.data_dir)
    retriever = build_retriever(config.retriever.name, config.retriever.params)

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.experiment)

    with mlflow.start_run(run_name=config.run_name):
        start = perf_counter()
        retriever.index(dataset.corpus)
        index_seconds = perf_counter() - start

        start = perf_counter()
        result = evaluate(
            retriever, dataset.queries, dataset.qrels,
            ks=config.metrics.ks, metrics=config.metrics.names,
        )
        eval_seconds = perf_counter() - start

        log_run(config, dataset, retriever, result, index_seconds, eval_seconds)

    logging.info(f"Run '{config.run_name}' logged to experiment '{config.experiment}'")
    return result
