"""Tune the weights of a two-retriever weighted hybrid on the TRAIN queries.

Weights picked by looking at the test queries would make the test result meaningless, so this sweeps the
weight on another split (train by default). Each retriever is searched once per query and the rankings
are reused for every weight, so the sweep costs about as much as one run. Every weight is logged to
MLflow as a run in the experiment "<experiment>-weight-tuning"; the best one is printed.

    python experiments/tune_weights.py experiments/configs/hybrid_weighted_bm25_bge.yaml

The config is a normal hybrid config with exactly two components; its `fusion` is ignored (always
"weighted") and its weights are replaced by the sweep. The weight of the FIRST component is swept from
0 to 1; the second gets the rest.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np

from rag_eval.config import ExperimentConfig, load_config
from rag_eval.data import Dataset
from rag_eval.evaluate import evaluate
from rag_eval.retrievers.base import Ranking, Retriever
from rag_eval.retrievers.hybrid import HybridRetriever
from rag_eval.retrievers.registry import build_retriever
from rag_eval.tracking import _mlflow_metric_name


class _MemoizedRetriever:
    """Wraps a retriever so each (query, k) is searched once and indexing happens once."""

    def __init__(self, retriever: Retriever):
        self._retriever = retriever
        self.name = retriever.name
        self._indexed = False
        self._searches: Dict[Tuple[str, int], Ranking] = {}

    def index(self, corpus) -> None:
        if not self._indexed:
            self._retriever.index(corpus)
            self._indexed = True

    def search(self, query: str, k: int = 10) -> Ranking:
        if (query, k) not in self._searches:
            self._searches[(query, k)] = self._retriever.search(query, k)
        return self._searches[(query, k)]


def _log_weight_run(config: ExperimentConfig, dataset: Dataset, weight: float, mean: Dict[str, float]) -> None:
    """Log one weight of the sweep as its own MLflow run.

    Args:
        config: the hybrid config being tuned; supplies the two component names.
        dataset: the dataset the weight was evaluated on; supplies the split and the query count.
        weight: the weight of the first component; the second component has 1 minus this.
        mean: result metric name -> mean over the queries, as returned by evaluate().
    """
    components = config.retriever.params["components"]
    with mlflow.start_run(run_name=f"weight {weight:.2f} / {1 - weight:.2f}"):
        mlflow.set_tags({"dataset": dataset.name, "split": dataset.split, "purpose": "weight tuning"})
        mlflow.log_params({
            "component_1": components[0]["name"],
            "component_2": components[1]["name"],
            "weight_1": round(weight, 4),
            "weight_2": round(1.0 - weight, 4),
            "num_queries": len(dataset.queries),
        })
        mlflow.log_metrics({_mlflow_metric_name(name): value for name, value in mean.items()})


def sweep(config: ExperimentConfig, split: str, weights: List[float], select_metric: str) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """Evaluate the weighted hybrid at each weight on `split`, log every weight, return the best.

    Args:
        config: a hybrid config with exactly two components. Its dataset name and data directory,
            its metrics (names and ks), its candidates_per_retriever and its MLflow options are used;
            its dataset split, fusion and weights are not.
        split: the split to evaluate on. It should not be the split the final result is reported on.
        weights: the weights to try for the FIRST component, each between 0 and 1; the second
            component gets 1 minus the weight.
        select_metric: the result metric the best weight is chosen by, e.g. "ndcg@10". It must be one
            of the metrics the config computes.

    Returns:
        (best weight of the first component, {weight: mean metrics}). The best is by `select_metric`;
        on a tie the smaller weight wins.

    Raises:
        ValueError: if the config is not a hybrid of exactly two components, or `select_metric` is not
            one of the config's result metrics.
    """
    if config.retriever.name != "hybrid" or len(config.retriever.params.get("components", [])) != 2:
        raise ValueError("the config must be a hybrid retriever with exactly two components")
    if select_metric not in config.metrics.result_names:
        raise ValueError(f"{select_metric!r} is not computed; choose one of {list(config.metrics.result_names)}")

    dataset = Dataset.load(config.dataset.name, split, config.dataset.data_dir)
    params = config.retriever.params
    parts = [_MemoizedRetriever(build_retriever(c["name"], c.get("params", {}))) for c in params["components"]]

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(f"{config.experiment}-weight-tuning")

    results: Dict[float, Dict[str, float]] = {}
    for weight in weights:
        hybrid = HybridRetriever(
            retrievers=parts,
            fusion="weighted",
            fusion_params={"weights": [weight, 1.0 - weight]},
            candidates_per_retriever=params.get("candidates_per_retriever", 100),
        )
        hybrid.index(dataset.corpus)
        mean = evaluate(hybrid, dataset.queries, dataset.qrels, ks=config.metrics.ks, metrics=config.metrics.names).mean
        results[weight] = mean

        _log_weight_run(config, dataset, weight, mean)

    best = max(results, key=lambda w: (results[w][select_metric], -w))  # ties: the smaller first weight
    return best, results


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Tune weighted-fusion weights on a non-test split.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--split", default="train", help="split to tune on (never the one you report on)")
    parser.add_argument("--steps", type=int, default=10, help="number of intervals between weight 0 and 1")
    parser.add_argument("--metric", default="ndcg@10", help="result metric used to pick the best weight")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    config = load_config(args.config)
    if args.split == config.dataset.split:
        parser.error(f"tuning split {args.split!r} is the config's evaluation split; tune on a different one")

    weights = [float(w) for w in np.linspace(0.0, 1.0, args.steps + 1)]
    best, results = sweep(config, args.split, weights, args.metric)

    names = [c["name"] for c in config.retriever.params["components"]]
    print(f"\n{args.metric} on {args.split}: weight of {names[0]} (the rest goes to {names[1]})")
    for weight in weights:
        marker = "  <- best" if weight == best else ""
        print(f"  {weight:.1f}  {results[weight][args.metric]:.4f}{marker}")
    print(f"\nbest weights: {names[0]} {best:.2f} / {names[1]} {1 - best:.2f}")


if __name__ == "__main__":
    main()
