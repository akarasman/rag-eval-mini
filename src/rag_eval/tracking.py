"""MLflow logging for one experiment run.

Logs a small, readable record: the config as params and as a YAML artifact, the headline metrics, a few
timings, the per-query scores, and the best and worst queries with what was relevant and what was
retrieved. The evaluation itself never touches MLflow; everything here only reads its results.

Layout in MLflow: experiment = config.experiment (comparable runs), run = one config, tags for
dataset / split / retriever so runs can be filtered and compared.

The only public function is log_run(); everything else is a private helper.
"""
import csv
import io
from typing import Any, Dict, List, Tuple

import mlflow
import yaml

from .config import ExamplesConfig, ExperimentConfig
from .data import Dataset
from .evaluate import EvaluationResult
from .retrievers.base import Retriever

_TOP_K_SHOWN = 10  # retrieved documents listed per example query


def _mlflow_metric_name(name: str) -> str:
    """Translate one of our metric names into a name MLflow accepts.

    Args:
        name: a result metric name such as "recall@10".

    Returns:
        The name with "@" replaced ("recall_at_10"); MLflow rejects "@".
    """
    return name.replace("@", "_at_")


def _select_examples(per_query: Dict[str, Dict[str, float]], metric: str, n: int) -> Tuple[List[str], List[str]]:
    """Pick the worst and best queries by one metric.

    Ties are broken by query ID, so the selection is reproducible.

    Args:
        per_query: query_id -> {result metric name -> value}.
        metric: the result metric to rank by, e.g. "ndcg@10".
        n: how many queries to take from each end.

    Returns:
        (worst, best): query IDs, lowest score first and highest score first. The lists never overlap,
        so with fewer than 2n queries the best list shrinks.
    """
    ordered = sorted(per_query, key=lambda query_id: (per_query[query_id][metric], query_id))
    worst = ordered[:n]
    best = [query_id for query_id in reversed(ordered) if query_id not in worst][:n]
    return worst, best


def _build_example(
    query_id: str,
    dataset: Dataset,
    retriever: Retriever,
    scores: Dict[str, float],
    search_k: int,
) -> Dict[str, Any]:
    """Describe one query for human inspection.

    Searches again for this query so it can report where each relevant document was found.

    Args:
        query_id: the query to describe; must be in dataset.queries and dataset.qrels.
        dataset: supplies the query text, the relevance judgments and the document titles.
        retriever: an indexed retriever.
        scores: this query's result metrics, stored as they are.
        search_k: how deep to search, so relevant documents ranked below the top are still located.

    Returns:
        A JSON-friendly dict with the query, its scores, "relevant" (each relevant doc with the rank it
        was found at, or None if not in the top search_k) and "retrieved" (the top few results with their
        relevance grade, 0 if not relevant).
    """
    ranking = retriever.search(dataset.queries[query_id], k=search_k)
    ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(ranking, start=1)}
    relevant = dataset.qrels[query_id]

    return {
        "query_id": query_id,
        "query": dataset.queries[query_id],
        "scores": scores,
        "relevant": [
            {
                "doc_id": doc_id,
                "grade": grade,
                "title": dataset.corpus[doc_id]["title"],
                "rank": ranks.get(doc_id),
            }
            for doc_id, grade in sorted(relevant.items(), key=lambda item: (-item[1], item[0]))
        ],
        "retrieved": [
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": score,
                "title": dataset.corpus[doc_id]["title"],
                "relevance": relevant.get(doc_id, 0),
            }
            for rank, (doc_id, score) in enumerate(ranking[:_TOP_K_SHOWN], start=1)
        ],
    }


def _log_examples(
    examples: ExamplesConfig,
    dataset: Dataset,
    retriever: Retriever,
    per_query: Dict[str, Dict[str, float]],
    ks: Tuple[int, ...],
) -> None:
    """Log the worst and best queries as the artifacts examples/worst.json and examples/best.json.

    Args:
        examples: how many queries to take and which metric ranks them.
        dataset: the dataset that was evaluated.
        retriever: the indexed retriever, searched again for the selected queries.
        per_query: query_id -> {result metric name -> value}.
        ks: the evaluated cutoffs; each example searches to the largest one.
    """
    worst, best = _select_examples(per_query, examples.metric, examples.n)
    search_k = max(ks)
    for label, query_ids in (("worst", worst), ("best", best)):
        records = [
            _build_example(query_id, dataset, retriever, per_query[query_id], search_k)
            for query_id in query_ids
        ]
        mlflow.log_dict({"ranked_by": examples.metric, "queries": records}, f"examples/{label}.json")


def _per_query_csv(result: EvaluationResult, queries: Dict[str, str]) -> str:
    """Format the per-query scores as CSV.

    Args:
        result: the evaluation result; every query must have the same metrics.
        queries: query_id -> query text.

    Returns:
        CSV text with the columns query_id, query, then one column per result metric.
    """
    metric_names = list(next(iter(result.per_query.values())))
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["query_id", "query", *metric_names])
    for query_id, scores in result.per_query.items():
        writer.writerow([query_id, queries[query_id], *(f"{scores[name]:.6f}" for name in metric_names)])
    return buffer.getvalue()


def log_run(
    config: ExperimentConfig,
    dataset: Dataset,
    retriever: Retriever,
    result: EvaluationResult,
    index_seconds: float,
    eval_seconds: float,
) -> None:
    """Log everything about a finished evaluation to the ACTIVE MLflow run.

    Call it inside mlflow.start_run(). Logs tags, params, metrics and timings, and the artifacts
    config.yaml, per_query.csv (if config.mlflow.log_per_query) and examples/*.json (if
    config.mlflow.examples.n > 0).

    Args:
        config: the experiment config that produced the run.
        dataset: the dataset that was evaluated.
        retriever: the indexed retriever, used to build the example queries.
        result: the evaluation result to log.
        index_seconds: time spent indexing the corpus.
        eval_seconds: time spent evaluating all queries (includes computing the metrics).
    """
    mlflow.set_tags({
        "dataset": config.dataset.name,
        "split": config.dataset.split,
        "retriever": config.retriever.name,
        **config.mlflow.tags,
    })
    mlflow.log_params({**config.flat_params(), "num_docs": len(dataset.corpus), "num_queries": len(dataset.queries)})
    mlflow.log_metrics({_mlflow_metric_name(name): value for name, value in result.mean.items()})
    mlflow.log_metrics({"index_seconds": index_seconds, "eval_seconds": eval_seconds})

    mlflow.log_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False), "config.yaml")
    if config.mlflow.log_per_query:
        mlflow.log_text(_per_query_csv(result, dataset.queries), "per_query.csv")

    if config.mlflow.examples.n > 0:
        _log_examples(config.mlflow.examples, dataset, retriever, result.per_query, config.metrics.ks)
