# Experiment config (`src/rag_eval/config.py`, `src/rag_eval/retrievers/registry.py`)

Design choices for describing an experiment run in a file. Example: `experiments/configs/bm25_baseline.yaml`.

## Overview
- One config describes ONE run: one retriever on one dataset split, with the metrics to compute and the MLflow options.
- `load_config(path)` reads YAML and validates it into an `ExperimentConfig` (pydantic, frozen, unknown keys rejected).
- Retrievers and metrics are chosen by registered name, so the config is plain data and adding either needs no config code.

## Choices
| Choice | Decision | Why |
|---|---|---|
| Format | YAML + pydantic validation | Comments next to settings, easy diffs. Adds `pyyaml`; pydantic was already in requirements (it was kept for exactly this) |
| Scope | One config = one run; a sweep is several files or a script that varies a base config | Simple schema; each MLflow run maps to exactly one file. A config holding a list of variants was rejected for now (shared settings would be written once, but hybrid retrievers add nesting) |
| Typos | `extra="forbid"` on every model | A misspelled key would otherwise be ignored and the run would silently use a default |
| Fail early | Retriever name, metric names, ks and the example metric are validated at load time | Nothing is downloaded, indexed or embedded before the config is known to be consistent |
| Retriever | `name` + free-form `params` passed to the constructor | `build_retriever` turns bad parameters into a `ValueError` naming the retriever. Params are not validated at load time (a dense retriever would have to load a model to check them) |
| Retriever registry | `@register_retriever("name")`; modules registered by importing them in `retrievers/__init__.py` | Same pattern as the metric registry. The oracle is deliberately not registered: it needs queries and qrels |
| Dataset | `name`, `data_dir`, `split`; path is `<data_dir>/<name>` | Matches `Dataset.load(name, split, data_dir)` |
| Metrics | `names` and `ks`; every metric is computed at every k, keyed `<name>@<k>` | Mirrors `evaluate`. Tuples, because `evaluate` takes tuples |
| MLflow experiment | Top-level `experiment` | Groups comparable runs (e.g. `scifact-baselines`) |
| MLflow run name | Configured, or derived from the retriever and its params (`bm25 k1=1.2 stem=True`) | Runs are recognizable in the UI without naming each one |
| MLflow options | `tracking_uri` (None = local `./mlruns`), `tags`, `log_per_query`, `examples` | `log_per_query` stores the per-query scores as an artifact |
| Query examples | The `n` best and `n` worst queries by one result metric (default 10, `ndcg@10`); `n: 0` disables | Shows what works and what fails; the metric must be one the config computes |

## Derived values
- `ExperimentConfig.run_name` and `flat_params()` (a flat `dataset.name`, `retriever.k1`, ... view for `mlflow.log_params`).
- `MetricsConfig.result_names`: every `<name>@<k>` key evaluation will produce.

## Tests (`tests/test_config.py`, registry part of `tests/test_retrievers.py`)
Defaults, the shipped example file, YAML lists becoming tuples, non-mapping files, immutability; unknown retriever, metric, key (typo); invalid ks; the example metric must be computed unless examples are disabled; run name and flat params; the retriever registry (build from params, unknown name, bad params, duplicate).

## Open items
- The default example metric `ndcg@10` breaks configs whose `ks` has no 10, until `examples.metric` is set (or `n: 0`). Strict on purpose; a default tied to the largest k would be friendlier.
- Hybrid retrievers need nested retriever configs; the `RetrieverConfig` schema will have to grow (or hybrid takes registered names plus params).
- Dataset splits: `split` is a single string; tuning on `train` and reporting on `test` is two runs.
- The runner (`experiments/run_experiments.py`) does not use the config yet.
