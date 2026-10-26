# Experiment runner (`experiments/run_experiments.py`, `src/rag_eval/experiment.py`, `src/rag_eval/tracking.py`)

Runs experiment configs and logs each to MLflow. See `docs/config-module.md` for the config itself.

    PYTHONPATH=src python experiments/run_experiments.py experiments/configs/bm25_baseline.yaml

## Structure
- `experiments/run_experiments.py`: thin CLI. Loads and validates ALL given configs first, then runs each one and prints the mean metrics.
- `experiment.run_experiment(config, dataset=None)`: load the dataset, build and index the retriever, `evaluate`, log. Taking an optional dataset lets tests run on a tiny in-memory one.
- `tracking.py`: everything MLflow-specific. `evaluate` never touches MLflow.

## What a run logs
| Kind | Content |
|---|---|
| MLflow experiment | `config.experiment` (comparable runs go under one name, e.g. `scifact-baselines`) |
| Run name | Configured, or derived (`bm25 k1=1.5 b=0.75 stem=False`) |
| Tags | `dataset`, `split`, `retriever`, plus the config's own tags |
| Params | Flat config (`retriever.k1`, `metrics.ks`, ...), plus `num_docs`, `num_queries` |
| Metrics | Mean of every `<name>@<k>`, plus `index_seconds` and `eval_seconds` |
| Artifacts | `config.yaml` (the validated config), `per_query.csv` (one row per query), `examples/worst.json` and `examples/best.json` |

Examples are the `n` worst and `n` best queries by one metric (default 10 by `ndcg@10`, ties broken by query ID). Each record has the query, its scores, its relevant documents with the rank they were found at (`null` = not in the top `max(ks)`), and the top 10 retrieved documents with their relevance grade. Titles only, no full text, to keep the artifacts small. Examples are built by searching again for just those queries.

## Choices
| Choice | Decision | Why |
|---|---|---|
| Metric names | `recall@10` is logged as `recall_at_10` | MLflow rejects `@` (it allows alphanumerics, `_`, `-`, `.`, space, `/`). Translation happens only at the logging boundary; everywhere else keeps `@` |
| Tracking backend | MLflow's default: SQLite `mlflow.db` in the working directory (gitignored) | MLflow 3.x refuses the old `./mlruns` file store unless opted in. A different backend is the config's `tracking_uri` |
| Fail early | Dataset and retriever are built before the MLflow run starts | A bad config or missing data leaves no empty failed run behind |
| Validate everything first | The CLI loads all configs before running any | A typo in the last file does not waste earlier runs |
| Logging scope | Per-query scores and examples are logged but can be turned off (`log_per_query`, `examples.n: 0`) | Lightweight by default, detail on request |
| Timings | `index_seconds` and `eval_seconds` (the latter includes metric computation) | Cheap, and index cost matters for the dense retriever |
| Dataset identity | Tags and params come from the config, not from the loaded object | Real runs load the dataset from the config, so they agree |

## First result (BM25, SciFact test, 300 queries)
nDCG@10 = 0.643, recall@10 = 0.765, recall@100 = 0.869, MRR@10 = 0.609. Indexing took about 1 s, evaluation about 0.15 s. This is below the roughly 0.67 usually quoted for BM25 on SciFact in the BEIR paper. Likely causes: our tokenizer (no stopword removal, optional stemming), different BM25 variant and defaults, and the library's idf. Not investigated yet.

## Tests
- `tests/test_tracking.py`: metric name translation; best/worst selection (order, tie-break, no overlap, zero); example records (ranks, grades, search depth, listing limit); CSV rows including a query containing a comma.
- `tests/test_experiment.py`: real BM25 on a four-document dataset, logged to a temporary SQLite database via a fixture that restores the global tracking URI. Checks the result, params, tags, metrics, artifacts, the optional-logging switches, and that bad retriever params fail before any run is created.

## Open items
- Explain the gap to the published BM25 number; the obvious first experiment is a config with `stem: true`, then `k1`/`b` variants on held-out queries.
- MLflow's SQLite store prints a SQLAlchemy deprecation warning during tests (from MLflow itself, not our code).
- No retry or resume: a failed run is just a failed MLflow run.
- Only one config at a time per process; sweeps are several config files given to the CLI.
- `experiments/` is not a package, so the CLI needs `PYTHONPATH=src` or an editable install (`pip install -e .`).
