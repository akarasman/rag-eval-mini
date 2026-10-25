# Evaluation module (`src/rag_eval/evaluate.py`, metric registry in `src/rag_eval/metrics.py`)

Design choices for scoring a retriever over a set of queries, and for choosing which metrics to compute.

## Overview
- `evaluate(retriever, queries, qrels, ks, metrics)` searches every query once and scores the ranking with each selected metric at each `k`.
- It returns an `EvaluationResult` with `mean` (flat dict, e.g. `{"recall@10": ...}`) and `per_query` (`query_id -> {metric: value}`).
- Metrics are pure functions in `metrics.py`, looked up by name in a registry, so an experiment config can pick them as a tuple of names.

## `evaluate`
| Choice | Decision | Why |
|---|---|---|
| Return value | `EvaluationResult(mean, per_query)` | The mean is derived from the per-query values, so keeping them costs nothing and enables error analysis ("which queries does BM25 fail?") and paired comparisons between retrievers. A flat dict of means would throw that away |
| Searching | Once per query with `k = max(ks)`; the same ranking is scored at every `k` | Avoids repeated searches. Valid because every metric only looks at the top `k` of the ranking |
| Averaging | Macro mean: every query counts equally | Standard for recall, MRR and nDCG (BEIR reports the same) |
| Short or empty rankings | Scored as they are | Missing documents count as misses. This is why BM25 does not pad (see `docs/bm25-module.md`) |
| Query without relevant docs | Raises `ValueError` naming the query | `Dataset.load` already guarantees every query has relevant docs, so a violation is a bug upstream. Skipping would silently shrink the query set and make runs incomparable |
| Validation timing | Everything (`ks`, metric names, queries, qrels) is checked before the first search | A bad input fails immediately, not after a long run |
| Result keys | `<name>@<k>`, e.g. `recall@10` | One bare metric name in the config, one value per cutoff in the output |

## Metric registry
| Choice | Decision | Why |
|---|---|---|
| Selection | `evaluate(..., metrics=("recall", "mrr", "ndcg"))`, names looked up in `METRIC_REGISTRY` | A config is just a tuple of strings |
| Registration | `@register_metric("name")` decorator; returns the function unchanged | The name sits next to the function; no separate dict to forget to update and no ordering problem between the dict and the functions. Duplicate names raise |
| Lookup | `get_metric(name)`; an unknown name raises `ValueError` listing the available ones | Fails early with a useful message |
| Names | Bare (`recall`, not `recall@k`); `evaluate` appends `@<k>` | Every registered metric takes a cutoff, so `@k` in the key was a template baked into the name |
| Type | `MetricFunction = Callable[[List[str], Dict[str, int], int], float]` (alias, not a Protocol) | Signature is (ranked IDs, relevance grades, `k`) -> score. An alias is enough; it does not check parameter names |
| Registry state | Module-level dict | Simple; tests patch it with `monkeypatch` |

**Adding a metric:** write a function with the `MetricFunction` signature in `metrics.py`, decorate it with `@register_metric("name")`, and list the name in the experiment's `metrics`.

## Tests
- `tests/test_evaluate.py`: fake retrievers with preset rankings, so expected numbers are worked out by hand; the mean equals the average of the per-query values; one search per query with the largest `k` (recording fake); short and empty rankings count as misses; the oracle scores perfectly (and `recall@1` is capped by the number of relevant docs); metric selection; a newly registered metric is picked up; every error case, including that errors are raised before any search.
- `tests/test_metrics.py` (registry part): every registered metric scores a trivially perfect ranking as 1.0 (catches a wrong function or signature registered under a name; assumes metrics are bounded with 1.0 as best); `get_metric` returns the right functions and lists names on error; the decorator registers, returns the function unchanged, and rejects duplicates.

## Open items
- **Metrics without a cutoff** (e.g. MAP over the whole list) do not fit the `(ranked_ids, relevant, k)` signature. Options considered: keep `k` required and use a large `k` (current); a `cutoff=False` flag on the decorator, where `evaluate` calls such a metric once and reports it without `@k`; or separate `@k` metrics from raw ones in the registry. Deferred until a real metric needs it.
- **MLflow metric names:** `recall@10` may be rejected by MLflow, which I believe restricts names to alphanumerics, `_`, `-`, `.`, space and `/`. Not verified (mlflow not installed here). If so, translate names at the logging boundary only.
- The oracle still pads rankings with real corpus documents by default, unlike BM25 (see `docs/bm25-module.md`).
- Per-query results are returned but not yet stored anywhere; decide in the experiment runner whether to save them (e.g. as an MLflow artifact).
- `reciprocal_rank` accepts `k=None` (whole ranking); the registry type only promises an `int`.
