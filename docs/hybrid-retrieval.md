# Hybrid retrieval (`retrievers/hybrid.py`, `retrievers/fusion.py`)

Asks several retrievers (here BM25 and a dense model) and merges their rankings into one with a **fusion strategy** chosen by name. The only one so far is **Reciprocal Rank Fusion (RRF)**; weighted fusion is next.

## How RRF works
Each document gets `1 / (rrf_k + rank)` points from every ranking it appears in; the points are added up and the documents sorted by total. Only positions are used, never scores, so it does not matter that BM25 scores (unbounded) and cosine similarities (about 0.3 to 0.8) are on different scales.

Example: rankings `[A, B, C]` and `[B, C, D]` with `rrf_k = 60` give B = 1/62 + 1/61 = 0.0325, C = 0.0320, A = 0.0164, D = 0.0159, so the order is B, C, A, D. A document ranked well by both lists beats one ranked first by only one.

## Fusion strategies (`fusion.py`)
A fusion function has the signature `fusion(rankings, k, **params) -> Ranking`: one ranking per retriever (each best first), how many results to return, and the strategy's own named settings. Strategies register with `@register_fusion("name")` and are looked up with `get_fusion(name)`, the same pattern as the retriever, vector store and metric registries. To add one, write the function and decorate it.

The contract every strategy must meet: return at most `k` unique documents, best first, with deterministic ties, and **validate its arguments even for empty rankings**. `HybridRetriever` relies on the last rule: when it is built, it calls the chosen fusion once on empty rankings, so a bad name, an unknown setting or an invalid value (e.g. `rrf_k: -5`, or later a weights list of the wrong length) fails immediately and not at the first query of an experiment.

Config: `fusion: rrf` and `fusion_params: {rrf_k: 60}` under the hybrid retriever's `params`.

## Choices
| Choice | Decision | Why |
|---|---|---|
| Fusion | Chosen by name from a registry; default `rrf`, unweighted | Needs no tuning and no score normalization, a standard first choice. Adding weighted score fusion later means writing one function and registering it |
| `rrf_k` | 60, set with `fusion_params` (must be positive) | The value from the original RRF paper. Larger values flatten the difference between ranks |
| Candidates | Each sub-retriever returns its top `candidates_per_retriever` (default 100, never fewer than `k`) | Documents found by only one retriever still get a chance to rank |
| Ties | Equal scores keep the order of first appearance (earlier ranking first, then better rank) | Deterministic results |
| Repeated documents | Only the first occurrence within one ranking counts | A sub-retriever bug cannot inflate a document |
| Indexing | `HybridRetriever.index` indexes every sub-retriever | Simple; the dense retriever's embedding cache keeps re-indexing cheap |
| Construction | Sub-retrievers by name (`components`, used by YAML configs) or as objects (`retrievers`, used by tests); at least two | Config stays plain data. Names, parameters and fusion settings are checked when the retriever is built, before any MLflow run starts |
| Scores returned | RRF scores (about 0.03), not BM25 or cosine | They are only meaningful for ordering |

Config example: `experiments/configs/hybrid_bm25_bge.yaml` (a `components` list with a `bm25` and a `dense` entry).

## Result (SciFact test, 300 queries, exact store)
| | BM25 | BGE-small | **Hybrid (RRF)** |
|---|---|---|---|
| nDCG@10 | 0.643 | **0.706** | 0.693 |
| recall@10 | 0.765 | **0.830** | 0.808 |
| recall@100 | 0.869 | 0.943 | **0.965** |
| MRR@10 | 0.609 | **0.673** | 0.662 |

**The hybrid does not beat BGE at the top of the ranking, but it finds the most relevant documents overall** (recall@100 0.965 against 0.943). RRF gives both retrievers an equal vote, and BM25 is clearly the weaker one here, so it drags the top ranks down while BM25's different mistakes help the deeper recall. Next experiments: weighted fusion that favours the dense model; fusing BM25 with a weaker dense model (MiniLM) where the gap is smaller; a reranker over the fused candidates.

## Tests (`tests/test_hybrid.py`)
- `tests/test_fusion.py`: the registry (lookup, unknown name, registering, duplicates); a **contract suite that runs against every registered fusion** through a parametrized fixture (unique documents, best first, trimmed to `k`, documents from one list kept, deterministic, empty rankings, `k` validated); and for RRF: exact points per document, ordering, agreement beats a single high rank, scores are ignored, ties by first appearance, effect of `rrf_k`, repeated documents, invalid arguments.
- `tests/test_hybrid.py`: `HybridRetriever` with recording fakes: indexes every sub-retriever on the same corpus, fuses their rankings, searches each to the candidate depth (or `k` if larger), trims, errors, and rejects a bad `k` before searching.
- Construction: exactly one of `components` / `retrievers`, at least two, positive parameters, components built from registered names, unknown names and bad parameters fail early, an end-to-end run with two real BM25 retrievers, registered and accepted in a config. Choosing the fusion: the default is RRF, the function is looked up by name and receives its params, an unknown fusion, unknown or clashing settings and invalid values all fail when the retriever is built.

## Open items
- Weighted score fusion as a second registered strategy (normalize each score list, then take a weighted average; tune the weight on the train split, never on the test queries). Its `weights` setting must have one entry per retriever, which the empty-rankings validation at construction will check. The contract test needs a `FUSION_PARAMS` entry for it.
- The default run name for a hybrid config is unreadable (it spells out every component), so set `mlflow.run_name` in the YAML. The component list is logged to MLflow as one long string param.
- Sub-retrievers are searched one after another; they could run in parallel if search speed ever matters.
