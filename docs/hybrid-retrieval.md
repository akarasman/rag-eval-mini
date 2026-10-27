# Hybrid retrieval (`retrievers/hybrid.py`, `retrievers/fusion.py`)

Asks several retrievers (here BM25 and a dense model) and merges their rankings into one with a **fusion strategy** chosen by name. Two are registered: **Reciprocal Rank Fusion (RRF)** and **weighted score fusion**.

## How RRF works
Each document gets `1 / (rrf_k + rank)` points from every ranking it appears in; the points are added up and the documents sorted by total. Only positions are used, never scores, so it does not matter that BM25 scores (unbounded) and cosine similarities (about 0.3 to 0.8) are on different scales.

Example: rankings `[A, B, C]` and `[B, C, D]` with `rrf_k = 60` give B = 1/62 + 1/61 = 0.0325, C = 0.0320, A = 0.0164, D = 0.0159, so the order is B, C, A, D. A document ranked well by both lists beats one ranked first by only one.

## Fusion strategies (`fusion.py`)
A fusion function has the signature `fusion(rankings, k, **params) -> Ranking`: one ranking per retriever (each best first), how many results to return, and the strategy's own named settings. Strategies register with `@register_fusion("name")` and are looked up with `get_fusion(name)`, the same pattern as the retriever, vector store and metric registries. To add one, write the function and decorate it.

The contract every strategy must meet: return at most `k` unique documents, best first, with deterministic ties, and **validate its arguments even for empty rankings**. `HybridRetriever` relies on the last rule: when it is built, it calls the chosen fusion once on empty rankings, so a bad name, an unknown setting or an invalid value (e.g. `rrf_k: -5`, or later a weights list of the wrong length) fails immediately and not at the first query of an experiment.

Config: `fusion: rrf` and `fusion_params: {rrf_k: 60}` under the hybrid retriever's `params`.

## Weighted score fusion (`fusion: weighted`)
Each retriever's scores are rescaled to 0 to 1 for the query (its best document 1, its worst 0), multiplied by that retriever's weight and added up. Min-max rescaling is what makes BM25 scores and cosine similarities comparable, and it means multiplying or shifting one retriever's scores changes nothing. A document missing from a retriever's list gets 0 from it, the same as that list's worst document. Unlike RRF it uses how far apart the scores are, not only the order.

Settings: `weights`, one per component, rescaled to sum to 1 (default: equal). A weight list of the wrong length, a negative or non-finite weight, or all zeros is rejected when the retriever is built. A list whose scores are all equal carries no ordering information, so each of its documents gets the full 1.0.

**Choosing the weights without cheating.** Weights picked by looking at the test results would make the test result meaningless. `experiments/tune_weights.py` sweeps the first component's weight from 0 to 1 on the **train** queries (809 queries, none shared with the 300 test queries), searching each query once per retriever and reusing the results for every weight. Each weight is logged to MLflow as a run in `<experiment>-weight-tuning`. The chosen weights are then written into the config and evaluated on test once. Train nDCG@10 for BM25 weight 0 to 1 (BGE gets the rest): 0.748, 0.753, 0.758, **0.758**, 0.754, 0.743, 0.725, 0.703, 0.683, 0.663, 0.642. So the best split is **BM25 0.3 / BGE 0.7**, and the curve is flat between 0.2 and 0.4.

## Choices
| Choice | Decision | Why |
|---|---|---|
| Fusion | Chosen by name from a registry; default `rrf`, unweighted | Needs no tuning and no score normalization, a standard first choice. Adding weighted fusion meant writing one function and registering it |
| `rrf_k` | 60, set with `fusion_params` (must be positive) | The value from the original RRF paper. Larger values flatten the difference between ranks |
| Candidates | Each sub-retriever returns its top `candidates_per_retriever` (default 100, never fewer than `k`) | Documents found by only one retriever still get a chance to rank |
| Ties | Equal scores keep the order of first appearance (earlier ranking first, then better rank) | Deterministic results |
| Repeated documents | Only the first occurrence within one ranking counts | A sub-retriever bug cannot inflate a document |
| Indexing | `HybridRetriever.index` indexes every sub-retriever | Simple; the dense retriever's embedding cache keeps re-indexing cheap |
| Construction | Sub-retrievers by name (`components`, used by YAML configs) or as objects (`retrievers`, used by tests); at least two | Config stays plain data. Names, parameters and fusion settings are checked when the retriever is built, before any MLflow run starts |
| Scores returned | RRF scores (about 0.03), not BM25 or cosine | They are only meaningful for ordering |

Config example: `experiments/configs/hybrid_bm25_bge.yaml` (a `components` list with a `bm25` and a `dense` entry).

## Results (SciFact test, 300 queries, exact store)
nDCG@10, with recall@100 in brackets:

| Setup | nDCG@10 | (recall@100) |
|---|---|---|
| BM25 | 0.643 | (0.869) |
| MiniLM | 0.648 | (0.925) |
| BGE-small | 0.706 | (0.943) |
| RRF: BM25 + MiniLM | 0.676 | (0.955) |
| RRF: MiniLM + BGE | 0.696 | (0.947) |
| RRF: BM25 + BGE | 0.693 | (0.965) |
| **Weighted 0.3 / 0.7: BM25 + BGE** | **0.716** | **(0.968)** |

What this shows:
- **RRF helps when the two retrievers are about equally strong** (BM25 + MiniLM gains 3 points over either alone). With a much stronger partner (BGE) an equal vote slightly lowers the top ranks, because the weaker retriever's mistakes count as much as the stronger one's.
- **Two dense models add little to each other** (MiniLM + BGE): their mistakes are similar. BM25 + a dense model works better because they fail on different queries.
- **Weighted fusion, with weights tuned on train, beats the best single retriever** (0.716 against 0.706 for BGE alone) and has the best recall@100. The gain over BGE is about one point; with 300 test queries it could be within noise, and I have not tested significance.

## Tests (`tests/test_hybrid.py`)
- `tests/test_fusion.py` also covers weighted fusion: exact fused scores, the weights deciding the winner, default equal weights, weights rescaled to sum to 1, scores in [0, 1], invariance to rescaling one retriever's scores, score gaps mattering (unlike RRF), a zero weight, constant-score lists, ties, repeated documents, and every invalid-weights case even for empty rankings.
- `tests/test_fusion.py`: the registry (lookup, unknown name, registering, duplicates); a **contract suite that runs against every registered fusion** through a parametrized fixture (unique documents, best first, trimmed to `k`, documents from one list kept, deterministic, empty rankings, `k` validated); and for RRF: exact points per document, ordering, agreement beats a single high rank, scores are ignored, ties by first appearance, effect of `rrf_k`, repeated documents, invalid arguments.
- `tests/test_hybrid.py`: `HybridRetriever` with recording fakes: indexes every sub-retriever on the same corpus, fuses their rankings, searches each to the candidate depth (or `k` if larger), trims, errors, and rejects a bad `k` before searching.
- Construction: exactly one of `components` / `retrievers`, at least two, positive parameters, components built from registered names, unknown names and bad parameters fail early, an end-to-end run with two real BM25 retrievers, registered and accepted in a config. Choosing the fusion: the default is RRF, the function is looked up by name and receives its params, an unknown fusion, unknown or clashing settings and invalid values all fail when the retriever is built.

## Open items
- Significance: no confidence intervals or paired test yet, so small differences between setups may be noise. The per-query scores in MLflow allow a paired bootstrap.
- Other normalizations for weighted fusion (z-score); a weight search for three or more retrievers (the script handles exactly two).
- A cross-encoder reranker over the fused candidates, since the relevant documents are mostly in the top 100 but not the top 10.
- The default run name for a hybrid config is unreadable (it spells out every component), so set `mlflow.run_name` in the YAML. The component list is logged to MLflow as one long string param.
- Sub-retrievers are searched one after another; they could run in parallel if search speed ever matters.
