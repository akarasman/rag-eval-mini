# Utils (`src/rag_eval/utils.py`)

Small helpers shared by several modules. Currently one: `top_k_indices`.

## `top_k_indices(scores, k)`
Indices of the `k` highest scores, best first; equal scores keep index order.

**Used by** `BM25Retriever.search` (over the documents that match the query) and `ExactVectorStore.search` (over all stored vectors). Anything that turns an array of scores into a ranking should use it.

| Choice | Decision | Why |
|---|---|---|
| Algorithm | Find the k-th largest score with `np.partition` (O(n)), take everything above it, fill the remaining places from the scores equal to it, sort only those k | The full `argsort` it replaces is O(n log n) and sorts scores nobody asked for. Measured with k=10: 6.6x faster at 5,000 scores, 25x at 500,000 |
| Same answer as `argsort` | Identical to `np.argsort(-scores, kind="stable")[:k]`, including tie order | Rankings stay reproducible, and results did not change when it replaced the sort (BM25 on SciFact: nDCG@10 0.6434 before and after) |
| Ties | Lowest index first, also at the cutoff | A plain `np.argpartition` picks arbitrarily among scores equal to the k-th, so which documents make the top k would depend on the implementation. Ties are common with short documents and identical term counts |
| `k >= len(scores)` | Returns all indices, sorted | Same as slicing a full sort |
| Invalid input | `ValueError` for `k <= 0` or non-1-D scores | A silent wrong answer is worse than an error; `np.partition` would otherwise fail with an obscure index error |
| NaN | Not supported (documented, not checked) | Checking costs another O(n) pass for something our scores never produce |

## Tests (`tests/test_utils.py`)
- **Equal to argsort:** many random score arrays and values of `k` (below, at and above the size); arrays with only 1, 2 or 6 distinct values so ties are everywhere; float32/float64/int32/int64.
- **Behaviour:** best first; ties by index including at the cutoff; negative and infinite scores; `k` at least the size; a single score; an empty array; integer index array usable for indexing; input not modified; deterministic.
- **Errors:** non-positive `k`, scores that are not 1-D.

## Open items
- If a retriever ever needs top-k for many queries at once (batch search), a 2-D version over rows would avoid a Python loop.
- NaN scores are undefined behavior.
