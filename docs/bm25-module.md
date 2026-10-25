# BM25 retriever (`src/rag_eval/retrievers/bm25.py`)

Design choices for the lexical baseline that the dense retriever has to beat.

## Overview
- `BM25Retriever(k1=1.5, b=0.75, stem=False)` implements the retriever interface: `index(corpus)` and `search(query, k)`.
- Documents are `tokenize(doc_to_text(doc))`; queries are `tokenize(query)`. See `docs/text-module.md`.

## Choices
| Choice | Decision | Why |
|---|---|---|
| Implementation | Library: `bm25s` | A baseline should be a trusted, widely used implementation. A hand-rolled version would make the comparison with dense retrieval less credible |
| Variant | Lucene (`method="lucene"`) | Non-negative idf, and closest to the Elasticsearch-style BM25 behind published BEIR baselines. "BM25" is a family of formulas, so library, version and variant are part of the result |
| Hyperparameters | Library defaults: `k1=1.5`, `b=0.75` | Untuned baseline. Tuning, if any, belongs on held-out queries, not on the ones reported |
| Tokenizer | Our own `tokenize`, passed to the library as token lists | Keeps the tokenizer under test and under our control; the library's own tokenizer is never used |
| Stemming | `stem` is a constructor argument, applied to documents and queries | A mismatch between index and query tokenization is silent (recall just drops), so the retriever owns the setting and it cannot diverge |
| Fewer than `k` matches | Pad with zero-score documents, in corpus order | Rankings always have `min(k, corpus size)` entries, like a retriever that scores everything. Padding documents have score `0.0` and no real signal |
| Scoring | Score every document per query, stable `argsort`, take top `k` | Simple and deterministic: ties (including all the zeros) keep corpus order. O(N log N) per query; fine for SciFact (~5k docs) |
| Empty query | All scores `0.0`, no library call | `bm25s.get_scores` raises `IndexError` on an empty token list; unknown words, by contrast, are handled by the library and score 0 |
| State on failure | `index()` assigns state last | A failed `index()` leaves the previous index intact, as in the oracle |

## Properties to remember
- Scores are not bounded and not comparable across queries or retrievers. This matters for the hybrid retriever, which should not add BM25 and cosine scores directly.
- Matching is exact on tokens: no synonyms ("tumour" vs "tumor"), and "heart attack" does not match "myocardial infarction". This is the gap dense retrieval is supposed to close.
- Padding documents tie at 0.0, so their order carries no information. Metrics should not be read as if the order of those entries meant anything.

## Tests (`tests/test_retrievers.py`)
Tiny four-document corpus where the right top result is obvious: correct top document, descending scores, zero-score padding in corpus order, cap at corpus size, trimming to `k`, empty/punctuation/unknown queries, case-insensitivity, stemming only when enabled, title is searchable, error cases, and re-indexing replacing the old corpus.

## Open items
- No comparison with reference scores yet; check against the published BM25 nDCG@10 for SciFact (about 0.67) once the evaluation loop runs on real data.
- Compare `stem=True` vs `False` and the title format on the real queries.
- `argpartition` instead of a full sort if the corpus ever gets large.
