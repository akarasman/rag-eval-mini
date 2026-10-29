# Dense retrieval (`retrievers/dense.py`, `embedding.py`, `embedding_cache.py`, `vectorstores/`)

Embeds documents and queries with a sentence-embedding model and ranks by cosine similarity. Built from three swappable pieces: an **embedder** (text -> vectors), an **embedding cache** (so the slow step runs once) and a **vector store** (nearest-neighbour search).

    DenseRetriever(embedder, store, cache)
      index(corpus):  doc_to_text -> embed (batched, cached) -> store.build(ids, vectors)
      search(query):  embed query -> store.search(vector, k)

## Vector stores (`vectorstores/`)
| Choice | Decision | Why |
|---|---|---|
| Interface | `VectorStore` protocol: `build(ids, vectors)`, `search(vector, k)`, `__len__` | One interface for the exact store now and FAISS / Qdrant later; the dense retriever does not care which |
| Similarity | Inner product, higher is better; stores never normalize | With unit-length vectors this is cosine similarity. Normalizing is the caller's job, so every backend scores the same thing |
| Updates | `build` replaces everything; no incremental add/delete | Matches `Retriever.index()`. Real databases do support updates; we do not need them |
| Result size | At most `k`, fewer only if the store holds fewer | Approximate stores may miss true neighbours; that is measured, not part of the contract (see tests) |
| Safety | `build` validates first and copies the caller's array; a failed `build` leaves the old content | Changing the caller's array later cannot corrupt the index |
| Exact store | Brute-force matrix multiply, then `utils.top_k_indices` | The reference every approximate store is measured against. Ties keep insertion order |
| Registry | `@register_vector_store("name")`, built with `build_vector_store(name, params)` | Same pattern as retrievers and metrics; the config picks the store by name |

For ~5k vectors, brute force is exact and takes milliseconds. Indexes (HNSW, IVF) only pay off at hundreds of thousands of vectors, and they trade accuracy for speed. FAISS (flat and HNSW) and Qdrant (a server, via Docker) are available as stores too; all three, the index cache and the rebuild flags are described in [vectorstores.md](vectorstores.md).

## Embedder (`embedding.py`)
- `Embedder` protocol: `fingerprint`, `embed_documents(texts)` and `embed_query(text)`, returning unit-length float32 vectors. Tests use a fake, so they never load PyTorch.
- `SentenceTransformerEmbedder(model_name, query_prefix, document_prefix, batch_size, max_seq_length, device)`. sentence-transformers is imported lazily.
- Prefixes exist because some retrieval models expect `query: ` or an instruction in front of queries; forgetting one hurts quality without any error.
- The **fingerprint** (model, prefixes, effective length limit, normalization) goes into the cache key: two embedders that produce different vectors must never share one.

## Embedding cache (`embedding_cache.py`)
The public interface is one small class: `EmbeddingCache(cache_dir)` with `load(fingerprint, ids, texts)` (the cached vectors, or `None`) and `save(fingerprint, ids, texts, vectors)`. The key, the file name and the row-count check are private, so a caller cannot pass a key that does not match its documents.

| Choice | Decision | Why |
|---|---|---|
| Key | SHA-256 of the fingerprint and the ordered (id, text) pairs, length-prefixed | A different model, setting, document text or document order gets a different file, so stale vectors are never reused. Length-prefixing stops `("ab", "c")` from colliding with `("a", "bc")` |
| Independent of the store | The store is not in the key | Embeddings do not depend on how they are searched; switching exact -> FAISS reuses them |
| File | `cache/embeddings-<key>.npy` (gitignored), vectors only | IDs are part of the key and keep their order |
| Damage | An unreadable file or wrong row count counts as a miss, logged | A damaged cache costs a recomputation, never a wrong result |
| Writes | Temporary file, then `os.replace` | An interrupted run cannot leave a half-written cache file |

## `DenseRetriever`
- Parameters (all plain values, so a YAML config can set them): `model_name`, `store`, `store_params`, `query_prefix`, `document_prefix`, `batch_size`, `max_seq_length`, `use_cache`, `cache_dir`, `rebuild_index`, `rebuild_embeddings`; `embedder` can be injected for tests. With `use_cache` on and a store that can save its index (FAISS), a saved index is loaded instead of embedding and building; see [vectorstores.md](vectorstores.md).
- The model loads on the **first `index()`**, not at construction, so building a retriever from a config is cheap. The vector store is built immediately, so an unknown store or bad store parameters fail early.
- Embedder output is checked to be unit length (`ValueError` otherwise). An unnormalized embedder would silently rank by raw dot product. Queries are checked too: scaling a query would not change one ranking, but it would make scores incomparable across queries.
- Unlike BM25, every document has a similarity with every query, so a dense ranking always has `k` results (when the corpus is large enough).
- The text embedded is `doc_to_text` (title + ". " + text), the same string BM25 indexes, so the comparison is like for like.

## First results (SciFact test, 300 queries, `all-MiniLM-L6-v2`, exact store)
| | BM25 | Dense | 
|---|---|---|
| nDCG@10 | 0.643 | **0.648** |
| recall@10 | 0.765 | **0.788** |
| recall@100 | 0.869 | **0.925** |
| MRR@10 | **0.609** | 0.607 |

Embedding 5,183 documents took 129 s on CPU; with the cache, indexing takes about 6 s (mostly loading the model). One search takes about 14 ms. The cache file is 8 MB.

**The two retrievers fail on different queries**, which motivates the hybrid retriever. nDCG@10: dense better on 77 queries, BM25 on 67, tied on 156. recall@100: 28 queries found only by dense, 13 only by BM25, 9 by neither. Taking the better of the two per query would give a recall@100 ceiling of 0.968 (versus 0.925 and 0.869).

## Tests
- `tests/test_dense.py` (fake embedder): topical ranking, descending cosine scores, always `k` results, title included in the indexed text, a document is its own best match, re-indexing, errors; the cache is hit for the same corpus and missed for changed text, a different fingerprint, `use_cache=False`, or a damaged file; the model is not loaded at construction.
- `tests/test_embedding_cache.py` (through the public `load` / `save` only): round trip; a change of model, any ID, any text, the order or the number of documents is a miss; no ambiguity between an ID and its text; non-ASCII text; misses for wrong-shaped and unreadable files; `save` rejects vectors that do not match the documents; the directory is created only on the first save; no temporary file left.
- `tests/test_vectorstores.py`, `tests/test_qdrant_store.py`, `tests/test_index_cache.py`: the vector store contract, the Qdrant store and the index cache; see [vectorstores.md](vectorstores.md).
- `tests/test_embedding_real.py` (marked `slow`, excluded by default, run with `pytest -m slow`): the real model matches meaning without shared words, returns unit-length float32 vectors, and its fingerprint reacts to prefixes and length limit.

## Open items
- The ANN-vs-exact recall experiment (sweep the HNSW search effort on SciFact for accuracy, synthetic vectors for speed); see the open items in [vectorstores.md](vectorstores.md).
- Truncation: MiniLM cuts inputs at about 256 word pieces; count how many SciFact abstracts are affected.
- Try a retrieval-tuned model (e.g. BGE small, which needs a query prefix) as a config-only experiment.
- Batch search (many queries per call) would speed up evaluation for stores that support it.
- Hybrid retriever to combine the two rankings.
