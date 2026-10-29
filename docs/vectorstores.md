# Vector stores (`vectorstores/`, `index_cache.py`)

A vector store holds document vectors and answers "which stored vectors are most similar to this one?". It knows nothing about text or models: the dense retriever embeds text and hands vectors over, so the same store works with any embedder. Three backends sit behind one interface, so they can be compared on the same data.

| Store | Name | What it is | Needs |
|---|---|---|---|
| `ExactVectorStore` | `exact` | Brute-force matrix multiply. Always exact; the reference everything else is measured against | numpy |
| `FaissVectorStore` | `faiss` | A library in our process. `flat` (exact) or `hnsw` (approximate graph search) | `faiss-cpu` |
| `QdrantVectorStore` | `qdrant` | A client for a Qdrant server (HNSW, stored and served by the server) | `qdrant-client` and a running server |

## The contract (`vectorstores/base.py`)
Every store follows the `VectorStore` protocol: `build(ids, vectors)`, `search(vector, k)`, `__len__`.

| Choice | Decision | Why |
|---|---|---|
| Similarity | Inner product, higher is better; stores never normalize | With unit-length vectors this is cosine similarity. Normalizing is the caller's job (the dense retriever checks it), so every backend scores the same thing |
| Updates | `build` replaces everything; no incremental add/delete | Matches `Retriever.index()` |
| Result size | At most `k`, fewer only if the store holds fewer | Approximate stores may miss true neighbours; that is measured, not part of the contract |
| Safety | Input is validated before anything is replaced (`validate_build_input`, `validate_search_input`, shared by all stores) | A failed `build` leaves the old content; every store rejects bad input the same way |
| Registry | `@register_vector_store("name")`, built with `build_vector_store(name, params)` | The config picks the store and its parameters by name |

## FAISS (`faiss_store.py`)
| Choice | Decision | Why |
|---|---|---|
| Index types | `IndexFlatIP` ("flat") and `IndexHNSWFlat` with inner product ("hnsw") | Flat is an exact cross-check; HNSW is the approximate index whose recall we want to measure |
| Parameters | `index_type`, `hnsw_m` (default 32), `ef_search` (default 64) | `hnsw_m` is the links per node (accuracy vs size and build time); `ef_search` is the candidates examined per query (accuracy vs speed). Both are ignored for flat |
| IDs | FAISS knows only row positions, so the store keeps the `_ids` list and maps back | Losing or reordering it would return wrong documents with no error |
| Short results | FAISS pads with position `-1` when `k` exceeds the size; those are dropped | The contract says "fewer than `k` only if the store is smaller" |
| CPU only | `faiss-cpu`; no GPU option | This machine has no usable GPU (CPU-only PyTorch), the FAISS GPU build is, as far as we know, not offered for Windows (not verified), and at ~5k vectors search time is not the bottleneck. Embedding is, and that is the embedder's `device` setting |

## Qdrant (`qdrant_store.py`)
Qdrant is a database server, not a library: the index lives in another process (a Docker container) and our code talks to it over REST through `qdrant-client`.

| Choice | Decision | Why |
|---|---|---|
| IDs | The row position is the point id; the real id is stored in the payload (`doc_id`) and read back with every result | Qdrant point ids must be unsigned integers or UUIDs, and ours are arbitrary strings |
| Metric | `Distance.DOT` | Matches the contract. Qdrant's `Cosine` rescales vectors, so non-unit vectors would score differently from the other stores (a test pins this) |
| `build` | Validate, then delete and recreate the collection, then upload with `wait=True` | The upload is asynchronous by default, so without `wait=True` a search right after `build` could miss points |
| Parameters | `collection`, `url`, `hnsw_m` (16), `ef_construct` (100), `ef_search` (`None` = server default), `timeout` | Qdrant's own defaults; `ef_search` is sent as `hnsw_ef` with each query |
| Construction | Does not contact the server (`check_compatibility=False`) | Building a retriever from a config stays cheap and offline |
| Errors | An unreachable server raises `RuntimeError` naming the URL | One clear message instead of a connection traceback |
| Persistence | Not a `PersistableVectorStore` | The server keeps the collection itself |
| Collection ownership | Each store owns its collection name; `build` deletes it | Give every experiment or model its own name |

**A failed upload** (the server goes away mid-way) leaves the collection incomplete. The store then refuses to search until the next `build`. A validation failure still leaves the old content untouched.

### Running the server
    docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:v1.20.0

REST is on 6333 (dashboard at `http://localhost:6333/dashboard`), gRPC on 6334. The version matches the pinned `qdrant-client`. The data lives in the Docker volume `qdrant_storage`, so it survives `docker restart qdrant`. The container does not start by itself after a reboot: `docker start qdrant`.

### Using it
    retriever:
      name: dense
      params:
        model_name: BAAI/bge-small-en-v1.5
        store: qdrant
        store_params: {collection: scifact-bge, hnsw_m: 16, ef_search: 64}

A collection that was filled by hand (for example from the dashboard) does not record which model produced its vectors. Querying it with another model gives no error, only meaningless results, because the dimensions match. Putting the model in the collection name avoids that. Through `DenseRetriever` this cannot happen, since `build` recreates the collection from the current embedder on every `index()`.

## Index cache (`index_cache.py`)
Building an HNSW index is wasted work when the embeddings are already cached, so a store that can write its index to a file is cached next to the embedding cache (same folder).

| Choice | Decision | Why |
|---|---|---|
| Optional ability | `PersistableVectorStore` protocol with `save(path)` / `load(path)`; only FAISS implements it | Qdrant persists on the server and the exact store builds instantly. The retriever checks `isinstance` |
| Owner | The retriever decides when to load, build and save, and where the file goes; the store only does the serialization | The store knows nothing about corpora, keys or folders |
| Key | SHA-256 of the corpus key (`embedding_key`: embedder fingerprint, ids, texts), the store name and the store parameters as sorted JSON | Changing a store parameter rebuilds the index without touching the embedding cache; the order of the parameters does not matter |
| File | `cache/index-<key>.npz`: the serialized index and the id list together | Ids and index can never get out of step |
| On a hit | Nothing is embedded and nothing is built | |
| Safe loading | Any unreadable or inconsistent file raises `ValueError` from `load`, and the cache turns it into a logged miss; `load` applies the store's own `ef_search` and keeps the previous content on failure | A damaged cache costs a rebuild, never a wrong result |
| Writes | Temporary file, then `os.replace` | An interrupted run cannot leave a half-written file |

Consequences of hashing the parameters: `ef_search` is in `store_params`, so changing it also rebuilds (cheap, a few seconds on SciFact). `{}` and an explicit `{"index_type": "flat"}` mean the same but hash differently, which costs one extra rebuild. Old index files are never deleted.

### Forcing a rebuild
`DenseRetriever(rebuild_index=True)` ignores the saved index and replaces it, reusing the cached embeddings. `rebuild_embeddings=True` re-embeds and replaces the embeddings, and also rebuilds the index, which depends on them. Both still write the result, so the next normal run uses the fresh files. With `use_cache=False` they do nothing, and neither cache is read or written.

## Tests
- `tests/test_vectorstores.py`: contract tests run against every store in `STORE_FACTORIES` plus the server-backed ones in `SERVER_STORES` (Qdrant), through one parametrized fixture; plus exact-store-only and FAISS-only tests (flat equals exact, HNSW recall against exact, the HNSW settings really build an HNSW index) and the registry.
- `tests/test_qdrant_store.py`: parameter validation and a clear error for an unreachable server (no server needed); ids that are not valid Qdrant ids come back unchanged, scores are inner products for non-unit vectors, recall against exact, a second build with a new dimension, the HNSW settings and `ef_search` reach the server (server needed).
- `tests/test_index_cache.py`: FAISS `save` / `load` round trip (flat and HNSW), bad files rejected with the previous content kept, `IndexCache` hits, misses, damaged files and atomic writes, and the retriever wiring (a second run neither embeds nor builds, changing a store parameter rebuilds only the index, the rebuild flags).
- `tests/conftest.py`: the Qdrant fixtures. Each test store gets a unique collection that is deleted afterwards. **When no server is running those tests are skipped, not failed**, so a green run with a high skip count means the server was down.

## Open items
- ANN-vs-exact recall experiment: sweep `ef_search` (FAISS) and `hnsw_ef` (Qdrant) against the exact store on SciFact for accuracy, and synthetic vectors (500k to 1M) for speed. SciFact (5k vectors) is too small for HNSW to show a speed advantage.
- Explored by hand in the Qdrant dashboard but not recorded: payload filtering (Qdrant filters while walking the graph; FAISS can only search first), exact search with `exact=true` as ground truth, a deliberately low `hnsw_ef`. No numbers were kept.
- Not tried: scalar quantization, cosine vs dot on unnormalized vectors, persistence across `docker rm`.
- The Qdrant payload holds only `doc_id`; adding the title would make dashboard results readable.
- Batch search (many queries per call) would speed up evaluation for stores that support it.
