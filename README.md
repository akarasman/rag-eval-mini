# rag-eval-mini

A minimal RAG pipeline in plain Python, focused on evaluating retrieval quality. It compares BM25, dense (sentence-embedding) and hybrid retrievers on a public benchmark, logs every run to MLflow, and adds a small generation step: an LLM answers a question from the retrieved passages and cites them.

Built as a learning project. Libraries are used where a trusted implementation matters (BM25, FAISS, Qdrant, sentence-transformers); the pipeline around them is written out so each step stays visible.

## Dataset

[BEIR](https://github.com/beir-cellar/beir) **SciFact**: 5,183 scientific abstracts, searched with claims that an abstract supports or refutes. Retrieval is judged on the **test split (300 queries)**. The 809 **train** queries are used only to choose settings (the hybrid weights), never to report results. The data is downloaded to `data/` on first use.

## Setup

    pip install -r requirements.txt
    pip install -e .

Python 3.10 or newer. Dense retrieval downloads its embedding model from Hugging Face on first use and caches the document vectors in `cache/`, so the slow step runs once.

## Run the experiments

    python experiments/run_experiments.py experiments/configs/bm25_baseline.yaml

Give several config files to run them one after another. Each run is logged to MLflow with its parameters, mean metrics, timings, the config, per-query scores, and the best and worst queries. Browse them with:

    mlflow ui

To choose the weights of a weighted hybrid on the train queries:

    python experiments/tune_weights.py experiments/configs/hybrid_weighted_bm25_bge.yaml

Experiments are described by YAML files in `experiments/configs/`: the dataset, the retriever (and its parameters), the metrics and the cutoffs. Retrievers, metrics, fusion strategies, vector stores and LLM backends are all chosen by name from registries, so adding one means writing a class and registering it.

## Results

SciFact test, 300 queries, exact vector store. Recall@100 shows how many relevant documents a later stage (such as a reranker) could still find.

| Retriever | nDCG@10 | Recall@10 | Recall@100 |
|---|---|---|---|
| BM25 | 0.643 | 0.764 | 0.869 |
| Dense: MiniLM (`all-MiniLM-L6-v2`) | 0.648 | 0.788 | 0.925 |
| Dense: BGE-small (`bge-small-en-v1.5`) | 0.706 | 0.830 | 0.943 |
| Hybrid RRF: BM25 + MiniLM | 0.676 | 0.803 | 0.955 |
| Hybrid RRF: MiniLM + BGE-small | 0.696 | 0.836 | 0.947 |
| Hybrid RRF: BM25 + BGE-small | 0.693 | 0.808 | 0.965 |
| **Hybrid weighted: 0.3 BM25 + 0.7 BGE-small** | **0.716** | **0.842** | **0.968** |

Numbers come from the MLflow runs; the full MRR and other cutoffs are logged there. Details: [docs/hybrid-retrieval.md](docs/hybrid-retrieval.md).

### What the results show

- **A good dense model beats BM25 by about 6 points nDCG@10.** The weaker dense model (MiniLM) roughly ties it on nDCG but finds more relevant documents deeper in the list (recall@100 0.925 against 0.869).
- **Combining retrievers helps most when they are about equally strong.** BM25 + MiniLM gains about 3 points over either alone. Adding BM25 to the much stronger BGE with an equal vote (RRF) slightly lowers nDCG@10 (0.693 against 0.706), because the weaker retriever's mistakes count as much as the stronger one's.
- **Weighting fixes that.** With weights picked on the train queries (BM25 0.3, BGE 0.7; the train curve is flat between 0.2 and 0.4) the hybrid reaches 0.716 and the best recall@100. The gain over BGE alone is about one point. With 300 test queries it could be within noise, and **no significance test has been run**.
- **BM25 here (0.643) is below the roughly 0.67 usually quoted for SciFact.** Likely causes are the tokenizer (no stopword removal, stemming off), the BM25 variant and its defaults. This has not been investigated.
- **Indexing cost differs a lot.** On this machine BM25 indexes in about 1 second; embedding the corpus took about 135 s for MiniLM and about 460 s for BGE-small on the CPU, which is why vectors are cached.
- **Vector stores.** The exact store, FAISS (flat and HNSW) and Qdrant all implement the same interface. At about 5,000 vectors brute force is already fast, so SciFact cannot show the speed advantage of approximate indexes. The ANN-versus-exact recall experiment is still to do. See [docs/vectorstores.md](docs/vectorstores.md).

## Ask a question

Retrieve the best passages for a question and let an LLM answer from them, citing its sources, or say that the passages do not answer it.

    python -m rag_eval.ask --config configs/ask/ollama.yaml "your question"

Pick a backend by choosing its config (each file says what it needs):

| Config | Runs | Needs |
|---|---|---|
| `configs/ask/ollama.yaml` | A local Ollama server | The server running and a model pulled (`ollama pull <model>`) |
| `configs/ask/transformers.yaml` | A chat model inside this process (CPU) | A model id from Hugging Face; it is downloaded on first use |
| `configs/ask/openai_compatible.yaml` | A hosted API with an OpenAI-style endpoint | `base_url`, a model name, and an environment variable holding your API key (the config names the variable, never the key) |

The retriever is set in the same file: inline, or `retriever_from:` an experiment config. Useful options: `--k` (passages sent to the model), `--json` (the full result, including the model's raw output), `-v` (progress, and full tracebacks on errors).

How to read the `Status:` line:

| Status | Meaning |
|---|---|
| `answered` | An answer with at least one valid citation; the cited sources are listed |
| `abstained` | The model says the retrieved passages do not answer the question |
| `no_passages` | Nothing was retrieved, so the model was not asked |
| `ungrounded` | The model gave an answer but no valid citation; treat it as unverified |
| `invalid_output` | The model never returned a usable JSON reply |

A citation only shows that the model pointed at a passage it was given; nothing checks that the passage supports the answer. Design notes: [docs/generation-module.md](docs/generation-module.md).

## Limitations

- **Generation is not evaluated.** Retrieval has numbers; answer quality does not. Small local models often return `ungrounded` or `invalid_output`, and the hosted-API backend has not been tried against a real API.
- Results are for one dataset, one split, one run per setup (no seeds or repeats), and without significance tests.
- Nothing is tuned for the BM25 baseline (`k1`, `b`, stemming), so the gap to the published number is open.

## Tests

    python -m pytest

The default run uses fakes and tiny corpora, so it needs no model download. Tests that need a Qdrant server are skipped when none is running (a high skip count means the server was down). Tests with the real embedding model are marked `slow`: `python -m pytest -m slow`.

## Project layout

| Path | Contents |
|---|---|
| `src/rag_eval/` | Data loading, text processing, retrievers, vector stores, metrics, evaluation, MLflow tracking, LLM backends, generation, and the `ask` command |
| `experiments/` | Experiment configs, the experiment runner, the weight tuner |
| `configs/ask/` | Configs for the `ask` command |
| `docs/` | Design notes for each module: choices and why, results, open items |
| `tests/` | Unit and contract tests |
