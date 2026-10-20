# Data module (`src/rag_eval/data.py`)

Design choices for dataset download, loading and validation. Dataset: BEIR SciFact.

## Overview
- `download_dataset(name, data_dir)` fetches and extracts a BEIR dataset once and returns its directory.
- `load_corpus`, `load_queries`, `load_qrels` parse the three files into plain dicts.
- `Dataset.load(name, split, data_dir, on_error)` ties them together and checks that IDs line up.
- Helpers `_select_dataset_url`, `_stream_to_file`, `_download_and_extract_zip` are private.

## Download
| Choice | Decision | Why |
|---|---|---|
| Fetching | `requests` + `zipfile`, no `beir` package | Fewer dependencies, mechanics stay visible |
| Source | BEIR's own host (UKP, Darmstadt) | The Hugging Face URL I first tried returned 404 |
| Memory | Stream to disk in 1 MiB chunks, then extract from the file | Memory stays flat regardless of dataset size |
| Timeout | `timeout=30`, no retries | Fails loudly; the timeout applies per read, not to the whole download |
| Completeness | `.zip.tmp` marker created before any network call, removed after extraction | A failed or interrupted run is detectable and the next run re-downloads. Created early so even a connection error leaves it |
| Layout | The zip's top-level folder is stripped during extraction | Avoids `data/scifact/scifact/`; files land directly in `data/<name>/`. Assumes one wrapper folder, as BEIR zips have |
| Return value | `download_dataset` returns the dataset **directory** | One consistent contract for the loaders |

## Loaders
- **Corpus:** keeps only `title` and `text` (missing become `""`); `metadata` is dropped as retrieval doesn't use it.
- **IDs** stay strings everywhere; the files already store them as strings, so no casts.
- **Duplicate IDs:** last one wins, with a warning. Generating new IDs or copying labels to duplicates was rejected: qrels refer to the original ID, so renamed copies can never match, and mirroring labels would invent judgments and make results incomparable with published BEIR numbers.
- **Missing `_id`:** row skipped with a warning (open question: should this raise instead?).
- **Queries:** `load_queries` returns every query in the file (all splits); filtering belongs to `Dataset.load`.
- **Qrels:** header skipped; rows with score <= 0 dropped so every judged query has at least one relevant doc; malformed rows raise `ValueError` with the line number.
- All files opened with `encoding="utf-8"` (Windows default is a legacy codepage). Files are read line by line.

## `Dataset`
- A frozen dataclass: `name`, `split`, `corpus`, `queries`, `qrels`. Lives in `data.py`, not a separate module.
- `load` keeps only queries that have qrels in the requested split.
- **Consistency checks** are private and split into detection and policy:
  - `_find_problems()` only reports: qrels docs missing from the corpus, qrels queries missing from the queries.
  - `_without_problems()` returns a new `Dataset` with those removed; dropping cascades to queries left with no judgments. Nothing is mutated in place.
  - `_validated(on_error)` applies the policy: `"raise"` (default) throws one `ValueError` with counts and examples; `"drop"` cleans the data and logs how much was removed. An invalid `on_error` value raises even when the data is clean.
- `summary()` returns `docs`, `queries`, `judged_queries`, `judgments` for sanity checks and later MLflow logging.

## Pydantic
Tried on `_select_dataset_url` as an exercise and removed it: pointless overhead for a hardcoded dict. Better places later: experiment configs and structured LLM output (see `NOTES.md`).

## Tests (`tests/test_data.py`)
Network access is faked by patching `_stream_to_file`; Dataset and loader tests use tiny in-memory objects or files in `tmp_path`. Not yet run in this environment (pytest was not installed).

## Open items
- Missing `_id`: skip vs raise.
- Not yet tested: missing `_id`, malformed JSON line, duplicate query IDs.
- Dropped counts are only logged, not stored on the `Dataset` (so they won't reach MLflow automatically).
- Compare `Dataset.load("scifact").summary()` against the real files once, by counting lines independently.
