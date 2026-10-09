"""Dataset loading.

Responsible for getting a BEIR dataset onto disk (download once) and exposing
it as plain Python structures: corpus, queries and relevance judgments (qrels).
Nothing in here knows about retrieval or metrics.
"""
from pathlib import Path

# Plain-dict representations; change if you prefer dataclasses.
Corpus = dict[str, dict[str, str]]       # doc_id -> {"title": ..., "text": ...}
Queries = dict[str, str]                 # query_id -> query text
Qrels = dict[str, dict[str, int]]        # query_id -> {doc_id: relevance grade}


def download_dataset(name: str, data_dir: Path = Path("data")) -> Path:
    """Download and extract a BEIR dataset (e.g. "scifact") if not already present."""
    # TODO: skip if already on disk; return the dataset folder
    # TODO: decide how to fetch (beir package vs. manual download)
    raise NotImplementedError


def load_corpus(dataset_dir: Path) -> Corpus:
    """Read corpus.jsonl."""
    # TODO: parse one JSON object per line; keep ids as strings
    raise NotImplementedError


def load_queries(dataset_dir: Path) -> Queries:
    """Read queries.jsonl."""
    # TODO: parse; decide whether to keep only queries that appear in the split's qrels
    raise NotImplementedError


def load_qrels(dataset_dir: Path, split: str = "test") -> Qrels:
    """Read qrels/<split>.tsv."""
    # TODO: skip header row; parse relevance score as int
    # TODO: decide what to do with score-0 rows (explicitly non-relevant)
    raise NotImplementedError
