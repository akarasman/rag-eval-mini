"""Dataset loading.

Responsible for getting a BEIR dataset onto disk (download once) and exposing
it as plain Python structures: corpus, queries and relevance judgments (qrels).
Nothing in here knows about retrieval or metrics.
"""
import csv
import logging
import requests
import zipfile
import json
from dataclasses import dataclass, replace
from typing import Dict, Literal
from pathlib import Path

# Offload to a config later
DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB
DOWNLOAD_TIMEOUT = 30  # seconds

# Plain-dict representations; change if you prefer dataclasses.
Corpus = Dict[str, Dict[str, str]]       # doc_id -> {"title": ..., "text": ...}
Queries = Dict[str, str]                 # query_id -> query text
Qrels = Dict[str, Dict[str, int]]        # query_id -> {doc_id: relevance grade}
OnError = Literal["raise", "drop"]       # what Dataset.load does with inconsistent IDs

URLS = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
}

def _select_dataset_url(name: str) -> str:
    """Return the download URL for a BEIR dataset.

    Args:
        name: Name of the dataset (e.g. "scifact").

    Returns:
        The URL to download the dataset from.
    """
    if name not in URLS:
        raise ValueError(f"Unknown dataset {name}. Supported: {list(URLS.keys())}")
    return URLS[name]

def _stream_to_file(url: str, dest_path: Path,
                   chunk_size: int = DOWNLOAD_CHUNK_SIZE,
                   timeout: int = DOWNLOAD_TIMEOUT) -> None:
    """Stream the response body of a GET request to dest_path in chunks.

    Args:
        url: The URL to download.
        dest_path: File to write to (overwritten if it exists).
        chunk_size: Bytes per chunk (default: 1 MiB).
    """
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)


def _download_and_extract_zip(url: str, dest_dir: Path) -> Path:
    """Download a zip file from a URL and extract it to dest_dir.
    
    Args:
        url: The URL to download the zip file from.
        dest_dir: The directory to extract the contents of the zip file to.
    
    Returns:
        The path to the directory where the zip file was extracted.
    """

    dest_dir.mkdir(parents=True, exist_ok=True)

    temp_zip_path = dest_dir / ".zip.tmp"
    temp_zip_path.touch() 
    
    logging.info(f"Downloading dataset from {url}...")
    _stream_to_file(url, temp_zip_path)

    logging.info(f"Extracting dataset to {dest_dir}...")
    with zipfile.ZipFile(temp_zip_path) as z:
        # BEIR zips wrap everything in one top-level folder (e.g. "scifact/").
        # Strip it so files land directly in dest_dir, not dest_dir/scifact/.
        for member in z.infolist():
            parts = member.filename.split("/")
            if len(parts) < 2 or not any(parts[1:]):
                continue  # the top-level folder entry itself
            member.filename = "/".join(parts[1:])
            z.extract(member, dest_dir)

    temp_zip_path.unlink()

    return dest_dir

def download_dataset(name: str, data_dir: Path = Path("data")) -> Path:
    """Download and extract a BEIR dataset (e.g. "scifact") if not already present.
    
    Args:
        name: Name of the dataset (e.g. "scifact").
        data_dir: Directory to store the dataset in (default: "data").
    
    Returns:
        The path to the dataset directory.
    """

    dataset_url = _select_dataset_url(name)

    if ( (data_dir / name).exists() and 
         (data_dir / name).is_dir() and
         not (data_dir / name / ".zip.tmp").exists() ):
        logging.info(f"Dataset {name} already exists at {data_dir / name}. Skipping download.")
        return data_dir / name 
    
    return _download_and_extract_zip(dataset_url, data_dir / name)

def load_corpus(dataset_dir: Path) -> Corpus:
    """Read corpus.jsonl.

    Only `title` and `text` are kept (missing ones become ""); `metadata` is dropped.
    If an ID appears more than once, the last occurrence wins.

    Args:
        dataset_dir: Path to the dataset directory.
    """
    corpus: Corpus = {}
    with open(dataset_dir / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:          # skip blank lines (e.g. a trailing newline)
                continue
            jsonrow = json.loads(line)
            
            if "_id" not in jsonrow:
                logging.warning(f"Skipping line {line_num} in corpus.jsonl: missing '_id' field")
                continue

            if jsonrow["_id"] in corpus:
                logging.warning(f"Duplicate document ID {jsonrow['_id']} at line {line_num} in corpus.jsonl. Overwriting previous entry.")

            corpus[jsonrow["_id"]] = {
                "title": jsonrow.get("title", ""),
                "text": jsonrow.get("text", ""),
            }

    return corpus


def load_queries(dataset_dir: Path) -> Queries:
    """Read queries.jsonl.
    
    Args:
        dataset_dir: Path to the dataset directory.

    Returns:
        All queries in the file (all splits); filtering to a split's qrels happens in Dataset.
    """
    queries: Queries = {}
    with open(dataset_dir / "queries.jsonl", "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            jsonrow = json.loads(line)

            if "_id" not in jsonrow:
                logging.warning(f"Skipping line {line_num} in queries.jsonl: missing '_id' field")
                continue

            if jsonrow["_id"] in queries:
                logging.warning(f"Duplicate query ID {jsonrow['_id']} at line {line_num} in queries.jsonl. Overwriting previous entry.")

            queries[jsonrow["_id"]] = jsonrow.get("text", "")

    return queries


def load_qrels(dataset_dir: Path, split: str = "test") -> Qrels:
    """Read qrels/<split>.tsv.

    Rows with a score <= 0 are dropped, so every query in the result has at least
    one relevant doc.

    Args:
        dataset_dir: Path to the dataset directory.
        split: Which split's qrels file to read (e.g. "test", "train").
    """
    qrels: Qrels = {}
    with open(dataset_dir / "qrels" / f"{split}.tsv", "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header: query-id, corpus-id, score
        for line_num, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(f"{split}.tsv line {line_num}: expected 3 tab-separated columns, got {len(row)}")
            query_id, doc_id, score = row[0], row[1], int(row[2])
            if score <= 0:
                continue
            qrels.setdefault(query_id, {})[doc_id] = score

    return qrels


@dataclass(frozen=True)
class Dataset:
    """Corpus, queries and qrels for one dataset + split, checked for consistency."""
    name: str
    split: str
    corpus: Corpus
    queries: Queries
    qrels: Qrels

    @classmethod
    def load(cls, name: str, split: str = "test", data_dir: Path = Path("data"),
             on_error: OnError = "raise") -> "Dataset":
        """Download (if needed), load all three parts and check they agree.

        Queries are restricted to those with qrels in the given split, since
        queries.jsonl covers every split.

        Args:
            on_error: What to do if qrels reference docs missing from the corpus or
                queries missing from the query set. "raise" fails loudly; "drop"
                removes the offending entries (and logs how many).
        """
        dataset_dir = download_dataset(name, data_dir)
        corpus = load_corpus(dataset_dir)
        qrels = load_qrels(dataset_dir, split)
        queries = {qid: text for qid, text in load_queries(dataset_dir).items() if qid in qrels}

        dataset = cls(name=name, split=split, corpus=corpus, queries=queries, qrels=qrels)
        return dataset._validated(on_error)

    def _find_problems(self) -> dict[str, list]:
        """Detect inconsistencies without changing anything."""
        missing_docs = [
            (qid, doc_id)
            for qid, docs in self.qrels.items()
            for doc_id in docs
            if doc_id not in self.corpus
        ]
        missing_queries = [qid for qid in self.qrels if qid not in self.queries]
        return {"qrels_docs_not_in_corpus": missing_docs, "qrels_queries_not_in_queries": missing_queries}

    def _without_problems(self) -> "Dataset":
        """Return a new Dataset with inconsistent qrels removed. Does not mutate self."""
        qrels: Qrels = {}
        for qid, docs in self.qrels.items():
            if qid not in self.queries:
                continue
            kept = {doc_id: grade for doc_id, grade in docs.items() if doc_id in self.corpus}
            if kept:  # a query left with no judgments can't be scored
                qrels[qid] = kept
        queries = {qid: text for qid, text in self.queries.items() if qid in qrels}
        return replace(self, qrels=qrels, queries=queries)

    def _validated(self, on_error: OnError) -> "Dataset":
        """Apply the on_error policy; returns self if nothing is wrong."""
        if on_error not in ("raise", "drop"):
            raise ValueError(f"on_error must be 'raise' or 'drop', got {on_error!r}")

        problems = self._find_problems()
        if not any(problems.values()):
            return self

        summary = "; ".join(f"{kind}: {len(items)} (e.g. {items[:3]})" for kind, items in problems.items() if items)
        if on_error == "raise":
            raise ValueError(f"Inconsistent dataset '{self.name}' ({self.split}): {summary}")

        cleaned = self._without_problems()
        logging.warning(
            f"Dataset '{self.name}' ({self.split}) had inconsistencies ({summary}). "
            f"Dropped: {len(self.qrels) - len(cleaned.qrels)} queries from qrels, "
            f"{sum(map(len, self.qrels.values())) - sum(map(len, cleaned.qrels.values()))} judgments."
        )
        return cleaned

    def summary(self) -> Dict[str, int]:
        """Counts for logging/sanity checks (docs, queries, judged queries, judgments)."""
        return {
            "docs": len(self.corpus),
            "queries": len(self.queries),
            "judged_queries": len(self.qrels),
            "judgments": sum(len(docs) for docs in self.qrels.values()),
        }
