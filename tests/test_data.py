"""Tests for data loading. The network is faked (_stream_to_file is patched); nothing is downloaded.

Dataset tests build small instances directly or write tiny files to tmp_path.
TODO: more loader edge cases (missing _id, malformed JSON line, queries.jsonl duplicates).
"""
import io
import zipfile

import pytest
import requests

from rag_eval import data


def test_select_dataset_url():
    from rag_eval.data import _select_dataset_url

    for name in data.URLS.keys():
        url = _select_dataset_url(name)
        assert url.startswith("https://") and url.endswith(".zip")

def test_select_dataset_url_invalid():
    from rag_eval.data import _select_dataset_url

    try:
        _select_dataset_url("invalid_dataset_name")
    except ValueError as e:
        assert "Unknown dataset" in str(e)
    else:
        assert False, "Expected ValueError for invalid dataset name"


# --- download / extraction (network faked) ---------------------------------

def _fake_zip_bytes() -> bytes:
    """A tiny zip with the same layout as the real BEIR one (top-level folder included)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("scifact/corpus.jsonl", '{"_id": "1", "title": "t", "text": "x"}\n')
        z.writestr("scifact/queries.jsonl", '{"_id": "1", "text": "q"}\n')
        z.writestr("scifact/qrels/test.tsv", "query-id\tcorpus-id\tscore\n1\t1\t1\n")
    return buf.getvalue()


@pytest.fixture
def fake_download(monkeypatch):
    """Patch stream_to_file to write the fake zip; returns a list recording each call's URL."""
    calls = []

    def fake(url, dest_path, chunk_size=1 << 20):
        calls.append(url)
        dest_path.write_bytes(_fake_zip_bytes())

    monkeypatch.setattr(data, "_stream_to_file", fake)
    return calls


def _marker(tmp_path):
    return tmp_path / "scifact" / ".zip.tmp"


def test_download_dataset_extracts_and_removes_marker(tmp_path, fake_download):
    result = data.download_dataset("scifact", data_dir=tmp_path)

    assert len(fake_download) == 1, f"Expected 1 download, got {len(fake_download)}"
    assert result.exists(), f"Expected result path {result} to exist"
    assert result.is_dir(), f"Expected result path {result} to be a directory"
    assert (tmp_path / "scifact" / "corpus.jsonl").is_file(), f"Expected corpus file not found at {tmp_path / 'scifact' / 'corpus.jsonl'}"
    assert not _marker(tmp_path).exists()


def test_failed_download_leaves_marker_and_next_run_retries(tmp_path, monkeypatch):
    
    # Patch stream_to_file to raise an exception on the first call, then succeed on the second call.
    def failing(url, dest_path, chunk_size=1 << 20):
        raise requests.exceptions.ConnectionError("boom")
    monkeypatch.setattr(data, "_stream_to_file", failing)
    with pytest.raises(requests.exceptions.ConnectionError):
        data.download_dataset("scifact", data_dir=tmp_path)
    assert _marker(tmp_path).exists()

    # Now patch stream_to_file to succeed, and call download_dataset again.
    def working(url, dest_path, chunk_size=1 << 20):
        dest_path.write_bytes(_fake_zip_bytes())
    monkeypatch.setattr(data, "_stream_to_file", working)
    data.download_dataset("scifact", data_dir=tmp_path)  # must not skip the half-done dataset
    assert (tmp_path / "scifact" / "corpus.jsonl").is_file()
    assert not _marker(tmp_path).exists()


def test_complete_dataset_is_not_downloaded_again(tmp_path, fake_download):
    dataset_dir = tmp_path / "scifact"
    dataset_dir.mkdir()
    (dataset_dir / "corpus.jsonl").write_text("")

    data.download_dataset("scifact", data_dir=tmp_path)

    assert fake_download == []


def test_dataset_with_marker_is_downloaded_again(tmp_path, fake_download):
    dataset_dir = tmp_path / "scifact"
    dataset_dir.mkdir()
    _marker(tmp_path).touch()

    data.download_dataset("scifact", data_dir=tmp_path)

    assert len(fake_download) == 1
    assert not _marker(tmp_path).exists()


# --- Dataset (built directly, no files) -------------------------------------

def _make_dataset(**overrides) -> data.Dataset:
    """Small consistent dataset: 2 docs, 2 queries, 3 judgments."""
    fields = dict(
        name="tiny",
        split="test",
        corpus={"d1": {"title": "t1", "text": "x1"}, "d2": {"title": "t2", "text": "x2"}},
        queries={"q1": "first", "q2": "second"},
        qrels={"q1": {"d1": 1, "d2": 1}, "q2": {"d2": 1}},
    )
    fields.update(overrides)
    return data.Dataset(**fields)


def test_clean_dataset_is_returned_unchanged_in_both_modes():
    ds = _make_dataset()

    assert ds._validated("raise") is ds
    assert ds._validated("drop") is ds


def test_invalid_on_error_is_rejected_even_for_clean_data():
    with pytest.raises(ValueError, match="on_error"):
        _make_dataset()._validated("dorp")


def test_raise_mode_reports_qrels_doc_missing_from_corpus():
    ds = _make_dataset(qrels={"q1": {"d1": 1, "d9": 1}, "q2": {"d2": 1}})

    with pytest.raises(ValueError, match="qrels_docs_not_in_corpus"):
        ds._validated("raise")


def test_raise_mode_reports_qrels_query_missing_from_queries():
    ds = _make_dataset(queries={"q1": "first"})

    with pytest.raises(ValueError, match="qrels_queries_not_in_queries"):
        ds._validated("raise")


def test_drop_mode_removes_only_the_bad_judgment():
    ds = _make_dataset(qrels={"q1": {"d1": 1, "d9": 1}, "q2": {"d2": 1}})

    cleaned = ds._validated("drop")

    assert cleaned.qrels == {"q1": {"d1": 1}, "q2": {"d2": 1}}
    assert set(cleaned.queries) == {"q1", "q2"}


def test_drop_mode_cascades_to_queries_left_without_judgments():
    ds = _make_dataset(qrels={"q1": {"d1": 1}, "q2": {"d9": 1}})  # q2 only points at a missing doc

    cleaned = ds._validated("drop")

    assert "q2" not in cleaned.qrels
    assert "q2" not in cleaned.queries


def test_drop_mode_removes_qrels_for_unknown_queries():
    ds = _make_dataset(queries={"q1": "first"})  # q2 has qrels but no query text

    cleaned = ds._validated("drop")

    assert set(cleaned.qrels) == {"q1"}
    assert set(cleaned.queries) == {"q1"}


def test_drop_mode_does_not_mutate_the_original():
    ds = _make_dataset(qrels={"q1": {"d1": 1, "d9": 1}, "q2": {"d2": 1}})

    ds._validated("drop")

    assert ds.qrels == {"q1": {"d1": 1, "d9": 1}, "q2": {"d2": 1}}
    assert set(ds.queries) == {"q1", "q2"}


def test_summary_counts():
    assert _make_dataset().summary() == {
        "docs": 2,
        "queries": 2,
        "judged_queries": 2,
        "judgments": 3,
    }


# --- Dataset.load / loaders (tiny files on disk, download faked) ------------

@pytest.fixture
def tiny_dataset_dir(tmp_path, monkeypatch):
    """Write a tiny BEIR-style dataset and make download_dataset return it."""
    root = tmp_path / "tiny"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        '{"_id": "1", "title": "Ünïcode título", "text": "alpha", "metadata": {"k": "v"}}\n'
        '{"_id": "2", "text": "beta"}\n'
        "\n",  # trailing blank line
        encoding="utf-8",
    )
    (root / "queries.jsonl").write_text(
        '{"_id": "10", "text": "¿qué?"}\n'
        '{"_id": "11", "text": "second"}\n'
        '{"_id": "12", "text": "only in train"}\n',
        encoding="utf-8",
    )
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n"
        "10\t1\t1\n"
        "11\t2\t1\n"
        "11\t1\t0\n",  # explicitly non-relevant
        encoding="utf-8",
    )
    (root / "qrels" / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\n12\t1\t1\n", encoding="utf-8"
    )
    monkeypatch.setattr(data, "download_dataset", lambda name, data_dir: root)
    return root


def test_load_builds_dataset_for_the_requested_split(tiny_dataset_dir):
    ds = data.Dataset.load("tiny", split="test")

    assert ds.name == "tiny" and ds.split == "test"
    assert set(ds.queries) == {"10", "11"}  # "12" belongs to train
    assert ds.qrels == {"10": {"1": 1}, "11": {"2": 1}}  # header skipped, score-0 row dropped


def test_load_keeps_ids_as_strings_and_text_as_utf8(tiny_dataset_dir):
    ds = data.Dataset.load("tiny")

    assert all(isinstance(k, str) for k in ds.corpus)
    assert ds.corpus["1"]["title"] == "Ünïcode título"
    assert ds.queries["10"] == "¿qué?"


def test_load_keeps_only_title_and_text_and_defaults_missing_ones(tiny_dataset_dir):
    ds = data.Dataset.load("tiny")

    assert ds.corpus["1"] == {"title": "Ünïcode título", "text": "alpha"}  # metadata dropped
    assert ds.corpus["2"] == {"title": "", "text": "beta"}


def test_load_summary_matches_files(tiny_dataset_dir):
    assert data.Dataset.load("tiny").summary() == {
        "docs": 2, "queries": 2, "judged_queries": 2, "judgments": 2,
    }


def test_load_raises_when_qrels_point_at_a_missing_doc(tiny_dataset_dir):
    (tiny_dataset_dir / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n10\t999\t1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="qrels_docs_not_in_corpus"):
        data.Dataset.load("tiny")


def test_load_can_drop_instead_of_raising(tiny_dataset_dir):
    (tiny_dataset_dir / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n10\t999\t1\n11\t2\t1\n", encoding="utf-8"
    )

    ds = data.Dataset.load("tiny", on_error="drop")

    assert ds.qrels == {"11": {"2": 1}}
    assert set(ds.queries) == {"11"}


def test_load_corpus_last_duplicate_wins(tmp_path):
    (tmp_path / "corpus.jsonl").write_text(
        '{"_id": "1", "title": "old", "text": "a"}\n'
        '{"_id": "1", "title": "new", "text": "b"}\n',
        encoding="utf-8",
    )

    assert data.load_corpus(tmp_path) == {"1": {"title": "new", "text": "b"}}


def test_load_qrels_malformed_row_raises_with_line_number(tmp_path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\n1\t2\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="line 2"):
        data.load_qrels(tmp_path)