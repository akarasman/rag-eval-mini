"""Tests for the embedding cache, through its public interface (EmbeddingCache.load / save)."""
import numpy as np
import pytest

from rag_eval.embedding_cache import EmbeddingCache

IDS = ["a", "b"]
TEXTS = ["first text", "second text"]
VECTORS = np.random.default_rng(0).normal(size=(2, 4)).astype(np.float32)


@pytest.fixture
def cache(tmp_path):
    return EmbeddingCache(tmp_path / "cache")


def _saved(cache):
    cache.save("model", IDS, TEXTS, VECTORS)
    return cache


# --- save and load ----------------------------------------------------------

def test_saved_vectors_load_back_unchanged(cache):
    _saved(cache)

    assert np.array_equal(cache.load("model", IDS, TEXTS), VECTORS)


def test_loading_from_an_empty_cache_gives_none(cache):
    assert cache.load("model", IDS, TEXTS) is None


def test_the_cache_directory_is_created_by_the_first_save_not_before(cache, tmp_path):
    assert cache.load("model", IDS, TEXTS) is None
    assert not (tmp_path / "cache").exists()

    _saved(cache)

    assert (tmp_path / "cache").is_dir()


def test_saving_again_replaces_the_earlier_vectors(cache):
    _saved(cache)
    newer = VECTORS + 1.0

    cache.save("model", IDS, TEXTS, newer)

    assert np.array_equal(cache.load("model", IDS, TEXTS), newer)


def test_a_new_cache_object_on_the_same_directory_sees_the_saved_vectors(cache, tmp_path):
    _saved(cache)

    assert np.array_equal(EmbeddingCache(tmp_path / "cache").load("model", IDS, TEXTS), VECTORS)


# --- what makes a cache entry match -----------------------------------------

@pytest.mark.parametrize("fingerprint, ids, texts", [
    ("other-model", IDS, TEXTS),  # model or setting changed
    ("model", ["a", "c"], TEXTS),  # a document ID changed
    ("model", IDS, ["first text", "second text!"]),  # a document text changed
    ("model", ["b", "a"], ["second text", "first text"]),  # same documents, different order
    ("model", ["a"], ["first text"]),  # fewer documents
])
def test_a_change_to_the_model_or_corpus_is_never_served_from_the_cache(cache, fingerprint, ids, texts):
    _saved(cache)

    assert cache.load(fingerprint, ids, texts) is None


def test_the_same_input_in_new_list_objects_still_matches(cache):
    _saved(cache)

    assert cache.load("model", list(IDS), list(TEXTS)) is not None


def test_moving_characters_between_an_id_and_its_text_does_not_match(cache):
    cache.save("model", ["ab"], ["c"], VECTORS[:1])

    assert cache.load("model", ["a"], ["bc"]) is None


def test_non_ascii_text_is_matched_exactly(cache):
    cache.save("model", ["a"], ["β-catenin α"], VECTORS[:1])

    assert cache.load("model", ["a"], ["β-catenin α"]) is not None
    assert cache.load("model", ["a"], ["β-catenin"]) is None


# --- damaged and invalid files ----------------------------------------------

def test_a_cache_file_with_the_wrong_number_of_rows_is_a_miss(cache, tmp_path, caplog):
    _saved(cache)
    (path,) = (tmp_path / "cache").glob("embeddings-*.npy")
    np.save(path, np.zeros((3, 4), dtype=np.float32))  # three rows for two documents

    assert cache.load("model", IDS, TEXTS) is None
    assert "expected 2 rows" in caplog.text


def test_an_unreadable_cache_file_is_a_miss(cache, tmp_path, caplog):
    _saved(cache)
    (path,) = (tmp_path / "cache").glob("embeddings-*.npy")
    path.write_bytes(b"garbage")

    assert cache.load("model", IDS, TEXTS) is None
    assert "unreadable" in caplog.text


@pytest.mark.parametrize("vectors", [np.zeros((3, 4)), np.zeros(2), np.zeros((0, 4))])
def test_saving_vectors_that_do_not_match_the_documents_is_rejected(cache, vectors):
    with pytest.raises(ValueError, match="one vector per document"):
        cache.save("model", IDS, TEXTS, vectors)


def test_a_rejected_save_leaves_nothing_behind(cache, tmp_path):
    with pytest.raises(ValueError):
        cache.save("model", IDS, TEXTS, np.zeros((3, 4)))

    assert not (tmp_path / "cache").exists()


def test_saving_leaves_no_temporary_file_behind(cache, tmp_path):
    _saved(cache)

    assert len(list((tmp_path / "cache").iterdir())) == 1
