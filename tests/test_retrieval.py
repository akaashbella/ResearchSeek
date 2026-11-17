import sqlite3
import struct
from dataclasses import replace
import hashlib

import numpy as np
import pytest

from numpy_reference import cosine_reference
from researchseek import retrieval
from researchseek.indexing import index_corpus
from researchseek.retrieval import retrieve, retrieval_snapshot
from researchseek.storage import Storage


@pytest.fixture
def corpus(tmp_path, fake_embedder):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.txt").write_text("Original research passage")
    (root / "b.md").write_text("Another distinct note")
    path = tmp_path / "corpus.sqlite3"
    index_corpus(root, path, fake_embedder)
    return root, path, fake_embedder


def test_snapshot_matrix_decoding_association_and_ranking(corpus):
    root, path, embedder = corpus
    query = embedder.encode(["A test query"])[0]
    with Storage(path) as store:
        with retrieval_snapshot(store) as snapshot:
            assert snapshot.vectors.dtype == np.float32
            assert snapshot.vectors.flags.c_contiguous and snapshot.vectors.flags.aligned
            assert snapshot.chunk_ids.dtype == np.int64
            assert list(snapshot.chunk_ids) == sorted(snapshot.chunk_ids)
            for identifier, vector in zip(snapshot.chunk_ids, snapshot.vectors):
                text = store.connection.execute("SELECT text FROM chunks WHERE id=?", (int(identifier),)).fetchone()[0]
                np.testing.assert_array_equal(vector, embedder.encode([text])[0])
            expected_ids, scores = cosine_reference(snapshot.vectors, query, snapshot.chunk_ids, 2)
            result = snapshot.top_k(query, 2)
            assert [item[0] for item in result] == expected_ids.tolist()
            np.testing.assert_allclose([item[1] for item in result], scores, atol=1e-5)
        assert retrieve(store, query, 2) == result
        assert retrieve(store, query, 2) == result


def test_updates_and_deletions_use_current_vectors(corpus):
    root, path, embedder = corpus
    (root / "a.txt").write_text("Changed passage with a different embedding")
    (root / "b.md").unlink()
    index_corpus(root, path, embedder)
    query = embedder.encode(["Changed passage with a different embedding"])[0]
    with Storage(path) as store:
        result = retrieve(store, query, 100)
        assert len(result) == 1
        identifier, score = result[0]
        assert identifier == store.chunks()[0]["id"]
        assert score == pytest.approx(1, abs=1e-12)
        assert store.connection.execute(
            "SELECT documents.path FROM chunks JOIN documents ON documents.id=chunks.document_id WHERE chunks.id=?",
            (identifier,),
        ).fetchone()[0] == "a.txt"


def test_empty_database(tmp_path):
    with Storage(tmp_path / "empty.sqlite3") as store:
        assert retrieve(store, np.ones(3, dtype=np.float32)) == []


def test_empty_indexed_corpus(tmp_path, fake_embedder):
    root = tmp_path / "empty"
    root.mkdir()
    path = tmp_path / "empty.sqlite3"
    index_corpus(root, path, fake_embedder)
    with Storage(path) as store:
        assert retrieve(store, np.ones(3, dtype=np.float32)) == []
        with pytest.raises(ValueError, match="dimensions"):
            retrieve(store, np.ones(2, dtype=np.float32))


@pytest.mark.parametrize("kind", ["missing", "length", "zero", "nonfinite", "norm", "configuration", "dtype"])
def test_corrupt_database_errors(corpus, kind):
    _, path, embedder = corpus
    with Storage(path) as store:
        with store.transaction():
            if kind == "missing":
                store.connection.execute("DELETE FROM embeddings")
            elif kind == "configuration":
                store.connection.execute("UPDATE corpus_config SET fingerprint='corrupt'")
            else:
                store.connection.execute("PRAGMA ignore_check_constraints = ON")
                if kind == "length":
                    store.connection.execute("DROP TRIGGER embeddings_update")
                blobs = {"length": b"bad", "zero": struct.pack("<fff", 0, 0, 0),
                         "nonfinite": struct.pack("<fff", np.nan, 0, 0), "norm": struct.pack("<fff", 3, 4, 0),
                         "dtype": "abcdefghijkl"}
                store.connection.execute("UPDATE embeddings SET vector=?", (blobs[kind],))
        with pytest.raises(ValueError, match="embedding|fingerprint"):
            retrieve(store, embedder.encode(["query"])[0])
        assert not store.connection.in_transaction


def test_corrupt_dimension_rejected_before_matrix_allocation(corpus):
    _, path, _ = corpus
    with Storage(path) as store:
        config = replace(store.configuration(), dimension=2 ** 50)
        serialized = config.serialize()
        with store.transaction():
            store.connection.execute(
                "UPDATE corpus_config SET dimension=?, config_json=?, fingerprint=?",
                (config.dimension, serialized, hashlib.sha256(serialized.encode()).hexdigest()),
            )
        with pytest.raises(ValueError, match="stored embedding"):
            retrieve(store, np.ones(3, dtype=np.float32))


def test_orphan_vector_rejected(corpus):
    _, path, _ = corpus
    with sqlite3.connect(path) as connection:
        connection.execute("INSERT INTO embeddings(chunk_id, vector) VALUES (?, ?)",
                           (999, struct.pack("<fff", 1, 0, 0)))
    with Storage(path) as store:
        with pytest.raises(ValueError, match="relationships"):
            retrieve(store, np.ones(3, dtype=np.float32))


def test_native_failure_leaves_read_transaction_closed(corpus, monkeypatch):
    from researchseek import _native
    _, path, _ = corpus
    def fail(*args, **kwargs):
        raise RuntimeError("injected native failure")
    monkeypatch.setattr(_native, "top_k", fail)
    with Storage(path) as store:
        with pytest.raises(RuntimeError, match="injected native failure"):
            retrieve(store, np.ones(3, dtype=np.float32))
        assert not store.connection.in_transaction


def test_missing_extension_has_actionable_error(corpus, monkeypatch):
    _, path, _ = corpus
    def missing(name):
        raise ImportError("no native module")
    monkeypatch.setattr(retrieval, "import_module", missing)
    with Storage(path) as store:
        with pytest.raises(RuntimeError, match="rebuild"):
            retrieve(store, np.ones(3, dtype=np.float32))


def test_snapshot_remains_consistent_across_writer_commit(corpus):
    root, path, embedder = corpus
    with Storage(path) as store:
        store.connection.execute("PRAGMA journal_mode=WAL")
        with retrieval_snapshot(store) as snapshot:
            original = store.chunks()[0]["text"]
            old_vectors = snapshot.vectors.copy()
            identifier = int(snapshot.chunk_ids[0])
            (root / "a.txt").write_text("A replacement passage")
            index_corpus(root, path, embedder)
            assert store.connection.execute("SELECT text FROM chunks WHERE id=?", (identifier,)).fetchone()[0] == original
            np.testing.assert_array_equal(snapshot.vectors, old_vectors)
            assert snapshot.top_k(old_vectors[0], 1)[0][0] == identifier
        assert store.chunks()[0]["text"] == "A replacement passage"


def test_query_dtype_error_is_not_hidden(corpus):
    _, path, _ = corpus
    with Storage(path) as store:
        with pytest.raises(TypeError, match="dtype"):
            retrieve(store, np.ones(3, dtype=np.float64))


def test_invalid_database_file(tmp_path):
    path = tmp_path / "broken.sqlite3"
    path.write_bytes(b"not a SQLite database")
    with pytest.raises(sqlite3.DatabaseError, match="not a database"):
        Storage(path)
