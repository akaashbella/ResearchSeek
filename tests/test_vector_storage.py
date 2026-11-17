import sqlite3
import struct

import numpy as np
import pytest

from researchseek.indexing import index_corpus
from researchseek.storage import Storage


@pytest.fixture
def vector_database(tmp_path, fake_embedder):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("A vector storage fixture")
    path = tmp_path / "vectors.sqlite3"
    index_corpus(root, path, fake_embedder)
    with Storage(path) as store:
        yield store


def test_vector_roundtrip_and_relationship(vector_database):
    store = vector_database
    identifier = store.chunks()[0]["id"]
    with store.transaction():
        store.put_embeddings([identifier], [[0, 0, 1]])
        store.finish_index()
    assert list(store.embeddings()) == [identifier]
    np.testing.assert_array_equal(store.embeddings()[identifier], [0, 0, 1])
    assert store.connection.execute("SELECT vector FROM embeddings").fetchone()[0] == struct.pack("<fff", 0, 0, 1)


@pytest.mark.parametrize("values", [[[1, 0]], [[0, 0, 0]], [[np.nan, 0, 1]], [[2, 0, 0]], [[1, 0, 0], [0, 1, 0]]])
def test_bad_vectors_do_not_replace_stored_data(vector_database, values):
    store = vector_database
    before = store.connection.execute("SELECT vector FROM embeddings").fetchone()[0]
    with store.transaction():
        with pytest.raises(ValueError):
            store.put_embeddings([store.chunks()[0]["id"]], values)
    assert store.connection.execute("SELECT vector FROM embeddings").fetchone()[0] == before


def test_duplicate_and_unknown_chunk_ids(vector_database):
    store = vector_database
    identifier = store.chunks()[0]["id"]
    with store.transaction():
        with pytest.raises(ValueError, match="Duplicate"):
            store.put_embeddings([identifier, identifier], [[1, 0, 0], [0, 1, 0]])
        with pytest.raises(sqlite3.IntegrityError):
            store.put_embeddings([identifier, 999], [[1, 0, 0], [0, 1, 0]])
    assert len(store.embeddings()) == 1


def test_corrupt_blob_rejected_on_read_and_finish(vector_database):
    store = vector_database
    with store.transaction():
        store.connection.execute("UPDATE embeddings SET vector = ?", (struct.pack("<fff", np.inf, 0, 0),))
        with pytest.raises(ValueError, match="NaN or infinity"):
            store.embeddings()
        with pytest.raises(ValueError):
            store.finish_index()


def test_writes_require_transaction(vector_database):
    with pytest.raises(RuntimeError, match="transaction"):
        vector_database.put_embeddings([vector_database.chunks()[0]["id"]], [[1, 0, 0]])
