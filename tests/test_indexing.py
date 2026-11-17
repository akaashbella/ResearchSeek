import sqlite3

import numpy as np
import pytest

from researchseek.chunking import ChunkConfig
from researchseek.documents import DocumentError
from researchseek.indexing import index_corpus
from researchseek.storage import CorpusConfig, Storage


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.txt").write_text("Original research passage")
    (root / "b.md").write_text("Stable reference material")
    return root, tmp_path / "corpus.sqlite3"


def state(path):
    with Storage(path) as store:
        rows = {table: [tuple(row) for row in store.connection.execute(f"SELECT * FROM {table}")]
                for table in ("corpus_config", "documents", "chunks", "embeddings")}
        rows["fts"] = [tuple(row) for row in store.connection.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rowid", ("original OR stable",))]
        return rows


def test_initial_and_unchanged(corpus, fake_embedder):
    root, path = corpus
    result = index_corpus(root, path, fake_embedder, batch_size=2)
    assert (result.added, result.updated, result.unchanged, result.deleted) == (2, 0, 0, 0)
    assert (result.documents, result.chunks) == (2, 2)
    assert all(batch == 2 for _, batch in fake_embedder.calls)
    before = state(path)
    fake_embedder.calls.clear()
    result = index_corpus(root, path, fake_embedder)
    assert result.unchanged == 2 and result.added == result.updated == result.deleted == 0
    assert fake_embedder.calls == []
    after = state(path)
    for table in ("documents", "chunks", "embeddings", "fts"):
        assert after[table] == before[table]
    with Storage(path) as store:
        assert len(store.embeddings()) == len(store.chunks()) == 2
        assert all(abs(np.linalg.norm(vector) - 1) < 1e-5 for vector in store.embeddings().values())


def test_modified_deleted_and_renamed(corpus, fake_embedder):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    with Storage(path) as store:
        stable = dict(store.get_document("b.md"))
        stable_chunks = [dict(row) for row in store.chunks(stable["id"])]
    (root / "a.txt").write_text("Replacement vocabulary")
    result = index_corpus(root, path, fake_embedder)
    assert result.updated == 1 and result.unchanged == 1
    with Storage(path) as store:
        assert dict(store.get_document("b.md")) == stable
        assert [dict(row) for row in store.chunks(stable["id"])] == stable_chunks
        assert store.connection.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'original'").fetchall() == []
    (root / "a.txt").rename(root / "renamed.txt")
    fake_embedder.calls.clear()
    result = index_corpus(root, path, fake_embedder)
    assert (result.added, result.deleted, result.unchanged) == (1, 1, 1)
    assert len(fake_embedder.calls) == 1
    (root / "renamed.txt").unlink()
    result = index_corpus(root, path, fake_embedder)
    assert result.deleted == 1 and result.documents == result.chunks == 1
    with Storage(path) as store:
        assert len(store.embeddings()) == 1


def test_empty_corpus_and_whitespace(tmp_path, fake_embedder):
    root = tmp_path / "empty"
    root.mkdir()
    path = tmp_path / "empty.sqlite3"
    result = index_corpus(root, path, fake_embedder)
    assert result.documents == result.chunks == 0
    (root / "blank.txt").write_text(" \n\t")
    result = index_corpus(root, path, fake_embedder)
    assert result.documents == 1 and result.chunks == 0
    assert fake_embedder.calls == []
    (root / "blank.txt").unlink()
    result = index_corpus(root, path, fake_embedder)
    assert result.deleted == 1


@pytest.mark.parametrize("change", ["model", "revision", "tokenizer", "packages", "chunks"])
def test_incompatible_configuration(corpus, fake_embedder, change):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    before = state(path)
    config = ChunkConfig()
    if change == "model":
        fake_embedder.model_id = "different-model"
    elif change == "revision":
        fake_embedder.model_revision = "b" * 40
    elif change == "tokenizer":
        fake_embedder.tokenizer.identity = "different-tokenizer"
    elif change == "packages":
        fake_embedder.package_versions = {"test-embedder": "2"}
    else:
        config = ChunkConfig(100, 20)
    with pytest.raises(ValueError, match="rebuild"):
        index_corpus(root, path, fake_embedder, config=config)
    assert state(path) == before
    result = index_corpus(root, path, fake_embedder, config=config, rebuild=True)
    assert result.added == result.deleted == 2 and result.unchanged == 0


def test_root_change_requires_rebuild(corpus, fake_embedder, tmp_path):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    other = tmp_path / "other"
    other.mkdir()
    (other / "new.md").write_text("New corpus")
    with pytest.raises(ValueError, match="rebuild"):
        index_corpus(other, path, fake_embedder)
    result = index_corpus(other, path, fake_embedder, rebuild=True)
    assert result.documents == 1 and result.added == 1 and result.deleted == 2


@pytest.mark.parametrize("rebuild", [False, True])
def test_embedding_failure_preserves_database(corpus, fake_embedder, monkeypatch, rebuild):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    before = state(path)
    (root / "a.txt").write_text("Changed passage")
    def fail(*args, **kwargs):
        raise RuntimeError("injected encoding failure")
    monkeypatch.setattr(fake_embedder, "encode", fail)
    with pytest.raises(RuntimeError, match="injected"):
        index_corpus(root, path, fake_embedder, rebuild=rebuild)
    assert state(path) == before


@pytest.mark.parametrize("rebuild", [False, True])
def test_database_failure_rolls_back(corpus, fake_embedder, monkeypatch, rebuild):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    before = state(path)
    (root / "a.txt").write_text("Changed passage")
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("injected database failure")
    monkeypatch.setattr(Storage, "put_embeddings", fail)
    with pytest.raises(sqlite3.OperationalError, match="injected"):
        index_corpus(root, path, fake_embedder, rebuild=rebuild)
    assert state(path) == before


def test_commit_failure_rolls_back(corpus, fake_embedder, monkeypatch):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    before = state(path)
    (root / "a.txt").write_text("Changed passage")
    original_connect = sqlite3.connect
    class FailingCommit(sqlite3.Connection):
        def commit(self):
            if self.total_changes:
                raise sqlite3.OperationalError("injected commit failure")
            return super().commit()
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: original_connect(*args, **kwargs, factory=FailingCommit))
    with pytest.raises(sqlite3.OperationalError, match="commit failure"):
        index_corpus(root, path, fake_embedder)
    assert state(path) == before


@pytest.mark.parametrize("values", [np.zeros((1, 3)), np.ones((1, 4)), np.array([[np.nan, 0, 1]])])
def test_invalid_embedder_output(corpus, fake_embedder, monkeypatch, values):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    before = state(path)
    (root / "a.txt").write_text("Changed passage")
    monkeypatch.setattr(fake_embedder, "encode", lambda *args, **kwargs: values)
    with pytest.raises(ValueError):
        index_corpus(root, path, fake_embedder)
    assert state(path) == before


def test_malformed_document_does_not_reconcile_deletion(corpus, fake_embedder):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    before = state(path)
    (root / "a.txt").unlink()
    (root / "b.md").write_bytes(b"\xff")
    with pytest.raises(DocumentError, match="UTF-8"):
        index_corpus(root, path, fake_embedder)
    assert state(path) == before


def test_source_edit_during_encoding_aborts(corpus, fake_embedder, monkeypatch):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    before = state(path)
    (root / "a.txt").write_text("Changed passage")
    original_encode = fake_embedder.encode
    def editing_encode(*args, **kwargs):
        (root / "a.txt").write_text("Another edit during encoding")
        return original_encode(*args, **kwargs)
    monkeypatch.setattr(fake_embedder, "encode", editing_encode)
    with pytest.raises(DocumentError, match="changed during indexing"):
        index_corpus(root, path, fake_embedder)
    assert state(path) == before


def test_concurrent_database_commit_is_detected_without_write_lock(corpus, fake_embedder, monkeypatch):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    (root / "a.txt").write_text("Changed passage")
    original_encode = fake_embedder.encode
    def concurrent_encode(*args, **kwargs):
        with Storage(path) as other, other.transaction():
            other.connection.execute("UPDATE corpus_config SET last_indexed_at = 'external commit'")
        return original_encode(*args, **kwargs)
    monkeypatch.setattr(fake_embedder, "encode", concurrent_encode)
    with pytest.raises(RuntimeError, match="Database changed"):
        index_corpus(root, path, fake_embedder)
    with Storage(path) as store:
        assert store.chunks()[0]["text"] == "Original research passage"


def test_missing_root_has_no_database_side_effect(tmp_path, fake_embedder):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(DocumentError):
        index_corpus(tmp_path / "missing", path, fake_embedder)
    assert not path.exists()


def test_missing_embeddings_rejected_even_when_hashes_unchanged(corpus, fake_embedder):
    root, path = corpus
    index_corpus(root, path, fake_embedder)
    with Storage(path) as store, store.transaction():
        store.connection.execute("DELETE FROM embeddings")
    before = state(path)
    with pytest.raises(ValueError, match="missing embeddings"):
        index_corpus(root, path, fake_embedder)
    assert state(path) == before


def test_stage2_database_needs_explicit_rebuild(corpus, fake_embedder):
    root, path = corpus
    with Storage(path) as store, store.transaction():
        store.configure(CorpusConfig(str(root.resolve())))
    with pytest.raises(ValueError, match="rebuild"):
        index_corpus(root, path, fake_embedder)
    assert index_corpus(root, path, fake_embedder, rebuild=True).added == 2
