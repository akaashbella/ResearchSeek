from dataclasses import replace
from pathlib import Path
import sqlite3
import struct

import pytest

from researchseek.chunking import ChunkConfig, WhitespaceTokenizer, chunk_text
from researchseek.documents import read_document
from researchseek import storage
from researchseek.storage import CorpusConfig, SCHEMA_VERSION, Storage


@pytest.fixture
def database(tmp_path):
    with Storage(tmp_path / "corpus.sqlite3") as database:
        with database.transaction():
            database.configure(CorpusConfig(str(tmp_path.resolve())))
        yield database


def make_document(root, text="original research passage", name="note.md"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    document = read_document(root, path)
    return document, chunk_text(document.text, WhitespaceTokenizer(), ChunkConfig(6, 0))


def matches(database, term):
    return [row[0] for row in database.connection.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rowid", (term,))]


def snapshot(database):
    return {table: [tuple(row) for row in database.connection.execute(f"SELECT * FROM {table}")]
            for table in ("corpus_config", "documents", "chunks", "embeddings")}


def test_schema_creation_and_reopen(tmp_path):
    path = tmp_path / "corpus.sqlite3"
    for _ in range(2):
        with Storage(path) as database:
            assert database.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
            assert database.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert database.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
            names = {row[0] for row in database.connection.execute("SELECT name FROM sqlite_master")}
            assert {"corpus_config", "documents", "chunks", "embeddings", "chunks_fts"} <= names


@pytest.mark.parametrize("version", [2, 99])
def test_unsupported_schema(tmp_path, version):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {version}")
    with pytest.raises(ValueError, match="Unsupported database schema"):
        Storage(path)


def test_unknown_unversioned_database_is_preserved(tmp_path):
    path = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES ('keep')")
    with pytest.raises(ValueError, match="nonempty"):
        Storage(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT value FROM unrelated").fetchone()[0] == "keep"


def test_incomplete_versioned_schema(tmp_path):
    path = tmp_path / "incomplete.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    with pytest.raises(ValueError, match="Incomplete"):
        Storage(path)


def test_schema_creation_rolls_back(tmp_path, monkeypatch):
    path = tmp_path / "failed.sqlite3"
    monkeypatch.setattr(storage, "_SCHEMA", [*storage._SCHEMA, "INVALID SQL"])
    with pytest.raises(sqlite3.OperationalError):
        Storage(path)
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT name FROM sqlite_master").fetchall() == []


def test_insert_and_retrieve(database, tmp_path):
    document, chunks = make_document(tmp_path, name="nested/研究.MD")
    with database.transaction():
        identifier = database.put_document(document, chunks)
    stored = database.get_document(document.path)
    assert stored["id"] == identifier
    assert stored["path"] == "nested/研究.MD"
    assert stored["content_hash"] == document.content_hash
    assert stored["size_bytes"] == document.size_bytes
    assert stored["mtime_ns"] == document.mtime_ns
    assert database.content_hashes() == {document.path: document.content_hash}
    assert [row["text"] for row in database.chunks(identifier)] == [c.text for c in chunks]
    assert matches(database, "research") == [database.chunks()[0]["id"]]
    assert database.connection.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0


def test_storage_persists_across_connections(tmp_path):
    document, chunks = make_document(tmp_path)
    path = tmp_path / "corpus.sqlite3"
    with Storage(path) as database, database.transaction():
        database.configure(CorpusConfig(str(tmp_path)))
        identifier = database.put_document(document, chunks)
    with Storage(path) as database:
        assert database.get_document(document.path)["id"] == identifier
        assert len(database.chunks(identifier)) == len(chunks)
        assert matches(database, "original")


def test_unchanged_preserves_ids_and_metadata(database, tmp_path):
    document, chunks = make_document(tmp_path)
    with database.transaction():
        identifier = database.put_document(document, chunks)
    before = snapshot(database)
    with database.transaction():
        assert database.put_document(replace(document, mtime_ns=0), chunks) == identifier
    assert snapshot(database) == before


def test_replace_and_delete_sync_fts(database, tmp_path):
    document, chunks = make_document(tmp_path)
    with database.transaction():
        identifier = database.put_document(document, chunks)
    document, chunks = make_document(tmp_path, "replacement vocabulary")
    with database.transaction():
        assert database.put_document(document, chunks) == identifier
    assert matches(database, "original") == []
    assert matches(database, "replacement")
    assert len(database.chunks()) == 1
    with database.transaction():
        assert database.delete_document(document.path)
        assert not database.delete_document(document.path)
    assert database.documents() == database.chunks() == []
    assert matches(database, "replacement") == []


def test_fts_update_trigger(database, tmp_path):
    document, chunks = make_document(tmp_path)
    with database.transaction():
        database.put_document(document, chunks)
        database.connection.execute("UPDATE chunks SET text = ?", ("updated",))
    assert matches(database, "original") == []
    assert matches(database, "updated")


def test_foreign_keys_and_uniqueness(database, tmp_path):
    document, chunks = make_document(tmp_path)
    with database.transaction():
        database.put_document(document, chunks)
        statements = [
            "INSERT INTO documents(path,content_hash,size_bytes,mtime_ns,indexed_at) SELECT path,content_hash,size_bytes,mtime_ns,indexed_at FROM documents",
            "INSERT INTO chunks(document_id,ordinal,text,start_char,end_char,start_line,end_line) SELECT document_id,ordinal,text,start_char,end_char,start_line,end_line FROM chunks",
            "INSERT INTO chunks(document_id,ordinal,text,start_char,end_char,start_line,end_line) VALUES (999,0,'missing',0,7,1,1)",
        ]
        for statement in statements:
            with pytest.raises(sqlite3.IntegrityError):
                database.connection.execute(statement)
    assert database.connection.execute("PRAGMA foreign_key_check").fetchall() == []


def test_embedding_dimensions_and_cascade(database, tmp_path):
    document, chunks = make_document(tmp_path)
    config = replace(database.configuration(), model_id="test-only", model_revision="fixed", dimension=2)
    with database.transaction():
        database.configure(config, rebuild=True)
        database.put_document(document, chunks)
        identifier = database.chunks()[0]["id"]
        with pytest.raises(sqlite3.IntegrityError, match="dimension"):
            database.connection.execute("INSERT INTO embeddings VALUES (?, ?)", (identifier, b"1234"))
        database.connection.execute("INSERT INTO embeddings VALUES (?, ?)", (identifier, struct.pack("<ff", 1, 0)))
        with pytest.raises(sqlite3.IntegrityError, match="dimension"):
            database.connection.execute("UPDATE embeddings SET vector = ?", (b"1234",))
        database.delete_document(document.path)
    assert database.connection.execute("SELECT * FROM embeddings").fetchall() == []
    assert matches(database, "original") == []


def test_empty_documents(database, tmp_path):
    document, chunks = make_document(tmp_path, " \n\n\t")
    with database.transaction():
        database.put_document(document, chunks)
    assert len(database.documents()) == 1
    assert database.chunks() == []


def test_missing_chunks_are_rejected(database, tmp_path):
    document, _ = make_document(tmp_path)
    with database.transaction():
        with pytest.raises(ValueError, match="require chunks"):
            database.put_document(document, [])


def test_reconcile_requires_complete_scan(database, tmp_path):
    first = make_document(tmp_path, name="first.txt")
    second = make_document(tmp_path, name="nested/second.md")
    with database.transaction():
        database.put_document(*first)
        database.put_document(*second)
    with database.transaction():
        with pytest.raises(ValueError, match="completed scan"):
            database.remove_missing([], scan_complete=False)
        assert len(database.documents()) == 2
        assert database.remove_missing(["nested/second.md"], scan_complete=True) == 1
    assert list(database.content_hashes()) == ["nested/second.md"]
    with database.transaction():
        assert database.remove_missing([], scan_complete=True) == 1
    assert matches(database, "original") == []


def test_whole_run_rollback_restores_metadata_chunks_and_fts(database, tmp_path):
    original = make_document(tmp_path)
    with database.transaction():
        database.put_document(*original)
    before = snapshot(database)
    with pytest.raises(RuntimeError, match="injected"):
        with database.transaction():
            database.put_document(*make_document(tmp_path, "changed terms"))
            database.put_document(*make_document(tmp_path, "another file", "second.md"))
            database.remove_missing(["second.md"], scan_complete=True)
            raise RuntimeError("injected failure")
    assert snapshot(database) == before
    assert matches(database, "original")
    assert matches(database, "changed") == matches(database, "another") == []


def test_replace_failure_is_atomic_even_if_caller_handles_error(database, tmp_path):
    with database.transaction():
        database.put_document(*make_document(tmp_path))
    before = snapshot(database)
    with database.transaction():
        database.connection.execute("""CREATE TEMP TRIGGER fail_second_chunk BEFORE INSERT ON chunks
            WHEN new.ordinal = 1 BEGIN SELECT RAISE(ABORT, 'injected'); END""")
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            database.put_document(*make_document(tmp_path, "replacement has more than four content tokens"))
    assert snapshot(database) == before
    assert matches(database, "original")
    assert matches(database, "replacement") == []


def test_failed_rebuild_restores_prior_configuration_and_corpus(database, tmp_path):
    with database.transaction():
        database.put_document(*make_document(tmp_path))
    before = snapshot(database)
    with pytest.raises(RuntimeError):
        with database.transaction():
            database.configure(replace(database.configuration(), chunk_max_tokens=100), rebuild=True)
            assert database.chunks() == []
            raise RuntimeError("failed rebuild")
    assert snapshot(database) == before
    assert matches(database, "original")


@pytest.mark.parametrize("field,value", [("root", "/different"), ("chunk_max_tokens", 100),
    ("chunk_overlap", 16), ("tokenizer_id", "different"), ("preprocessing_version", "v2"),
    ("package_versions", {"test": "1"})])
def test_incompatible_configuration(database, field, value):
    with database.transaction():
        with pytest.raises(ValueError, match="rebuild"):
            database.configure(replace(database.configuration(), **{field: value}))


@pytest.mark.parametrize("changes", [{"root": "relative"}, {"chunk_overlap": 500},
    {"dimension": 0}, {"model_id": "incomplete"}, {"normalization": "unknown"},
    {"special_tokens": -1}, {"tokenizer_id": ""}, {"chunk_max_tokens": 2.5}, {"dimension": True}])
def test_invalid_configuration(tmp_path, changes):
    with pytest.raises(ValueError):
        replace(CorpusConfig(str(tmp_path)), **changes).serialize()


def test_configuration_fingerprint_and_columns(database):
    original = database.configuration()
    with database.transaction():
        database.configure(original)
    assert database.configuration() == original
    with database.transaction():
        database.connection.execute("UPDATE corpus_config SET fingerprint = 'corrupt'")
        with pytest.raises(ValueError, match="fingerprint"):
            database.configuration()


def test_configuration_column_corruption(database):
    with database.transaction():
        database.connection.execute("UPDATE corpus_config SET normalization = 'corrupt'")
        with pytest.raises(ValueError, match="columns"):
            database.configuration()


def test_unconfigured_corpus_rejects_documents(tmp_path):
    with Storage(tmp_path / "empty.sqlite3") as database, database.transaction():
        with pytest.raises(ValueError, match="Configure"):
            database.put_document(*make_document(tmp_path))


@pytest.mark.parametrize("path", ["", "/absolute.txt", "../outside.txt", "a/../b.txt", "./a.txt", "a//b.txt"])
def test_invalid_relative_paths(database, tmp_path, path):
    document, chunks = make_document(tmp_path)
    with database.transaction():
        with pytest.raises(ValueError, match="POSIX"):
            database.put_document(replace(document, path=path), chunks)


def test_writes_require_transaction(database, tmp_path):
    with pytest.raises(RuntimeError, match="transaction"):
        database.put_document(*make_document(tmp_path))
    with database.transaction():
        with pytest.raises(RuntimeError, match="Nested"):
            with database.transaction():
                pass


def test_source_validation_precedes_replacement(database, tmp_path):
    document, chunks = make_document(tmp_path)
    with database.transaction():
        database.put_document(document, chunks)
    before = snapshot(database)
    invalid = [replace(chunks[0], text="wrong source")]
    with database.transaction():
        with pytest.raises(ValueError, match="positions"):
            database.put_document(document, invalid)
    assert snapshot(database) == before


def test_sql_parameters_preserve_unusual_filename(database, tmp_path):
    name = "note'; DROP TABLE chunks;--.md"
    with database.transaction():
        identifier = database.put_document(*make_document(tmp_path, name=name))
    assert database.get_document(name)["id"] == identifier
    assert len(database.chunks()) == 1


def test_replacement_and_rollback_preserve_embedding_state(database, tmp_path):
    config = replace(database.configuration(), model_id="test-only", model_revision="fixed", dimension=2)
    with database.transaction():
        database.configure(config, rebuild=True)
        database.put_document(*make_document(tmp_path))
        database.connection.execute("INSERT INTO embeddings VALUES (?, ?)",
                                    (database.chunks()[0]["id"], struct.pack("<ff", 1, 0)))
    before = snapshot(database)
    with pytest.raises(RuntimeError):
        with database.transaction():
            database.put_document(*make_document(tmp_path, "different vocabulary"))
            assert database.connection.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
            raise RuntimeError("rollback")
    assert snapshot(database) == before
    with database.transaction():
        database.put_document(*make_document(tmp_path, "different vocabulary"))
    assert database.connection.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
