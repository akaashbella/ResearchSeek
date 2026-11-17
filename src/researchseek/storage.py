"""Transactional corpus metadata, chunks, and FTS5 storage."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from researchseek.chunking import CHUNKER_VERSION, Chunk
from researchseek.documents import Document

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CorpusConfig:
    root: str
    tokenizer_id: str = "whitespace-v1"
    tokenizer_max_tokens: int = 256
    special_tokens: int = 2
    chunk_max_tokens: int = 200
    chunk_overlap: int = 32
    chunker_version: str = CHUNKER_VERSION
    preprocessing_version: str = "utf8-lf-v1"
    model_id: str | None = None
    model_revision: str | None = None
    dimension: int | None = None
    normalization: str = "l2-float32-le"
    package_versions: dict[str, str] = field(default_factory=dict)

    def serialize(self) -> str:
        if not Path(self.root).is_absolute() or str(Path(self.root).resolve()) != self.root:
            raise ValueError("Corpus root must be a resolved absolute path")
        numeric = (self.chunk_max_tokens, self.tokenizer_max_tokens, self.special_tokens, self.chunk_overlap)
        if any(type(value) is not int for value in numeric):
            raise ValueError("Chunk/tokenizer limits must be integers")
        capacity = min(self.chunk_max_tokens, self.tokenizer_max_tokens) - self.special_tokens
        if self.special_tokens < 0 or capacity < 1 or not 0 <= self.chunk_overlap < capacity:
            raise ValueError("Invalid chunk/tokenizer configuration")
        if self.dimension is not None and (type(self.dimension) is not int or self.dimension < 1):
            raise ValueError("Embedding dimension must be positive")
        model_fields = (self.model_id, self.model_revision, self.dimension)
        if any(value is not None for value in model_fields) and not all(model_fields):
            raise ValueError("Model ID, immutable revision, and dimension must be specified together")
        if not all((self.tokenizer_id, self.chunker_version, self.preprocessing_version)):
            raise ValueError("Tokenizer, chunker, and preprocessing versions are required")
        if self.normalization != "l2-float32-le":
            raise ValueError("Unsupported embedding normalization")
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)


_SCHEMA = [
    """CREATE TABLE corpus_config (
        id INTEGER PRIMARY KEY CHECK (id = 1), root TEXT NOT NULL,
        model_id TEXT, model_revision TEXT, dimension INTEGER CHECK (dimension > 0),
        normalization TEXT NOT NULL, config_json TEXT NOT NULL,
        fingerprint TEXT NOT NULL, last_indexed_at TEXT
    )""",
    """CREATE TABLE documents (
        id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE,
        content_hash TEXT NOT NULL CHECK (length(content_hash) = 64),
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        mtime_ns INTEGER NOT NULL, indexed_at TEXT NOT NULL
    )""",
    """CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        text TEXT NOT NULL CHECK (length(text) > 0),
        start_char INTEGER NOT NULL CHECK (start_char >= 0),
        end_char INTEGER NOT NULL CHECK (end_char > start_char),
        start_line INTEGER NOT NULL CHECK (start_line >= 1),
        end_line INTEGER NOT NULL CHECK (end_line >= start_line),
        UNIQUE(document_id, ordinal)
    )""",
    "CREATE INDEX chunks_document_id ON chunks(document_id)",
    """CREATE TABLE embeddings (
        chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
        vector BLOB NOT NULL CHECK (typeof(vector) = 'blob' AND length(vector) > 0
                                   AND length(vector) % 4 = 0)
    )""",
    """CREATE TRIGGER embeddings_insert BEFORE INSERT ON embeddings BEGIN
        SELECT CASE WHEN (SELECT dimension FROM corpus_config WHERE id = 1) IS NULL
          OR length(new.vector) != 4 * (SELECT dimension FROM corpus_config WHERE id = 1)
          THEN RAISE(ABORT, 'Embedding dimension mismatch') END;
    END""",
    """CREATE TRIGGER embeddings_update BEFORE UPDATE ON embeddings BEGIN
        SELECT CASE WHEN (SELECT dimension FROM corpus_config WHERE id = 1) IS NULL
          OR length(new.vector) != 4 * (SELECT dimension FROM corpus_config WHERE id = 1)
          THEN RAISE(ABORT, 'Embedding dimension mismatch') END;
    END""",
    """CREATE VIRTUAL TABLE chunks_fts USING fts5(
        text, content='chunks', content_rowid='id', tokenize='unicode61'
    )""",
    """CREATE TRIGGER chunks_insert AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END""",
    """CREATE TRIGGER chunks_delete AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END""",
    """CREATE TRIGGER chunks_update AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
        INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
    END""",
]


def _relative_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if not path or parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != path or path == ".":
        raise ValueError(f"Expected a root-relative POSIX document path: {path!r}")


class Storage:
    def __init__(self, path: Path | str):
        self.connection = sqlite3.connect(path, isolation_level=None, timeout=5)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        try:
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                if self.connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
                ).fetchone():
                    raise ValueError("Unversioned, nonempty database; use a new database")
                with self.transaction():
                    for statement in _SCHEMA:
                        self.connection.execute(statement)
                    self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            elif version != SCHEMA_VERSION:
                raise ValueError(f"Unsupported database schema version {version}; expected {SCHEMA_VERSION}")
            else:
                required = {"corpus_config", "documents", "chunks", "embeddings", "chunks_fts",
                            "chunks_insert", "chunks_update", "chunks_delete",
                            "embeddings_insert", "embeddings_update"}
                found = {row[0] for row in self.connection.execute("SELECT name FROM sqlite_master")}
                if not required <= found:
                    raise ValueError("Incomplete database schema; use a new database or restore a backup")
        except BaseException as error:
            self.connection.close()
            if isinstance(error, sqlite3.OperationalError) and "no such module: fts5" in str(error):
                raise RuntimeError("SQLite FTS5 is unavailable; use an FTS5-enabled Python build") from error
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def transaction(self, *, write: bool = True) -> Iterator["Storage"]:
        if self.connection.in_transaction:
            raise RuntimeError("Nested corpus transactions are not supported")
        self.connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield self
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    @contextmanager
    def _write(self) -> Iterator[None]:
        if not self.connection.in_transaction:
            raise RuntimeError("Storage writes require 'with storage.transaction()'")
        self.connection.execute("SAVEPOINT operation")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK TO operation")
            self.connection.execute("RELEASE operation")
            raise
        else:
            self.connection.execute("RELEASE operation")

    def configuration(self) -> CorpusConfig | None:
        row = self.connection.execute("SELECT * FROM corpus_config WHERE id = 1").fetchone()
        if row is None:
            return None
        config = CorpusConfig(**json.loads(row["config_json"]))
        serialized = config.serialize()
        if hashlib.sha256(serialized.encode()).hexdigest() != row["fingerprint"]:
            raise ValueError("Corrupt corpus configuration fingerprint")
        if any(row[name] != getattr(config, name) for name in
               ("root", "model_id", "model_revision", "dimension", "normalization")):
            raise ValueError("Corrupt corpus configuration columns")
        return config

    def configure(self, config: CorpusConfig, *, rebuild: bool = False) -> None:
        serialized = config.serialize()
        with self._write():
            existing = self.configuration()
            if existing is not None and existing.serialize() != serialized and not rebuild:
                raise ValueError("Incompatible corpus configuration; explicit rebuild required")
            if existing is not None and not rebuild:
                return
            if rebuild:
                self.connection.execute("DELETE FROM documents")
                self.connection.execute("DELETE FROM corpus_config")
            self.connection.execute(
                """INSERT INTO corpus_config
                   (id, root, model_id, model_revision, dimension, normalization, config_json, fingerprint)
                   VALUES (1, ?, ?, ?, ?, ?, ?, ?)""",
                (config.root, config.model_id, config.model_revision, config.dimension,
                 config.normalization, serialized, hashlib.sha256(serialized.encode()).hexdigest()),
            )

    def get_document(self, path: str) -> sqlite3.Row | None:
        _relative_path(path)
        return self.connection.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()

    def documents(self) -> list[sqlite3.Row]:
        return self.connection.execute("SELECT * FROM documents ORDER BY path").fetchall()

    def content_hashes(self) -> dict[str, str]:
        return {row["path"]: row["content_hash"] for row in self.documents()}

    def chunks(self, document_id: int | None = None) -> list[sqlite3.Row]:
        if document_id is None:
            return self.connection.execute("SELECT * FROM chunks ORDER BY document_id, ordinal").fetchall()
        return self.connection.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY ordinal", (document_id,)
        ).fetchall()

    def put_document(self, document: Document, chunks: Iterable[Chunk]) -> int:
        """Insert or replace a document; identical hashes preserve all stored records."""
        _relative_path(document.path)
        chunks = list(chunks)
        if not re.fullmatch(r"[0-9a-f]{64}", document.content_hash) or document.size_bytes < 0:
            raise ValueError("Invalid document hash or byte size")
        if "\r" in document.text:
            raise ValueError("Document text must be LF-normalized")
        if bool(document.text.strip()) != bool(chunks):
            raise ValueError("Nonempty documents require chunks; whitespace-only documents must not have chunks")
        for ordinal, chunk in enumerate(chunks):
            if (chunk.ordinal != ordinal or not chunk.text.strip()
                or not 0 <= chunk.start_char < chunk.end_char <= len(document.text)
                or document.text[chunk.start_char:chunk.end_char] != chunk.text
                or chunk.start_line != document.text.count("\n", 0, chunk.start_char) + 1
                or chunk.end_line != document.text.count("\n", 0, chunk.end_char - 1) + 1):
                raise ValueError("Invalid chunk order, text, or source positions")
            if ordinal and (chunk.start_char <= chunks[ordinal - 1].start_char
                            or chunk.end_char <= chunks[ordinal - 1].end_char):
                raise ValueError("Chunk source positions must advance")
        with self._write():
            if self.configuration() is None:
                raise ValueError("Configure the corpus before adding documents")
            existing = self.get_document(document.path)
            if existing is not None and existing["content_hash"] == document.content_hash:
                return existing["id"]
            metadata = (document.content_hash, document.size_bytes, document.mtime_ns,
                        datetime.now(timezone.utc).isoformat())
            if existing is None:
                cursor = self.connection.execute(
                    "INSERT INTO documents (content_hash, size_bytes, mtime_ns, indexed_at, path) VALUES (?, ?, ?, ?, ?)",
                    (*metadata, document.path),
                )
                document_id = cursor.lastrowid
            else:
                document_id = existing["id"]
                self.connection.execute(
                    "UPDATE documents SET content_hash=?, size_bytes=?, mtime_ns=?, indexed_at=? WHERE id=?",
                    (*metadata, document_id),
                )
                self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            self.connection.executemany(
                """INSERT INTO chunks
                   (document_id, ordinal, text, start_char, end_char, start_line, end_line)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(document_id, chunk.ordinal, chunk.text, chunk.start_char, chunk.end_char,
                  chunk.start_line, chunk.end_line) for chunk in chunks],
            )
            return document_id

    def delete_document(self, path: str) -> bool:
        _relative_path(path)
        with self._write():
            return self.connection.execute("DELETE FROM documents WHERE path = ?", (path,)).rowcount > 0

    def remove_missing(self, present_paths: Iterable[str], *, scan_complete: bool) -> int:
        if not scan_complete:
            raise ValueError("Deletion reconciliation requires a completed scan")
        present = set(present_paths)
        for path in present:
            _relative_path(path)
        with self._write():
            missing = [row["path"] for row in self.documents() if row["path"] not in present]
            self.connection.executemany("DELETE FROM documents WHERE path = ?", [(path,) for path in missing])
            return len(missing)

    def data_version(self) -> int:
        """Detect commits by other connections while indexing is prepared."""
        return self.connection.execute("PRAGMA data_version").fetchone()[0]

    def put_embeddings(self, chunk_ids: Iterable[int], vectors: object) -> None:
        from researchseek.embeddings import validate_embeddings, vector_to_blob

        identifiers = list(chunk_ids)
        config = self.configuration()
        if config is None or config.dimension is None:
            raise ValueError("An embedding configuration is required")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate chunk IDs in embedding batch")
        matrix = validate_embeddings(vectors, config.dimension, rows=len(identifiers))
        with self._write():
            self.connection.executemany(
                "INSERT INTO embeddings(chunk_id, vector) VALUES (?, ?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET vector=excluded.vector",
                [(identifier, vector_to_blob(vector, config.dimension))
                 for identifier, vector in zip(identifiers, matrix)],
            )

    def embeddings(self) -> dict[int, "np.ndarray"]:
        from researchseek.embeddings import vector_from_blob

        config = self.configuration()
        if config is None or config.dimension is None:
            raise ValueError("An embedding configuration is required")
        return {row["chunk_id"]: vector_from_blob(row["vector"], config.dimension)
                for row in self.connection.execute("SELECT * FROM embeddings ORDER BY chunk_id")}

    def finish_index(self) -> None:
        """Validate a complete corpus before marking the run successful."""
        with self._write():
            if self.connection.execute(
                "SELECT 1 FROM chunks LEFT JOIN embeddings ON chunks.id=embeddings.chunk_id "
                "WHERE embeddings.chunk_id IS NULL LIMIT 1"
            ).fetchone():
                raise ValueError("Indexed chunks are missing embeddings; rebuild required")
            if self.connection.execute("PRAGMA foreign_key_check").fetchone():
                raise ValueError("Database foreign key integrity check failed")
            self.embeddings()
            self.connection.execute(
                "UPDATE corpus_config SET last_indexed_at = ? WHERE id = 1",
                (datetime.now(timezone.utc).isoformat(),),
            )
