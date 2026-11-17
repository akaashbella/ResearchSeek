"""Read-only semantic and literal-term FTS5 passage search."""

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
import re
import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING

from researchseek.storage import CorpusConfig, Storage

if TYPE_CHECKING:
    from researchseek.embeddings import Embedder


class QueryError(ValueError):
    """Invalid search input, rather than an operational failure."""


@dataclass(frozen=True)
class SearchResult:
    rank: int
    chunk_id: int
    path: str
    text: str
    start_line: int
    end_line: int
    mode: str
    score: float
    score_type: str


def keyword_expression(query: str) -> str:
    """Tokenize as unicode61, then quote each term instead of parsing user syntax."""
    try:
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute("CREATE VIRTUAL TABLE query_text USING fts5(text, tokenize='unicode61')")
            connection.execute("CREATE VIRTUAL TABLE terms USING fts5vocab(query_text, 'row')")
            connection.execute("INSERT INTO query_text(text) VALUES (?)", (query,))
            terms = [row[0] for row in connection.execute("SELECT term FROM terms ORDER BY term")]
    except sqlite3.OperationalError as error:
        if "no such module" in str(error):
            raise RuntimeError("SQLite FTS5 support is unavailable; use an FTS5-enabled Python build") from error
        raise
    if not terms:
        raise QueryError("Keyword query contains no searchable terms")
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def _configuration(storage: Storage) -> CorpusConfig:
    try:
        config = storage.configuration()
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid stored corpus configuration: {error}") from error
    if config is None:
        raise ValueError("Database has no indexed corpus configuration; index documents first")
    if storage.connection.execute("PRAGMA foreign_key_check").fetchone():
        raise ValueError("Database has invalid document/chunk/embedding relationships")
    try:
        storage.connection.execute("SELECT rowid FROM chunks_fts LIMIT 0")
    except sqlite3.OperationalError as error:
        raise RuntimeError(f"SQLite FTS5 index is unavailable or invalid: {error}") from error
    return config


_PASSAGE_SQL = """SELECT chunks.id, documents.path, chunks.text, chunks.start_char,
    chunks.end_char, chunks.start_line, chunks.end_line FROM chunks
    JOIN documents ON documents.id=chunks.document_id WHERE chunks.id=?"""


def _passage(storage: Storage, identifier: int, score: float, rank: int, mode: str) -> SearchResult:
    row = storage.connection.execute(_PASSAGE_SQL, (identifier,)).fetchone()
    if row is None:
        raise ValueError(f"Ranked chunk {identifier} has no document metadata")
    path, text = row["path"], row["text"]
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError(f"Chunk {identifier} has an invalid source path")
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != path or path == ".":
        raise ValueError(f"Chunk {identifier} has an invalid relative source path")
    positions = [row[name] for name in ("start_char", "end_char", "start_line", "end_line")]
    if (not isinstance(text, str) or not text.strip() or "\r" in text
        or not all(type(value) is int for value in positions)
        or row["start_char"] < 0 or row["end_char"] - row["start_char"] != len(text)
        or row["start_line"] < 1 or row["end_line"] < row["start_line"]
        or row["end_line"] != row["start_line"] + text[:-1].count("\n") or not math.isfinite(score)):
        raise ValueError(f"Chunk {identifier} has invalid text, source positions, or score")
    return SearchResult(rank, identifier, path, text, row["start_line"], row["end_line"], mode,
                        float(score), "cosine_similarity" if mode == "semantic" else "fts5_bm25")


def search(database: Path, query: str, *, mode: str = "semantic", top_k: int = 5,
           cache_dir: Path = Path(".tools/model-cache"), embedder: "Embedder | None" = None) -> list[SearchResult]:
    if not isinstance(query, str) or not query.strip():
        raise QueryError("Query must contain non-whitespace text")
    if type(top_k) is not int or top_k < 1:
        raise QueryError("top-k must be a positive integer")
    if mode not in {"semantic", "keyword"}:
        raise QueryError("Search mode must be semantic or keyword")
    query = query.replace("\r\n", "\n").replace("\r", "\n")
    with Storage(database, readonly=True) as storage:
        if mode == "keyword":
            with storage.transaction(write=False):
                _configuration(storage)
                expression = keyword_expression(query)
                count = storage.connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
                rows = storage.connection.execute(
                    "SELECT rowid, bm25(chunks_fts) AS score FROM chunks_fts "
                    "WHERE chunks_fts MATCH ? ORDER BY score ASC, rowid ASC LIMIT ?",
                    (expression, min(top_k, count)),
                ).fetchall()
                return [_passage(storage, row["rowid"], row["score"], rank, mode)
                        for rank, row in enumerate(rows, 1)]

        from researchseek.chunking import ChunkConfig
        from researchseek.embeddings import SentenceTransformerEmbedder, validate_embeddings
        from researchseek.indexing import corpus_configuration
        from researchseek.retrieval import retrieval_snapshot

        with retrieval_snapshot(storage) as snapshot:
            config = _configuration(storage)
            if config.dimension is None or not re.fullmatch(r"[0-9a-f]{40}", config.model_revision or ""):
                raise ValueError("Corpus has no immutable embedding model configuration; rebuild required")
            if not len(snapshot.chunk_ids):
                return []
            if embedder is None:
                embedder = SentenceTransformerEmbedder(config.model_id, revision=config.model_revision,
                                                       cache_dir=cache_dir, allow_download=False)
            expected = corpus_configuration(Path(config.root), embedder,
                                             ChunkConfig(config.chunk_max_tokens, config.chunk_overlap))
            if expected.serialize() != config.serialize():
                raise ValueError("Query model/tokenizer/package configuration is incompatible with the indexed corpus; "
                                 "restore matching artifacts/versions or explicitly rebuild the corpus")
            if embedder.tokenizer.count_tokens(query) > embedder.tokenizer.max_tokens:
                raise QueryError("Query exceeds the model token limit; shorten the query")
            vector = validate_embeddings(embedder.encode([query]), config.dimension, rows=1)[0]
            ranked = snapshot.top_k(vector, top_k)
            return [_passage(storage, identifier, score, rank, mode)
                    for rank, (identifier, score) in enumerate(ranked, 1)]
