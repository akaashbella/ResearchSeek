"""Native vector retrieval within a consistent SQLite read snapshot."""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from collections.abc import Iterator

import numpy as np

from researchseek.embeddings import vector_from_blob
from researchseek.storage import CorpusConfig, Storage


@dataclass(frozen=True)
class VectorSnapshot:
    configuration: CorpusConfig | None
    chunk_ids: np.ndarray
    vectors: np.ndarray

    def top_k(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        try:
            native = import_module("researchseek._native")
        except ImportError as error:
            raise RuntimeError("Native retrieval is unavailable; rebuild with pip install -e '.[dev]'") from error
        matrix = self.vectors
        if matrix.shape == (0, 0) and isinstance(query, np.ndarray) and query.ndim == 1:
            matrix = np.empty((0, query.shape[0]), dtype=np.float32)
        ids, scores = native.top_k(matrix, query, self.chunk_ids, k)
        return [(int(identifier), float(score)) for identifier, score in zip(ids, scores)]


@contextmanager
def retrieval_snapshot(storage: Storage) -> Iterator[VectorSnapshot]:
    """Keep the read transaction open for ranking and subsequent passage lookup."""
    with storage.transaction(write=False):
        try:
            config = storage.configuration()
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid stored corpus configuration: {error}") from error
        if storage.connection.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Database contains invalid chunk/embedding relationships")
        count = storage.connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        if config is None or config.dimension is None:
            if count:
                raise ValueError("Corpus has chunks but no embedding configuration; rebuild required")
            yield VectorSnapshot(config, np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32))
            return
        if config.dimension > np.iinfo(np.intp).max // 4:
            raise ValueError("Invalid stored embedding dimension")
        invalid = storage.connection.execute(
            "SELECT chunks.id FROM chunks LEFT JOIN embeddings ON chunks.id=embeddings.chunk_id "
            "WHERE vector IS NULL OR typeof(vector) != 'blob' OR length(vector) != ? LIMIT 1",
            (4 * config.dimension,),
        ).fetchone()
        if invalid:
            raise ValueError(f"Invalid or missing stored embedding for chunk {invalid['id']}; rebuild required")
        ids = np.empty(count, dtype=np.int64)
        matrix = np.empty((count, config.dimension), dtype=np.float32)
        rows = storage.connection.execute(
            "SELECT chunks.id, embeddings.vector FROM chunks "
            "LEFT JOIN embeddings ON chunks.id = embeddings.chunk_id ORDER BY chunks.id"
        )
        for index, row in enumerate(rows):
            if row["vector"] is None:
                raise ValueError(f"Chunk {row['id']} is missing its embedding; rebuild required")
            try:
                matrix[index] = vector_from_blob(row["vector"], config.dimension)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid stored embedding for chunk {row['id']}: {error}") from error
            ids[index] = row["id"]
        yield VectorSnapshot(config, ids, matrix)


def retrieve(storage: Storage, query: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    with retrieval_snapshot(storage) as snapshot:
        return snapshot.top_k(query, k)
