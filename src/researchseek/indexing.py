"""Atomic incremental indexing with preparation outside the write transaction."""

from dataclasses import dataclass
from pathlib import Path
import re
from time import perf_counter

from researchseek.chunking import ChunkConfig, chunk_text
from researchseek.documents import DocumentError, discover_documents, read_document
from researchseek.embeddings import Embedder, validate_embeddings
from researchseek.storage import CorpusConfig, Storage


@dataclass(frozen=True)
class IndexResult:
    added: int
    updated: int
    unchanged: int
    deleted: int
    documents: int
    chunks: int
    elapsed_seconds: float


def corpus_configuration(root: Path, embedder: Embedder, config: ChunkConfig) -> CorpusConfig:
    if not re.fullmatch(r"[0-9a-f]{40}", embedder.model_revision):
        raise ValueError("Embedder must identify an immutable 40-character commit SHA")
    result = CorpusConfig(
        root=str(root.resolve()), model_id=embedder.model_id, model_revision=embedder.model_revision,
        dimension=embedder.dimension, tokenizer_id=embedder.tokenizer.identity,
        tokenizer_max_tokens=embedder.tokenizer.max_tokens, special_tokens=embedder.tokenizer.special_tokens,
        chunk_max_tokens=config.max_tokens, chunk_overlap=config.overlap,
        package_versions=dict(embedder.package_versions),
    )
    result.serialize()
    return result


def index_corpus(root: Path, database_path: Path, embedder: Embedder, *,
                 config: ChunkConfig = ChunkConfig(), rebuild: bool = False,
                 batch_size: int = 32) -> IndexResult:
    started = perf_counter()
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be a positive integer")
    root = Path(root).expanduser().absolute()
    files = discover_documents(root, database_path)
    configuration = corpus_configuration(root, embedder, config)
    added = updated = unchanged = 0
    with Storage(database_path) as storage:
        version = storage.data_version()
        with storage.transaction(write=False):
            existing = storage.configuration()
            old_hashes = storage.content_hashes()
        if existing is not None and existing.serialize() != configuration.serialize() and not rebuild:
            raise ValueError("Incompatible corpus configuration; use --rebuild to replace it")
        comparison = {} if rebuild else old_hashes
        prepared = []
        documents = {}
        for path in files:
            document = read_document(root, path)
            documents[document.path] = document
            if comparison.get(document.path) == document.content_hash:
                unchanged += 1
                continue
            chunks = chunk_text(document.text, embedder.tokenizer, config)
            vectors = embedder.encode([chunk.text for chunk in chunks], batch_size=batch_size) if chunks else []
            if chunks:
                vectors = validate_embeddings(vectors, embedder.dimension, rows=len(chunks))
            prepared.append((document, chunks, vectors))
            if document.path in comparison:
                updated += 1
            else:
                added += 1

        if discover_documents(root, database_path) != files:
            raise DocumentError("Corpus file list changed during indexing; retry the run")
        for path in files:
            current = read_document(root, path)
            if current != documents[current.path]:
                raise DocumentError(f"Document changed during indexing: {path}")

        with storage.transaction():
            if storage.data_version() != version:
                raise RuntimeError("Database changed during index preparation; retry the run")
            storage.configure(configuration, rebuild=rebuild)
            for document, chunks, vectors in prepared:
                identifier = storage.put_document(document, chunks)
                if chunks:
                    storage.put_embeddings([row["id"] for row in storage.chunks(identifier)], vectors)
            deleted = storage.remove_missing(documents, scan_complete=True)
            if rebuild:
                deleted = len(old_hashes)
            storage.finish_index()
            total_documents = len(storage.documents())
            total_chunks = len(storage.chunks())
    return IndexResult(added, updated, unchanged, deleted, total_documents, total_chunks,
                       perf_counter() - started)
