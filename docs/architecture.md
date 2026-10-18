# Architecture

Proposed Stage 0 design. The modules and schema below are not implemented yet.

## Components and data flow

```text
index: CLI -> discover/read/hash -> chunk -> embed -> transactional SQLite + FTS5
search semantic: CLI -> embed query -> load vector snapshot -> C++ top-k -> fetch passages
search keyword:  CLI -> literal FTS query -> FTS5 BM25 -> fetch passages
benchmark: fixtures/labels -> shared indexing/search services -> JSON measurements
```

Proposed layout and responsibilities:

| Location | Responsibility |
| --- | --- |
| `src/researchseek/cli.py` | argparse commands, validation, formatting, exit codes |
| `src/researchseek/documents.py` | recursive discovery, UTF-8 decoding, hashes, source spans |
| `src/researchseek/chunking.py` | deterministic boundaries, overlap, tokenizer limit checks |
| `src/researchseek/embeddings.py` | model loading, identity, batching, normalization |
| `src/researchseek/storage.py` | schema, transactions, reconciliation, FTS synchronization |
| `src/researchseek/indexing.py` | complete index/rebuild workflow and counts |
| `src/researchseek/search.py` | semantic/keyword search and passage lookup |
| `src/researchseek/benchmark.py` | experiment execution, metrics, environment and raw results |
| `cpp/` | cosine kernel, bounded heap, pybind11 binding, CMake target |
| `tests/`, `samples/`, `benchmarks/` | offline tests, realistic notes, evaluation labels/workloads |

Services accept an embedder interface so deterministic fakes can exercise the
pipeline without model downloads. CLI code does not contain retrieval logic.
Use the standard library for CLI, SQLite, hashes, and timing. NumPy supports the
array boundary and reference tests. Proposed build packaging is scikit-build-core
with pybind11/CMake; confirm the build dependencies in Stage 1.

## Index lifecycle and consistency

Resolve and validate the root, enumerate supported regular files in sorted
root-relative order, read raw bytes, and compute SHA-256. Use normalized text for
chunking and source spans; hashes describe raw bytes, so line-ending changes still
trigger replacement. Record byte size and mtime_ns for diagnostics; use hashes to
detect content changes. Reject symlink roots and symlink descendants.

Start a single write transaction for database reconciliation. Compare hashes and
configuration, retain unchanged records, and batch changed-document embeddings.
Delete old chunks and insert replacements within the same transaction. Reconcile
missing paths only after complete discovery/read success. Commit once all work
succeeds; roll back on any failure, including a rebuild. Large transactions and
longer write locks are an accepted MVP tradeoff. Support one writer; use SQLite
locking and a bounded busy timeout. Concurrent indexing is outside the MVP.

A database represents files successfully read during a run, not an atomic filesystem
snapshot. Detect ordinary concurrent edits using file stats before/after reads and
abort when detected; an additional run reconciles later changes. No watcher or
filesystem snapshot guarantee is planned.

Chunk by paragraphs, splitting oversized passages into tokenizer-bounded windows
with deterministic overlap. Preserve offsets into the LF-normalized document and
derive 1-based inclusive line spans for display. Store start-inclusive/end-exclusive
character offsets. Persist chunker version, sizes, overlap, tokenizer identity, and
effective model token limit including special tokens. Choose actual defaults and
exact whitespace/boundary rules in Stage 2; test all boundaries before embedding.

## Storage design

Use standard-library sqlite3 with foreign keys enabled on every connection and a
schema version in `PRAGMA user_version`. Initial schema creation and later rebuilds
must be transactional. Unsupported schema versions fail with guidance; automatic
cross-version migration is outside the MVP.

| Table | Planned fields / constraints |
| --- | --- |
| `corpus_config` | singleton key, resolved root, model ID and immutable revision, dimension, normalization, tokenizer/chunker/preprocessing configuration, embedding package versions, configuration fingerprint, last successful index time |
| `documents` | integer primary key, unique root-relative path, SHA-256, byte size, mtime_ns, indexed timestamp |
| `chunks` | integer primary key, document foreign key with cascade delete, ordinal, text, start/end character offsets, start/end lines; unique(document_id, ordinal) |
| `embeddings` | chunk_id primary/foreign key with cascade delete, vector BLOB |
| `chunks_fts` | FTS5 external-content table over chunks.text, content_rowid=id, proposed `unicode61` tokenizer |

Store vectors as explicitly little-endian contiguous float32 bytes; dimension comes
from the singleton configuration. Enforce byte length `4 * dimension`, finite
values, and valid norms at write/read boundaries; SQLite constraints alone cannot
validate vector contents. Do not pickle arrays. Index chunks.document_id.
Use insert/update/delete triggers on chunks to keep the external-content FTS index
consistent within the transaction, including cascades. Test rollback and deletion
behavior explicitly. Persist configuration using canonical serialization and a
fingerprint; matching dimensions alone do not establish compatibility.

Chunk IDs are stable for unchanged documents, but are not durable public identifiers
across replacements/rebuilds. Source path plus span is the human-facing citation.
Deleting a document removes its chunks and vectors through foreign keys and its
FTS entries through triggers. No separate on-disk native index is needed.

## Native integration and retrieval

Proposed pybind11 API accepts a C-contiguous float32 matrix `(N, d)`, a float32 query
`(d,)`, int64 chunk IDs `(N,)`, and positive k; returns ordered IDs and scores.
Python decodes SQLite blobs into a matrix and ID mapping inside one read snapshot,
then fetches passages from that same snapshot. A new CLI invocation loads a fresh
snapshot; there is no persistent process cache to invalidate in the MVP.

Both documents and queries are normalized by the embedding adapter. The native
boundary validates shapes, d>0, finite entries, unique IDs, and nonzero norms;
compute cosine with explicit norms and double-precision accumulation over float32
inputs. This retains cosine correctness if normalization has small rounding drift.
Document all conversion/copy behavior and raise Python exceptions on invalid input.

The kernel scans every row and keeps at most min(k,N) candidates in a min-heap
whose root is the worst retained candidate: lowest score, then largest chunk ID.
Final results sort by descending score, then ascending ID. Cost is O(Nd + N log k)
time and O(k) extra selection memory, in addition to the O(Nd) input matrix.
Release the GIL during computation after validation; borrowed input arrays remain
alive and must not be mutated during the call. No threads or native allocations
outlive the call. Semantic search fails if the extension is unavailable.
FTS search does not require model loading.

NumPy reference tests independently compute float64 cosine and sort, rather than
copying the heap algorithm. Near-tied scores may reorder across numeric backends.
The acceptance criteria in requirements.md distinguish these from exact ties.

## Experiments and tradeoffs

Measure native kernel time separately from model load, query encoding, matrix load,
passage lookup, and total search/CLI startup. Compare modes on the same chunk snapshot
and query labels. Proposed labels use query IDs and path/span references, resolved
to chunks for the frozen corpus/configuration; finalize their format in Stage 6.
Aggregate chunk judgments consistently and document metric definitions, relevance
grades, corpus hashes, seeds, thread counts, toolchain, model revision, and versions.
An optional NumPy latency comparison must use equivalent inputs and timing boundaries.
Neither C++ speedups nor semantic quality gains are assumed.

SQLite keeps metadata, vectors, and keyword search together with transactional
updates; vector loading and exhaustive scanning limit scale. Sentence-transformers
provides pretrained representations but adds runtime dependencies and requires
local model files. Paragraph windows preserve readable context but require
explicit token limits and overlap policy. Strict whole-run rollback simplifies
integrity at the cost of rejecting a run for one malformed file.

Open choices: exact supported toolchain versions and packaging pins (Stage 1),
chunk sizes/overlap and normalization details (Stage 2), resolved model revision
and cache/offline CLI options (Stage 3), binding copy policy and norm tolerances
(Stage 4), and evaluation label format/workload sizes (Stage 6). Changes to these
choices must remain consistent with the requirements.
