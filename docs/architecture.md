# Architecture

Stage 2 implements document discovery/reading, chunking, SQLite persistence, and
FTS5 synchronization. Stage 1's package, CLI placeholders, and native diagnostic
are unchanged. The complete indexing workflow, model adapter, retrieval, and
benchmarking remain planned.

## Components and data flow

```text
index: CLI -> discover/read/hash -> chunk -> embed -> transactional SQLite + FTS5
search semantic: CLI -> embed query -> load vector snapshot -> C++ top-k -> fetch passages
search keyword:  CLI -> literal FTS query -> FTS5 BM25 -> fetch passages
benchmark: fixtures/labels -> shared indexing/search services -> JSON measurements
```

Module responsibilities (embedding, indexing, search, and benchmark modules remain planned):

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
array boundary and reference tests in later stages. Packaging uses scikit-build-core
with pybind11/CMake and a `src/` layout. CMake requires 3.18+ and C++17; the build
backend supplies CMake/Ninja when needed. The current extension is
`researchseek._native`, built from `cpp/bindings.cpp`; `diagnostic()` returns the
compiler's `__cplusplus` value. No similarity code is implemented yet.

`cli.py` uses argparse, and `__main__.py` supports `python -m researchseek`.
`diagnostics.py` probes FTS5 using a temporary in-memory table and raises an
actionable error if unavailable. The package currently has no runtime dependencies;
pytest is a development extra. Build dependency bounds and test settings live in
`pyproject.toml`. No automatic native rebuild occurs after C++ edits.

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

Stage 2 exposes `discover_documents` and `read_document` separately; it does not
orchestrate an indexing run. Discovery returns absolute paths sorted by their
root-relative POSIX strings. Database exclusion requires passing `database_path`;
it excludes that file and its `-wal`, `-shm`, and `-journal` sidecars. Read failures
raise `DocumentError` with path/context. Reads compare device, inode, byte size,
mtime_ns, and ctime_ns before opening, on the open descriptor before/after reading,
and on the path afterward. The file is opened with `O_NOFOLLOW`. These checks detect
ordinary edits/replacements, not every possible filesystem race.

### Chunking rules

`chunk_text(text, tokenizer, ChunkConfig(...))` uses `paragraph-windows-v1`:

- Default limit: 200 tokens including special tokens; default overlap: 32 content
  tokens. The effective limit is the smaller of the configured and tokenizer limits.
  Overlap must be nonnegative and below the available content-token capacity.
- Prefer the last blank-line paragraph boundary that fits the current window and
  leaves enough content for progress after overlap. Blank lines may contain spaces
  or tabs. Combine small paragraphs when they fit; split long ones at token offsets.
- The tokenizer supplies ordered, non-overlapping character spans, an exact
  untruncated token count including special tokens, its identity, model limit, and
  special-token count. Recount each candidate slice and shorten it if necessary;
  raise an error if even one token with its surrounding whitespace cannot fit.
- Start the next window up to 32 content tokens before the previous end, always
  advancing. Stop when the final window reaches the source end; do not emit an
  extra overlap-only tail. Equal text at different source positions is retained.
- Preserve source slices exactly, including indentation and blank-line separators.
  Whitespace between tokens belongs to the preceding window; leading/trailing
  whitespace stays in the first/last slice. Empty/whitespace-only documents yield
  no chunks. Input must already use LF line endings.
- Ordinals start at zero. Character offsets are Python Unicode character indices,
  start-inclusive/end-exclusive. Line spans are 1-based and inclusive, counting the
  line containing the final included character. A final newline does not create
  a citation to the following empty line.

`WhitespaceTokenizer` is an offline stub: each `\S+` match is one content token,
with two simulated special tokens and a 256-token limit. Defaults therefore allow
198 content tokens per window. It does not model transformer tokenization; Stage 3
must supply the real tokenizer and rebuild incompatible corpora. Configuration
persists these settings, tokenizer identity, and preprocessing/chunker versions.

## Storage design

Use standard-library sqlite3 with foreign keys enabled on every connection and a
schema version in `PRAGMA user_version`. Initial schema creation and later rebuilds
must be transactional. Unsupported schema versions fail with guidance; automatic
cross-version migration is outside the MVP.

| Table | Fields / constraints |
| --- | --- |
| `corpus_config` | singleton id=1, resolved root, nullable model ID/revision/dimension, normalization, canonical config_json (tokenizer/chunker/preprocessing settings and package versions), SHA-256 fingerprint, nullable last_indexed_at |
| `documents` | integer primary key, unique root-relative path, SHA-256, byte size, mtime_ns, indexed timestamp |
| `chunks` | integer primary key, document foreign key with cascade delete, ordinal, text, start/end character offsets, start/end lines; unique(document_id, ordinal) |
| `embeddings` | chunk_id primary/foreign key with cascade delete, vector BLOB |
| `chunks_fts` | FTS5 external-content table over chunks.text, content_rowid=id, `unicode61` tokenizer |

The current schema version is 1. `Storage` enables foreign keys and a five-second
busy timeout on every connection. It rejects unsupported versions, unversioned
nonempty databases, and versioned databases missing required tables/triggers.
Schema creation uses individual SQL statements in one transaction; it does not
use `executescript`, which could commit pending work.

`CorpusConfig` stores the root as a resolved absolute path. Document paths are
canonical root-relative POSIX strings, without `.` or `..` components. Canonical
JSON and its fingerprint must agree with the separately stored configuration
columns. All configuration fields must match when reopening for writes, not just
the embedding dimension. `configure(..., rebuild=True)` explicitly clears corpus
records and replaces the configuration inside the caller's transaction.

Model ID, immutable revision, and dimension are unset together during Stage 2.
The embeddings table is intentionally empty; document/chunk persistence is tested
independently. Stage 3 must enforce exactly one finite, nonzero L2-normalized vector
per chunk before committing a complete indexing run. The reserved table already
checks BLOB storage/float32 byte alignment and uses insert/update triggers to
require the configured dimension. Finite values and norms still require Python
validation in Stage 3; SQLite does not validate float contents.

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

### Storage operations and transactions

`put_document(document, chunks)` inserts a new document or replaces a changed
document's chunks, preserving its document ID. Equal hashes leave all existing
records, IDs, and timestamps unchanged. It validates chunk text/order/positions
against the supplied normalized source. `get_document`, `documents`, `chunks`,
and `content_hashes` provide metadata/text access. `delete_document` removes a
single path; `remove_missing(present_paths, scan_complete=True)` reconciles deletions
only after the caller explicitly confirms a complete scan. It does not scan files.

All write operations require `with storage.transaction():`, which uses
`BEGIN IMMEDIATE` and commits once or rolls back on an escaping exception. Each
mutator uses a savepoint so a failed replacement cannot leave half its chunks
inserted even if the caller handles the error. If the error propagates, the outer
transaction rolls back all operations. Nested corpus transactions are rejected.
Stage 3 must let indexing failures propagate to preserve whole-run rollback.

Insert/update/delete triggers keep the external-content FTS5 index synchronized
with chunks, including foreign-key cascades. Tests query `MATCH` after replacement,
deletion, and rollback to check the inverted index rather than only its content view.
No public keyword query service or ranking command is implemented yet.

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

Open choices: additional supported toolchain versions (Ubuntu verification pending),
resolved model revision and cache/offline CLI options (Stage 3), binding copy policy
and norm tolerances (Stage 4), and evaluation label format/workload sizes (Stage 6). Changes to these
choices must remain consistent with the requirements.
