# Architecture

Stage 5 adds semantic and literal-term keyword search services and the `search`
CLI. Indexing and native retrieval algorithms are unchanged. Storage adds a
read-only connection option. Benchmarking remains planned.

## Components and data flow

```text
index: CLI -> discover/read/hash -> chunk -> embed -> transactional SQLite + FTS5
search semantic: CLI -> embed query -> load vector snapshot -> C++ top-k -> fetch passages
search keyword:  CLI -> literal FTS query -> FTS5 BM25 -> fetch passages
benchmark: fixtures/labels -> shared indexing/search services -> JSON measurements
```

Module responsibilities (the benchmark module remains planned):

| Location | Responsibility |
| --- | --- |
| `src/researchseek/cli.py` | argparse commands, validation, formatting, exit codes |
| `src/researchseek/documents.py` | recursive discovery, UTF-8 decoding, hashes, source spans |
| `src/researchseek/chunking.py` | deterministic boundaries, overlap, tokenizer limit checks |
| `src/researchseek/embeddings.py` | model loading, identity, batching, normalization |
| `src/researchseek/storage.py` | schema, transactions, reconciliation, FTS synchronization |
| `src/researchseek/indexing.py` | complete index/rebuild workflow and counts |
| `src/researchseek/retrieval.py` | validated SQLite vector snapshots and native ranking |
| `src/researchseek/search.py` | semantic/keyword search and passage lookup |
| `src/researchseek/benchmark.py` | experiment execution, metrics, environment and raw results |
| `cpp/` | cosine kernel, bounded heap, pybind11 binding, CMake target |
| `tests/`, `samples/`, `benchmarks/` | offline tests, realistic notes, evaluation labels/workloads |

Services accept an embedder interface so deterministic fakes can exercise the
pipeline without model downloads. CLI code does not contain retrieval logic.
Use the standard library for CLI, SQLite, hashes, and timing. NumPy supports the
array boundary, vector validation, and reference tests. Packaging uses scikit-build-core
with pybind11/CMake and a `src/` layout. CMake requires 3.18+ and C++17; the build
backend supplies CMake/Ninja when needed. The current extension is
`researchseek._native`, built from `cpp/bindings.cpp` and `cpp/vector_search.cpp`;
`diagnostic()` returns the compiler's `__cplusplus` value. The binding validates
NumPy layout; the independent C++ kernel handles cosine scoring and heap selection.

`cli.py` uses argparse, and `__main__.py` supports `python -m researchseek`.
`diagnostics.py` probes FTS5 using a temporary in-memory table and raises an
actionable error if unavailable. Runtime dependencies are NumPy, sentence-transformers,
and huggingface-hub; pytest is a development extra. Dependency bounds and test settings live in
`pyproject.toml`. No automatic native rebuild occurs after C++ edits.

## Index lifecycle and consistency

Resolve and validate the root, enumerate supported regular files in sorted
root-relative order, read raw bytes, and compute SHA-256. Use normalized text for
chunking and source spans; hashes describe raw bytes, so line-ending changes still
trigger replacement. Record byte size and mtime_ns for diagnostics; use hashes to
detect content changes. Reject symlink roots and symlink descendants.

`index_corpus` reads configuration/hashes in a short read transaction and captures
SQLite `data_version`. It reads every source, compares hashes, and chunks/encodes
only changed or new files. Changed passages are encoded in batches (default 32).
Preparation retains normalized documents and changed vectors in memory. It runs
outside the write transaction to avoid holding a writer lock during model work.

After preparation, repeat discovery and reread files to reject detected source
changes. Acquire `BEGIN IMMEDIATE` and recheck `data_version`; if another connection
committed during preparation, abort and require a retry. Then configure/rebuild,
replace changed documents/chunks/vectors, reconcile deletions, validate completeness,
mark the successful index time, and commit once. Any escaping failure, including a
commit failure, rolls back all corpus mutations. One writer is supported with a
five-second busy timeout. This replaces Stage 2's proposed longer write transaction.

A database represents files successfully read during a run, not an atomic filesystem
snapshot. Detect ordinary concurrent edits using file stats before/after reads and
abort when detected; an additional run reconciles later changes. No watcher or
filesystem snapshot guarantee is planned.

`discover_documents` and `read_document` remain independently usable. Discovery
returns absolute paths sorted by their
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
now supplies the real tokenizer and requires rebuilding incompatible corpora.
Configuration persists these settings, tokenizer identity, and preprocessing/chunker versions.

`ModelTokenizer` adapts the fast tokenizer loaded from the model snapshot. It uses
untruncated, non-special-token offsets for splitting and recounts each source slice
with special tokens included. Its identity includes the repository/commit and a
SHA-256 of the serialized tokenizer backend. For MiniLM, the effective model limit
is 256 tokens with two special tokens; the configured 200-token default leaves
198 content tokens. The existing chunking algorithm and source-position rules are
unchanged. Each passage is checked again before encoding to prevent silent truncation.

### Embedding adapter and artifacts

`SentenceTransformerEmbedder` uses sentence-transformers on CPU. The initial model
is `sentence-transformers/all-MiniLM-L6-v2`, with pinned default commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and dimension 384. Explicitly selected
tags/branches resolve through the Hugging Face snapshot cache to a 40-character
commit SHA; only that immutable revision is persisted. Normal indexing is
cache-only. `--download-model` permits acquisition, and `--cache-dir` selects the
cache (default `.tools/model-cache`). Only the default model has been verified.

The adapter loads the resolved snapshot locally with remote code disabled, uses
no query/passage prefix, and shares one `encode` method for passages and future
queries. It validates shapes and finite/nonzero norms, normalizes with float64
accumulation, and returns contiguous float32 vectors. Storage requires unit norms
within absolute tolerance `1e-5`. Relevant versions include sentence-transformers,
Transformers, tokenizers, PyTorch, NumPy, and huggingface-hub. They participate in
the configuration fingerprint; dependency upgrades can require rebuilding.

Ordinary tests use a deterministic fake embedder. The separate `real_model` test
requires `--real-model` and a provisioned local cache; it does not download files.

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

Stage 2 databases may have no model metadata or embeddings and require an explicit
rebuild before full indexing. The standalone metadata/chunk API remains usable
without vectors for those tests. Successful Stage 3 indexing enforces exactly one
finite, nonzero L2-normalized vector per chunk before committing. The table
checks BLOB storage/float32 byte alignment and uses insert/update triggers to
require the configured dimension. Python validates finite values and norms;
SQLite does not validate float contents.

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
Indexing lets failures propagate to preserve whole-run rollback. Read snapshots
use `transaction(write=False)` without acquiring a writer lock.

`put_embeddings` validates dimensions, row counts, finite values, and unit norms
before storing little-endian float32 BLOBs by chunk ID. `embeddings` deserializes
and validates them. `finish_index` rejects missing vectors or foreign-key errors,
validates stored vectors, and updates `last_indexed_at`. Chunk IDs are unique
embedding keys; replacement/deletion cascades remove old vectors and FTS entries.

Insert/update/delete triggers keep the external-content FTS5 index synchronized
with chunks, including foreign-key cascades. Tests query `MATCH` after replacement,
deletion, and rollback to check the inverted index rather than only its content view.
No public keyword query service or ranking command is implemented yet.

## Native integration and retrieval

`_native.top_k(matrix, query, chunk_ids, k)` accepts native-endian float32 arrays
`(N, d)` and `(d,)`, native-endian int64 IDs `(N,)`, and a positive integer k.
Arrays must be C-contiguous and naturally aligned. Wrong dtypes (including swapped
byte order), lists, noncontiguous arrays, and unaligned buffers are rejected rather
than converted. Read-only arrays and contiguous views are accepted. k supports
integer index values, including NumPy integers; booleans/floats are rejected and
arbitrarily large positive Python integers are clipped to the corpus size.

Validation checks shapes, d>0, lengths, finite values, nonzero norms, unique IDs,
and address-size products before pointer arithmetic. Duplicate detection uses an
unordered set with average O(N) time and O(N) temporary storage. Numeric validation
scans the inputs while holding the GIL. Only the scoring/selection kernel releases
it. Array arguments hold their owners alive; no borrowed pointer survives the call.
Callers must not mutate or resize inputs through any alias during the call.

The C++ kernel scans every vector. It accumulates dot products and squared norms
in double precision, then divides by explicit norms. Native input vectors need
not be unit-normalized; stored vectors still follow Stage 3's unit-norm policy.
No fast-math flags or approximate ranking are used. Tiny and large finite float32
magnitudes are tested. Invalid values/norms raise ValueError, incorrect types raise
TypeError, and standard C++ exceptions are translated by pybind11.

The bounded min-heap retains at most min(k,N) candidates. Its comparator puts the
worst retained result at the root: lowest score, then largest ID. A better candidate
replaces the root. Only retained candidates are finally sorted by descending score
and ascending ID. Scoring is O(Nd), selection is O(N log k), and heap/result storage
is O(k), separately from duplicate-validation storage. There is no full score array
or full-corpus sort. Empty matrices return no matches after validating the query
and k; k>N returns all rows, while k<1 is invalid.

The binding allocates independent int64 ID and float64 score output arrays after
reacquiring the GIL. Inputs are borrowed without copies. The adapter decodes each
little-endian float32 BLOB through the existing validator and copies it into one
native-endian contiguous `(N,d)` matrix, plus an `(N,)` ID array in ascending ID
order. This costs O(Nd) matrix and O(N) ID memory, with an O(d) row-decoding temporary.
Stored dimensions/type/byte lengths are checked before matrix allocation; missing,
nonfinite, zero/non-unit vectors, corrupt configuration, and orphan relationships
are rejected. Empty unconfigured or indexed corpora are supported.

`retrieval_snapshot(storage)` holds a read transaction open while yielding a
`VectorSnapshot` with configuration, arrays, and `top_k`. Stage 5 can hydrate results
inside that context from the same snapshot. Tests use a WAL database to verify
that an intervening writer commit does not change either vectors or passage text
seen by the reader. `retrieve(storage, query, k)` is a convenience wrapper returning
ranked `(int, float)` pairs; callers needing metadata must use the context manager.
Neither low-level API encodes a query string or validates its model identity;
`search.py` uses the snapshot configuration to load a compatible embedder.
No persistent native index/cache or silent NumPy fallback is implemented.

NumPy tests independently compute float64 cosine and lexicographically sort all
reference scores. Exact ties use ascending IDs; near ties receive no epsilon or
rounding in the C++ comparator. Different summation orders can reorder near ties
across backends. Tests require absolute score error <=1e-5 and matching ordering
outside that tolerance, with separate exact-tie and near-tie fixtures.

`RESEARCHSEEK_SANITIZERS=ON` instruments the extension with AddressSanitizer and
UndefinedBehaviorSanitizer under Clang/GCC. A standalone safety executable also
exercises overflow guards, invalid inputs, and heap boundaries. The test runner
can load a separate instrumented extension, preserving the regular installation.
Sanitizer results and platform coverage are recorded in the roadmap.

## Search service and read-only access

`search(database, query, mode='semantic', top_k=5, cache_dir=...)` returns ranked
`SearchResult` records. A test embedder may be injected for semantic search.
`Storage(..., readonly=True)` requires an existing file, uses a safely escaped
SQLite `mode=ro` URI, and sets `query_only=ON`. It never initializes schema or
changes corpus data. Schema versions, configuration fingerprints, foreign keys,
and FTS5 availability are checked before scoring. Historical Stage 2 corpora can
use keyword search, but semantic search requires immutable model metadata/vectors.

Semantic search opens `retrieval_snapshot`, validates vectors, and loads the exact
stored repository/commit from the local cache once. It compares the reconstructed
configuration with every stored compatibility field, including tokenizer identity,
preprocessing/chunker versions, dimensions, normalization, and package versions.
LF-normalized query text must fit the model token limit. Query vectors pass the
existing shape/finite/unit-norm validator, then native top-k supplies ordered IDs
and scores. Passage lookup stays inside the same snapshot. Empty indexed corpora
skip model loading. No downloads, revision substitutions, or Python ranking fallback
occur. Read transactions remain open through model loading and scoring.

Keyword search imports neither the transformer adapter nor native retrieval. A
separate in-memory FTS5 table tokenizes the query with `unicode61`; `fts5vocab`
extracts distinct normalized terms. Quote each term and OR-connect them, then bind
the resulting expression as a SQL parameter. Thus punctuation cannot become SQL
or FTS syntax, and operator words remain literal tokens. No persistent or temporary
tables are written on the corpus connection. Rank using `bm25(chunks_fts)` ascending,
then chunk ID ascending, with a bound limit. No searchable terms is a query error;
valid unmatched queries return no results. Duplicate query terms are deduplicated.

Both modes resolve IDs to document paths, passage text, and source line spans in
the scoring transaction. Missing metadata, invalid relative paths, inconsistent
text/positions, or nonfinite scores fail the entire request. The result fields are
rank, chunk_id, path, text, start_line, end_line, mode, score, and score_type. Scores
are labeled `cosine_similarity` or `fts5_bm25` and have different scales. No score
fusion, relevance threshold, or retrieval-quality comparison is implemented.

The CLI returns text or JSON (`query`, `mode`, `top_k`, `results`), with diagnostics
on stderr. Invalid input, including overlong semantic or punctuation-only keyword
queries, exits 2; operational errors exit 1; successful/no-match queries exit 0.
Only `benchmark` remains a placeholder. Existing Ubuntu CI discovers Stage 5 tests
through the ordinary offline pytest command; new GitHub results remain pending.

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
evaluation label format/workload sizes (Stage 6). Changes to these
choices must remain consistent with the requirements.
