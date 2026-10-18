# Requirements

Stage 0 specification. The requirements and CLI examples describe planned behavior;
none is implemented or measured yet.

## MVP scope

One local corpus root per SQLite database, one compatible embedding configuration
per database, synchronous indexing, and exact in-memory vector retrieval. Index
regular UTF-8 `.txt` and `.md` files recursively (case-insensitive extensions).
Do not follow symlink files or directories. Treat Markdown as plain text in the
MVP. Exclude the database and its sidecars from scanning. Multiple independent
corpora use separate databases.

PDF parsing, OCR, other encodings, file watchers, concurrent writers, hybrid score
fusion, reranking, GPU-specific optimization, cloud infrastructure, web UI, LSH,
approximate nearest-neighbor indexes, and external vector databases are excluded.

## Functional requirements

1. **Discovery and chunking.** Store root-relative paths. Decode UTF-8 strictly;
   normalize line endings to LF without removing other text. Produce repeatable
   nonempty chunks with order, normalized-text character offsets, and line spans.
   Empty or whitespace-only files have document metadata but no chunks. Chunk
   parameters and algorithm version are persisted; tokenizer-aware size limits
   must prevent silent embedding truncation.
2. **Embeddings.** Use sentence-transformers with initial model candidate
   `sentence-transformers/all-MiniLM-L6-v2`. Record resolved model revision,
   dimension, preprocessing and chunker versions, relevant package versions, and
   normalization policy. Store finite L2-normalized float32 vectors; reject zero
   vectors. Apply the same embedding configuration to queries and passages.
   Incompatible configurations require an explicit rebuild.
3. **Persistence.** SQLite stores corpus configuration, metadata, SHA-256 raw-byte
   content hashes, chunks, and embeddings. FTS5 indexes the same chunk text.
   Foreign keys and transactions enforce consistency; schema versions are checked.
4. **Incremental indexing.** Hash every supported file on each run; timestamps
   alone cannot establish equality. Skip embedding unchanged content. Replace all
   chunks for changed documents and delete records for missing documents. A rename
   is deletion plus insertion. Report added, updated, unchanged, deleted, document,
   and chunk counts with elapsed time.
5. **Failure handling.** A scan, read, decode, embedding, or database failure
   aborts the run with a nonzero exit and leaves the prior committed corpus
   intact. Only a completed scan may trigger deletion reconciliation. An absent
   root is an error; an existing empty root removes all indexed documents.
   Rebuilds have the same rollback guarantee. Report the offending path/context.
6. **Semantic retrieval.** Implement exact cosine scoring in C++17, scanning all
   vectors and retaining top-k in a bounded min-heap. Return descending scores;
   break exact score ties by ascending chunk ID. Validate dimensions, finite
   inputs, vector norms, and k. An empty corpus returns no matches; k greater
   than the corpus size returns all chunks; k less than one is an error.
7. **Keyword baseline.** Use SQLite FTS5 BM25 ranking with ascending chunk ID
   as the secondary order. Treat CLI query words as literal OR-connected terms
   by default; FTS expression syntax is not exposed in the MVP. Parameterize SQL
   and safely quote FTS terms. Reject queries with no searchable terms.
8. **CLI.** Provide `index`, `search`, and `benchmark`, help text, readable output,
   and machine-readable JSON. Search returns path, chunk ID, passage, line span,
   rank, score, and mode. Label cosine and BM25 scores distinctly; they are not
   comparable scales. Reject empty queries, invalid options, incompatible models,
   missing databases, and unavailable FTS5/native support with actionable errors.
9. **Experiments.** Benchmark correctness, vector-kernel and end-to-end search
   latency, indexing throughput, and semantic versus keyword quality. Use checked-in
   sample notes and query/relevance labels plus documented seeded synthetic vector
   workloads. Publish raw machine-readable results and reproducible commands.

## Nonfunctional requirements

- Support Python 3.11+ and C++17 on macOS and Ubuntu Linux, with tested versions
  stated when implementation exists. Fail clearly if SQLite lacks FTS5.
- Keep document content, embeddings, and queries local. An initial model download
  is the only anticipated network need; cached or user-provided model artifacts
  must permit fully offline indexing and search. No telemetry.
- Use CPU as the reproducibility baseline. Record hardware and thread settings;
  do not assume bitwise-identical transformer output across platforms.
- Bound top-k auxiliary storage to O(k); exact retrieval scans O(Nd) data
  and holds O(Nd) vector memory. Report corpus size and resource limits.
  Batch embedding generation.
- Keep ordinary tests deterministic and offline. Pin the experiment environment
  and resolved model revision when publishing results.

## Proposed CLI

Default database location is `./researchseek.sqlite3`; `--db` overrides it on each
command. `index` stores the resolved corpus root; `search` needs only the database.

```sh
researchseek index ./notes --db ./researchseek.sqlite3
researchseek index ./notes --db ./researchseek.sqlite3 --rebuild
researchseek search "methods for reducing model overfitting" --mode semantic --top-k 5
researchseek search "regularization dropout" --mode keyword --top-k 5 --json
researchseek benchmark --db ./researchseek.sqlite3 --queries ./benchmarks/queries.json --output ./results/run.json
```

Semantic search is the default mode and top-k defaults to 5. `--rebuild` explicitly
replaces the corpus/configuration atomically; changing roots also requires it.
Chunking options will be finalized in Stage 2 and model selection options in Stage 3.
No-match search is successful with an empty result set. Normal success exits 0, operational
failure exits 1, and invalid CLI usage exits 2. A benchmark output path must not
silently overwrite an existing report.

## Measurable acceptance criteria

- Fixture tree discovery includes supported files at multiple depths, excludes
  symlinks/unsupported files, and handles Unicode paths. Two runs on the same
  content/configuration produce identical chunk text, order, and boundaries.
- Incremental fixtures exercise add, unchanged, modify, rename, delete, and empty
  root. Unchanged runs make zero passage-embedding calls and preserve chunk IDs.
  Injected failures leave document, chunk, embedding, and FTS state unchanged.
- Every stored chunk has exactly one valid embedding and corresponding FTS row;
  deleted/replaced chunks leave no stale semantic or keyword results. Stored
  vector dimensions and byte lengths agree with database configuration.
- Against seeded NumPy float64 cosine reference scores, C++ absolute score error
  is at most `1e-5` on normalized float32 fixtures. Ordered IDs match when scores
  differ by more than that tolerance; exact ties use ascending chunk ID. Cover
  negative scores, duplicates, k=1, k>N, empty input, malformed shapes, nonfinite
  values, zero norms, and repeated calls. Near ties are reported separately.
- CLI checks cover all commands, help, JSON, exit codes, and error paths. A
  real-model opt-in smoke test indexes samples, performs both search modes, and
  demonstrates offline reuse of cached artifacts.
- A clean build and offline unit suite pass on both macOS and Ubuntu before MVP
  completion. Record actual platform/toolchain versions and check outcomes.
- Publish at least 20 curated queries with graded relevance labels, including
  paraphrase and exact-term cases. Report Recall@5 and nDCG@5 for both modes,
  their paired differences, and failure cases. Report negative results; no minimum
  semantic improvement is required.
- Report indexing seconds and chunks/second for initial, unchanged, and modified
  runs; separate cold model acquisition/load from embedding time. For latency,
  record at least 5 warmups and 30 measured repetitions per workload, p50/p95,
  vector-kernel versus CLI/end-to-end timing, raw samples, and environment metadata.
  Stage 0 sets no latency or throughput target.
