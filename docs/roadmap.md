# Roadmap

Stage 0 is the current stage. Implementation is pending Stage 1 approval.
Later stages depend on the exit criteria listed below.

## Stage 0 — requirements and design (current)

- Tasks: inspect the repository; document requirements, architecture, roadmap,
  and project overview; record unresolved choices.
- Verification: review scope, CLI semantics, schema, failure guarantees, numeric
  ordering, platform expectations, and experiment claims for consistency.
- Exit: planning documents are complete and consistent, links resolve, and existing
  files are preserved. No application code or dependencies are added.

## Stage 1 — repository and build foundation

Dependency: Stage 0 and Stage 1 approval.

- Tasks: establish Python package layout and pyproject, CMake/pybind11 extension
  scaffold, pytest configuration, ignore rules, and reproducible developer commands.
  Select packaging pins, minimum CMake/compiler versions, and tested Python versions.
  Add macOS/Ubuntu build checks and an FTS5 probe.
- Tests: import/install from a clean build, native binding smoke test, offline
  pytest discovery, missing-FTS5 diagnostics. No retrieval algorithm required yet.
- Exit: documented build/test commands work on macOS and Ubuntu; record actual
  environments and results. Both platforms must be tested to complete this stage.

## Stage 2 — documents, chunks, and SQLite

Dependency: Stage 1.

- Tasks: create realistic sample notes; implement discovery, decoding, hashing,
  deterministic chunking, schema/version checks, FTS triggers, transactions, and
  reconciliation primitives. Finalize tokenizer-aware chunk contract and defaults.
  Use a tokenizer stub and deterministic embedder fixtures until Stage 3.
- Tests: nested/Unicode paths, symlink exclusions, empty files/root, LF/CRLF,
  malformed UTF-8, long passages, overlap/spans, hash changes with equal timestamps,
  schema validation, foreign keys, FTS synchronization, cascade delete, rollback.
- Exit: repeatable chunks and valid source spans; add/update/skip/delete storage
  scenarios and injected failures satisfy requirements without a downloaded model.

## Stage 3 — embeddings and incremental indexing

Dependency: Stage 2.

- Tasks: implement the embedder interface and sentence-transformers adapter,
  batching/normalization, model/configuration fingerprints, offline provisioning,
  transactional index/rebuild service, counts, and the `index` CLI.
  Pin the model revision and validate chunks against the actual tokenizer limit.
- Tests: fake embedder call counts, same-dimension incompatible models, malformed
  vectors, changed/deleted/renamed files, root mismatch, partial failures, failed
  rebuild, concurrent-edit detection; separate real-model cached/offline smoke test.
- Exit: initial and incremental indexing work end to end; unchanged documents
  generate no new passage embeddings; all failure cases preserve the prior corpus.

## Stage 4 — exact C++ cosine and bounded top-k

Dependency: Stage 1; use the vector/storage contract finalized in Stage 3 for
integration and stage completion.

- Tasks: implement cosine kernel, deterministic bounded min-heap, pybind11
  validation/ownership rules, GIL release, and Python vector snapshot adapter.
  Finalize normalization tolerance and copy policy. Do not add approximate search.
- Tests: seeded NumPy reference comparisons, exact/near ties, negative scores,
  empty matrix, k boundaries, zero/nonfinite inputs, dimensions/layout, repeated
  calls and input lifetime. Exercise native code under a sanitizer where available.
- Exit: score tolerance and ordered-ID acceptance criteria pass on both platforms;
  selection retains O(k) candidates by code inspection. Report sanitizer availability
  and actual outcomes; no performance improvement claim is required.

## Stage 5 — search CLI and keyword baseline

Dependency: Stages 2–4.

- Tasks: semantic query encoding, snapshot-consistent search and passage lookup,
  FTS5 literal term handling/BM25, text/JSON results, complete help and error/exit behavior.
- Tests: semantic and keyword integration, punctuation/quoting/Unicode queries,
  deterministic ordering, empty results, k validation, missing native/model/DB,
  stale-result prevention after index updates, and realistic sample walkthrough.
- Exit: documented `index`/`search` examples execute; results include usable source
  citations and labeled scores; offline cached operation and failure diagnostics
  are demonstrated. Ordinary tests remain independent of network/model downloads.

## Stage 6 — reproducible evaluation and MVP review

Dependency: Stage 5.

- Tasks: finalize at least 20 graded queries and label format; implement `benchmark`
  command, seeded workloads, raw JSON reports, metric calculations, warmup/repeat
  protocol, and environment capture. Document corpus/model fingerprints and commands.
  Measure initial/unchanged/modified indexing and semantic/keyword search quality
  and latency. Add clean Ubuntu/macOS validation and update README with actual setup.
- Tests: hand-calculated Recall@5/nDCG@5 fixtures, percentile/report validation,
  reproducible workloads, mismatched labels/configuration rejection, existing
  report protection, and CLI benchmark smoke tests using fakes.
- Exit: publish measured results with raw samples, p50/p95 timing boundaries,
  indexing throughput and both quality metrics; discuss limitations and negative
  findings. Record evidence for all MVP acceptance criteria. Distinguish estimates
  from measurements and support performance claims with results.

## After the MVP

Base further work on measured bottlenecks and user needs. Larger corpora,
alternative chunking/models, reranking, and hybrid retrieval require a separate proposal.
