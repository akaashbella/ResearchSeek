# Roadmap

Stage 0 is complete. Stage 1 is implemented and verified on macOS; Ubuntu
verification is pending. Stage 2 foundations are implemented and verified on macOS.
Stage 3 is implemented and verified on macOS, including cached real-model indexing.
Stage 4 native retrieval is verified on macOS; its Ubuntu exit criterion remains open.
Stage 5 has not started and requires approval.
Later stages depend on the exit criteria listed below.

## Stage 0 — requirements and design (complete)

- Tasks: inspect the repository; document requirements, architecture, roadmap,
  and project overview; record unresolved choices.
- Verification: review scope, CLI semantics, schema, failure guarantees, numeric
  ordering, platform expectations, and experiment claims for consistency.
- Exit: planning documents are complete and consistent, links resolve, and existing
  files are preserved. No application code or dependencies are added.

## Stage 1 — repository and build foundation

Dependency: Stage 0 and Stage 1 approval (both satisfied).

- Tasks: establish Python package layout and pyproject, CMake/pybind11 extension
  scaffold, pytest configuration, ignore rules, and reproducible developer commands.
  Select packaging pins, minimum CMake/compiler versions, and tested Python versions.
  Add macOS/Ubuntu build checks and an FTS5 probe.
- Tests: import/install from a clean build, native binding smoke test, offline
  pytest discovery, missing-FTS5 diagnostics. No retrieval algorithm required yet.
- Exit: documented build/test commands work on macOS and Ubuntu; record actual
  environments and results. Both platforms must be tested to complete this stage.

### Stage 1 verification — October 17, 2026

Implemented the `src/` package, argparse entry points, C++ diagnostic binding,
CMake/scikit-build-core configuration, pytest suite, FTS5 probe, ignore rules,
and development instructions. The CLI commands remain placeholders.

| Check | Actual result |
| --- | --- |
| Initial tools | macOS 27.0.1 arm64; Apple Clang 21.0.0; system Python 3.9.6; no CMake/Ninja on PATH |
| Python provisioning | Python 3.11.17 downloaded into ignored `.tools/python`; `.venv` created with `python3.11 -m venv` |
| Editable install | `.venv/bin/python -m pip install -v -e '.[dev]'` succeeded; native extension compiled and linked |
| Build dependencies | scikit-build-core 0.12.2, pybind11 3.1.0, CMake 4.4.4, Ninja 1.13.2 in pip's isolated build environment |
| Test environment | Python 3.11.17, pip 24.0, pytest 9.1.1, SQLite 3.53.1 |
| Offline tests | `.venv/bin/python -m pytest -q`: 17 passed, 0 failed |
| CLI | `.venv/bin/researchseek --help` returned 0; module/subcommand help and invalid usage covered by tests |
| Native call | Installed `.so` imported; `_native.diagnostic()` returned `201703` (C++17) |
| FTS5 | In-memory FTS5 table creation succeeded; missing-FTS5 diagnostic tested with a stub |
| Ubuntu | Not run; no Linux runtime available in this development environment |

The first dependency download failed because network access was restricted; a
retry with network access succeeded. Python was provisioned locally using a
temporary uv 0.13.1 bootstrap. No system packages were installed or upgraded.
The isolated build emitted a nonfatal `lipo` diagnostic while probing Ninja
(`can't open input file: ninja`). Configuration, compilation, linking, installation,
and native execution all succeeded. No tests failed.

Stage 1's local functionality is verified. The full exit criterion remains open
until the documented clean build and tests pass on Ubuntu. Dependency ranges are
in `pyproject.toml`; the versions above record this run, not a locked environment.

## Stage 2 — documents, chunks, and SQLite

Dependency: Stage 1's local foundation (implemented); Ubuntu verification remains open.

- Tasks: create realistic sample notes; implement discovery, decoding, hashing,
  deterministic chunking, schema/version checks, FTS triggers, transactions, and
  reconciliation primitives. Finalize tokenizer-aware chunk contract and defaults.
  Use an offline tokenizer stub; add deterministic embedder fixtures in Stage 3.
- Tests: nested/Unicode paths, symlink exclusions, empty files/root, LF/CRLF,
  malformed UTF-8, long passages, overlap/spans, hash changes with equal timestamps,
  schema validation, foreign keys, FTS synchronization, cascade delete, rollback.
- Exit: repeatable chunks and valid source spans; add/update/skip/delete storage
  scenarios and injected failures satisfy requirements without a downloaded model.

### Stage 2 verification — October 18, 2026

Implemented `documents.py`, `chunking.py`, and `storage.py`, five sample notes,
and four test modules. `.gitattributes` preserves the CRLF/CR sample bytes.
Default chunks allow 200 total tokens and up to 32 content
tokens of overlap, using the offline whitespace tokenizer. The database schema
is version 1; embeddings are reserved but not generated in this stage.

| Check | Actual result |
| --- | --- |
| Environment | macOS arm64, Python 3.11.17, SQLite 3.53.1, existing editable installation |
| Complete offline suite | `.venv/bin/python -m pytest -q`: 109 passed, 0 failed; includes all 17 unchanged Stage 1 tests |
| Discovery/reads | Supported extensions, Unicode/nested paths, symlink exclusions, empty/missing roots, invalid UTF-8, raw-byte hashes, equal timestamps with changed content, injected read changes/replacement, and scan failures passed |
| Chunking | Determinism, paragraph boundaries, token limits, overlap, long passages, whitespace preservation, source offsets/lines, and invalid tokenizer/configuration checks passed |
| SQLite | Schema creation/reopen, schema rollback/rejection, configuration checks, foreign keys, uniqueness, stable unchanged records, replacement, reconciliation, and empty-document persistence passed |
| FTS5/rollback | MATCH queries confirmed insert/update/delete, cascades, operation savepoints, whole-run rollback, failed rebuild, and synthetic embedding cascade/rollback behavior |
| Sample round-trip | Five copied sample files discovered/read/chunked and persisted in a temporary database; stored text/hashes and source spans checked |
| Native/CLI | `_native.diagnostic()` returned `201703`; `researchseek --help` returned 0 |
| Rebuild | Not needed: only Python modules, tests, sample notes, and docs changed; existing editable/native installation used |
| Ubuntu | Not run; compatibility remains unverified |

Stage 2's functional exit criteria are satisfied locally. No model was downloaded
or loaded, and no runtime dependencies were added. The tokenizer remains a stub,
embeddings may be absent, and callers must supply completed-scan information and
manage the full indexing transaction. Public CLI commands remain placeholders.

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

### Stage 3 verification — November 17, 2026

Implemented `embeddings.py`, `indexing.py`, vector storage operations, and the
functional `index` CLI. Added fake-embedder, vector, indexing, CLI, and opt-in model
tests. Existing chunking tests and C++ sources are unchanged. Two former index
placeholder assertions now check functional help and the required directory argument.

The initial model is `sentence-transformers/all-MiniLM-L6-v2`, pinned to commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its fast tokenizer has a 256-token
limit and two special tokens. Defaults remain 200 total tokens and up to 32 content
tokens of overlap. Model acquisition is opt-in; routine indexing is cache-only.

Preparation now reads/chunks/encodes outside the write transaction. A final source
check and SQLite `data_version` check protect against detected intervening changes.
Configuration, replacements, vectors, deletions, and completion validation commit
together. Preparation holds normalized source text and changed vectors in memory;
it is not a streaming indexer. A filesystem-wide atomic snapshot is not guaranteed.

| Check | Actual result |
| --- | --- |
| Environment | macOS arm64, Python 3.11.17, SQLite 3.53.1 |
| Editable installation | `pip install -e '.[dev]'` succeeded, including native build; `pip check` found no broken requirements |
| Installed runtime versions | sentence-transformers 5.7.0, Transformers 4.57.6, tokenizers 0.22.2, PyTorch 2.15.0, NumPy 2.4.6, huggingface-hub 0.36.2 |
| Ordinary offline suite | `python -m pytest -q`: 182 passed, 1 opt-in model test skipped, 0 failed |
| Existing Stage 2 test files | 109 passed; only the two obsolete index-placeholder expectations changed |
| Cached real-model suite | `HF_HUB_OFFLINE=1 python -m pytest --real-model -q`: 183 passed, 0 failed |
| Initial sample CLI index | Five documents, six chunks, six 384-dimensional vectors |
| Unchanged CLI index | Five unchanged documents, no additions/updates/deletions; fake and real encoder guards confirm zero new passage embeddings |
| Modified/deleted sample copies | One updated, one deleted, three unchanged; four documents/five chunks remained |
| Index/rebuild failure | Malformed UTF-8 returned exit 1; prior configuration, documents, chunks, and vectors unchanged |
| Successful rebuild | Replaced all four remaining documents; complete vector/chunk relationships validated |
| Vector checks | Little-endian float32 serialization, shape/dimension checks, finite/unit-norm checks, and cascades passed; real stored norms ranged from 0.99999994 to 1.0 |
| Injected failures | Encoding, vector validation, database writes/commit, rebuild, source changes, and intervening database commits rejected without partial index updates |
| CLI/native | Root/index help worked; native diagnostic returned `201703`; operational errors returned 1 and invalid usage returned 2 |
| Ubuntu | Not run; compatibility remains unverified |

The real model was explicitly provisioned into the ignored project cache before
offline verification. No artifacts, local databases, or model weights are tracked.
Dependency bounds are in `pyproject.toml`; the recorded versions are this run's
environment, not a lockfile. Model/revision/tokenizer/chunker/root/package changes
require an explicit rebuild, including conversion from Stage 2-only corpora.

Stage 3's functional exit criteria are satisfied locally. Retrieval, public keyword
search, and benchmarking remain unimplemented. Stage 4 should add the C++ cosine
kernel and bounded top-k heap, validate the pybind11 array boundary, and compare
scores/ordered results with the NumPy reference before integrating retrieval.

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

### Stage 4 verification — November 17, 2026

Implemented the independent C++ kernel/header, strict NumPy binding, and Python
`retrieval.py` snapshot adapter. Added independent NumPy reference tests, native
edge cases, persisted-vector integration tests, and standalone/instrumented safety
runners. Stage 3 application code, schema, model configuration, and CLI are unchanged.

| Check | Actual result |
| --- | --- |
| Toolchain | macOS arm64, Python 3.11.17, NumPy 2.4.6, Apple Clang 21.0.0 (clang-2100.3.34.2), CMake 4.4.5, pybind11 3.1.0, scikit-build-core 0.12.2 |
| Editable rebuild | `python -m pip install --no-deps -v -e .` succeeded; native diagnostic returned `201703` (C++17). The known nonfatal Ninja/lipo probe warning remains |
| Native/adapter tests | `python -m pytest tests/test_native_search.py tests/test_retrieval.py -q`: 112 passed |
| NumPy comparison | Nine seeded normalized fixtures, dimensions 1/7/384: maximum observed absolute score error 3.0531133177191805e-16; ordered IDs matched. Separate fixtures exercise exact and near ties |
| Ordinary offline suite | `python -m pytest -q`: 294 passed, 1 opt-in real-model test skipped, 0 failed |
| Complete cached-model suite | `HF_HUB_OFFLINE=1 python -m pytest --real-model -q`: 295 passed, 0 failed; includes all 183 pre-existing cases |
| Index CLI | Indexed the five samples into a temporary local database: five documents, six chunks. Help and native diagnostic passed |
| Real vector retrieval | Cached model query against six stored 384-dimensional vectors: native top-5 IDs/scores agreed with NumPy |
| Snapshot/corruption checks | Writer commit under WAL preserved reader vectors/text; missing, malformed, nonfinite, zero/non-unit vectors, corrupt dimensions/configuration, and orphan IDs were rejected |
| Standalone sanitizers | Clang C++17 executable with `-fsanitize=address,undefined -fno-omit-frame-pointer` passed kernel safety checks; ASAN_OPTIONS=detect_leaks=0, UBSAN_OPTIONS=halt_on_error=1 |
| Instrumented extension | Separate wheel with RESEARCHSEEK_SANITIZERS=ON loaded with the Clang ASan runtime preloaded; all 112 native/adapter tests passed without ASan/UBSan findings. Regular installation was preserved |
| Ubuntu | Not run; compatibility remains unverified |

The first targeted pytest run reported 109 passed/1 failed: the malformed-length
fixture was blocked by the schema's dimension trigger before reaching retrieval.
The fixture now deliberately bypasses that trigger to seed corruption. The final
suite passes. LeakSanitizer and ThreadSanitizer were not run.

Heap inspection confirms at most min(k,N) candidates and no full score sort.
Selection/output use O(k) storage; duplicate-ID validation uses a separate O(N)
set, and the Python adapter allocates an O(Nd) matrix. Inputs are borrowed during
the native call and must not be mutated concurrently. No performance gain is claimed.

Local Stage 4 functionality and safety checks are complete. The formal exit
criterion requiring both macOS and Ubuntu remains open. Stage 5 should add matching
query-model loading, snapshot-consistent passage lookup, semantic/FTS5 search
services, and text/JSON search CLI output. Benchmarking remains Stage 6 work.

### Ubuntu CI configuration — November 17, 2026

`.github/workflows/ci.yml` runs on pushes to `main` and pull requests targeting
`main`. Each Ubuntu 24.04 job starts from a clean checkout with Python 3.11 or
3.12, installs `build-essential`, CMake, and Ninja, then runs:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
researchseek --help
```

The editable install builds the native C++17 extension. A separate check verifies
its compiled-module import, `diagnostic() == 201703`, and SQLite FTS5 support.
The workflow uses official `actions/checkout@v7.0.1` and
`actions/setup-python@v7.1.0`, read-only contents permission, no persisted checkout
credentials, a 25-minute timeout per job, and pip caching keyed by `pyproject.toml`.
No repository secrets or service containers are required.

`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`, and
`DO_NOT_TRACK=1` disable model acquisition and telemetry. PyPI dependency downloads
remain allowed. The real-model test stays skipped without `--real-model`.

Local validation on macOS Python 3.11: PyYAML parsing, duplicate-key/structure
checks, Bash syntax for every run step, and embedded Python syntax passed. The
native/FTS5 check and CLI help also executed successfully. `actionlint` was not
installed. The offline suite with the CI model environment flags passed:
294 tests passed, one opt-in test skipped, zero failures.

GitHub execution is pending. Neither Ubuntu matrix job nor Python 3.12 has been
verified locally. Both jobs must pass in GitHub Actions before recording Ubuntu
verification or closing the cross-platform exit criteria. No application behavior
was changed, and Stage 5 has not started.

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
