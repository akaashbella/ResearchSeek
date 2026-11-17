# ResearchSeek

ResearchSeek is a local-first CLI project for semantic and keyword search over
research notes. It is an independent software engineering portfolio project.

Status: Stage 5. Incremental indexing, semantic search, and FTS5 keyword search
are implemented. Benchmarking and quantitative retrieval evaluation remain Stage 6 work.

Intended MVP features:

- Recursively index UTF-8 `.txt` and `.md` notes into deterministic passages.
- Embed passages with a pretrained transformer and persist them in SQLite.
- Skip unchanged notes, replace modified notes, and remove deleted notes.
- Retrieve exact cosine-similarity top-k results using custom C++ and a bounded
  min-heap, exposed to Python through pybind11.
- Offer SQLite FTS5 keyword search and reproducible correctness, performance,
  and retrieval-quality comparisons.

Stack: Python 3.11+, C++17, pybind11, sentence-transformers (`all-MiniLM-L6-v2`),
SQLite/FTS5, CMake, pytest, and NumPy as a test
reference. Development targets macOS, with validation on Ubuntu Linux. No cloud
infrastructure, web interface, approximate search, or external vector database is planned.

Design documents:

- [Requirements and proposed CLI](docs/requirements.md)
- [Architecture and storage design](docs/architecture.md)
- [Stages and exit criteria](docs/roadmap.md)

## Development

Requires Python 3.11+ with venv support and a C++17 compiler. On macOS, use the
Xcode Command Line Tools. On Ubuntu, use a C++ toolchain and Python development
headers/venv support matching the interpreter. Install missing system prerequisites
separately. The setup below only installs Python packages in a virtual environment.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
researchseek --help
python -m researchseek --help
python -c 'from researchseek import _native; print(_native.diagnostic())'
python -c 'from researchseek.diagnostics import check_fts5; check_fts5()'
```

The editable install compiles the extension through scikit-build-core, CMake,
and pybind11. Build isolation downloads CMake and Ninja if needed; the first
install requires network access unless dependencies are already cached/provided.
Runtime dependencies are NumPy, sentence-transformers, and huggingface-hub;
sentence-transformers installs PyTorch and Transformers. The `dev` extra installs pytest.

Python edits are available immediately. After C++ or build-configuration changes,
rerun `python -m pip install -e '.[dev]'` to rebuild the extension. The diagnostic
should print `201703` (C++17). Tests run offline and require no embedding model.

`benchmark` remains a placeholder and returns exit code 1. Invalid CLI usage returns
2; operational errors return 1; successful indexing/search (including no matches) returns 0.

Verified on macOS arm64 with Python 3.11.17 and SQLite 3.53.1. Offline tests and a
separate cached real-model smoke test passed. The existing C++ diagnostic still
works. Ubuntu 24.04 CI passed through Stage 4 with Python 3.11/3.12; Stage 5 changes
are verified locally and await their GitHub Actions run. See the [roadmap](docs/roadmap.md).

## Indexing

Run these commands from the repository root after installation. Model acquisition
is explicit; the first command permits a download into the ignored local cache.
Subsequent commands use cached artifacts without requesting network access.

```sh
researchseek index ./samples --download-model
researchseek index ./samples
researchseek index ./samples --db ./researchseek.sqlite3 --json
researchseek index ./samples --db ./researchseek.sqlite3 --rebuild
```

The default model is `sentence-transformers/all-MiniLM-L6-v2`, pinned initially to
commit `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It produces 384-dimensional
CPU embeddings. `--revision` can select a commit/tag; the resolved commit is stored
and reused on later runs. `--cache-dir` defaults to `.tools/model-cache`. An empty
or incomplete cache produces a provisioning error; it never triggers an implicit
download. To transfer a cache, preserve its Hugging Face directory structure and
symlinks. Cached license/model-card files remain with the model artifacts.

Chunks default to 200 tokens including the two model special tokens and up to 32
content tokens of overlap. The actual fast tokenizer supplies WordPiece offsets
and exact token counts; oversized documents are split before encoding, without
truncation. `--max-tokens`, `--overlap`, and `--batch-size` configure indexing.
Encoded vectors are normalized to unit L2 norm and stored as little-endian float32.

Indexing reports added, updated, unchanged, and deleted documents, total documents
and chunks, and elapsed seconds (including model load for the CLI). Unchanged
documents retain chunk IDs and generate no passage embeddings. A rename is a
deletion plus insertion. Rebuild reports all previous documents deleted and all
replacement documents added.

The database records the resolved root, model/revision/dimension, tokenizer,
chunker/preprocessing configuration, normalization, and package versions. Changes
require `--rebuild`; equal dimensions alone do not establish compatibility.
Stage 2 databases with no model metadata also require rebuilding. Preparation
happens before the write transaction; a failed scan, encoding, validation, or commit
preserves the previous corpus, including during rebuilds.

To run the opt-in model test after provisioning (it never downloads artifacts):

```sh
HF_HUB_OFFLINE=1 python -m pytest --real-model tests/test_real_model.py -q
```

## Searching

```sh
researchseek search "How do transactions preserve a consistent index?"
researchseek search "transaction rollback" --mode keyword
researchseek search "regularization" --mode semantic --top-k 3 --json
researchseek search "rollback" --mode keyword --db ./researchseek.sqlite3 --json
HF_HUB_OFFLINE=1 researchseek search "transaction integrity" --cache-dir .tools/model-cache
```

Semantic search is the default, with top-k=5. It loads the database's exact model
commit once per call, validates model/tokenizer/package compatibility, encodes the
query, and uses the C++ cosine engine. Higher cosine scores rank first. Query line
endings normalize to LF; queries exceeding the model's token limit are rejected.
Search never downloads model files or substitutes another revision. Empty indexed
corpora return no results without loading a model.

Keyword search uses FTS5 BM25: lower scores rank first. It uses the same `unicode61`
tokenizer as the index, including case/diacritic handling. Distinct query terms are
quoted and OR-connected. Quotes, punctuation, and operators do not enable FTS
expression syntax; words such as OR and NOT are searched literally. There is no
stemming, phrase, prefix, or wildcard query syntax. Punctuation-only queries are
invalid. Keyword search works without model artifacts or the native extension.

Both modes break exact score ties by ascending chunk ID. Cosine and BM25 have
different scales and should not be compared numerically. Semantic search returns
the highest-ranked passages without a relevance cutoff; keyword search returns
an empty result set when no terms match. No quality advantage is claimed for either mode.

Search opens an existing database with SQLite `mode=ro` and `query_only=ON`.
Scoring and passage lookup share a read snapshot. Missing/corrupt databases,
incompatible configurations, and invalid passage records produce errors rather
than creating or repairing an index.

Human-readable output shows rank, labeled score, relative path, line range, and
passage text. `--json` writes only one JSON object to stdout:

- Top-level fields: `query` (original input), `mode`, `top_k`, and `results` (array).
- Each result: `rank`, `chunk_id`, `path`, `text`, `start_line`, `end_line`, `mode`,
  `score`, and `score_type` (`cosine_similarity` or `fts5_bm25`).
- Ranks and line numbers are 1-based. Scores are finite numbers. No matches use
  `results: []`; operational errors go to stderr and return exit code 1.

Ordinary tests use fake embedders. The cached-model test now covers indexing,
both search modes, source positions, JSON output, and repeated offline use.

## Document and storage APIs

- `discover_documents(root, database_path=None)` finds `.txt` and `.md` files in
  root-relative order, excluding symlinks and the specified database/sidecars.
- `read_document(root, path)` hashes original bytes, reads strict UTF-8, normalizes
  CRLF/CR to LF, and records byte size and mtime. Detected changes during reading
  raise an error.
- `chunk_text(text, tokenizer, config)` returns source slices with character and
  line positions. Defaults are 200 tokens including special tokens and up to 32
  content tokens of overlap. Indexing uses the actual model tokenizer; the offline
  whitespace tokenizer remains available for deterministic unit tests.
- `Storage` holds corpus configuration, document metadata, chunks, an
  embeddings table, and a trigger-maintained FTS5 index. Configure and modify it
  inside `with storage.transaction():`; an escaping exception rolls back the run.

The five notes under `samples/` cover nested paths, Unicode, long passages, and
LF/CRLF/CR line endings. Tests exercise discovery, chunk positions, persistence,
replacement, deletion, and rollback. Ordinary tests use a deterministic fake
embedder and do not load models or access the network.
See [architecture](docs/architecture.md) for chunking rules and transaction semantics.

## Native retrieval API

`researchseek._native.top_k(matrix, query, chunk_ids, k)` accepts aligned,
C-contiguous NumPy arrays: native float32 `(N, d)` vectors and `(d,)` query, and
int64 `(N,)` IDs. It rejects wrong dtypes/layouts, duplicate IDs, nonfinite values,
zero norms, and incompatible shapes. It never casts or copies input arrays.
Inputs must remain unchanged throughout the call, including through other aliases.

The result is a pair of owned arrays: int64 IDs and float64 cosine scores. Scores
use explicit norms and double accumulation, so inputs need not be normalized.
Results sort by descending score, then ascending ID for exact ties. Positive k
larger than N returns all rows; an empty matrix returns empty results after input
validation. Nearly tied results can differ from other numerical implementations;
the score tolerance is `1e-5`, without rounding scores for ranking.

The C++ scan takes O(Nd + N log k) time. Heap selection and output use O(k) storage;
duplicate-ID validation separately uses O(N) storage. No complete score array is
sorted. Validation holds the GIL; the computational kernel releases it while the
Python array owners remain alive.

For SQLite-backed retrieval, `retrieve(storage, query, k=5)` in
`researchseek.retrieval` returns `(chunk_id, score)` pairs. It decodes and validates
stored little-endian vectors in ascending ID order. Use the context manager when
passages must be fetched from the same read snapshot:

```python
from researchseek.retrieval import retrieval_snapshot
from researchseek.storage import Storage

with Storage("researchseek.sqlite3") as storage:
    with retrieval_snapshot(storage) as snapshot:
        ranked = snapshot.top_k(query, 5)  # query is a matching float32 embedding
        # Fetch passage metadata here, before the read transaction closes.
```

The adapter allocates an O(Nd) vector matrix and O(N) IDs. The search service uses
the snapshot configuration for query-model compatibility. There is no native
fallback or persistent vector/model cache across CLI invocations. Performance
measurements remain pending.

Native tests run with `python -m pytest tests/test_native_search.py tests/test_retrieval.py -q`.
The CMake option `RESEARCHSEEK_SANITIZERS=ON` enables ASan/UBSan for a separate
instrumented build. The standalone kernel safety checks can also be run with Clang:

```sh
mkdir -p .tools
clang++ -std=c++17 -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \
  -Icpp cpp/vector_search.cpp tests/cpp/test_vector_search.cpp -o .tools/native-sanitizer-tests
ASAN_OPTIONS=detect_leaks=0 UBSAN_OPTIONS=halt_on_error=1 .tools/native-sanitizer-tests
```

On macOS, an instrumented Python extension needs the Clang ASan runtime preloaded.
`tests/run_native_sanitizers.py EXTENSION_PATH` runs the native/adapter tests against
that separate extension without replacing the regular installation. See the
[roadmap](docs/roadmap.md) for the actual sanitizer and regression results.
