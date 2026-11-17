# ResearchSeek

ResearchSeek is a local-first CLI project for semantic and keyword search over
research notes. It is an independent software engineering portfolio project.

Status: Stage 3. The `index` command generates transformer embeddings and updates
a local SQLite corpus incrementally. Semantic retrieval, keyword search commands,
and benchmarking are not implemented yet.

Intended MVP features:

- Recursively index UTF-8 `.txt` and `.md` notes into deterministic passages.
- Embed passages with a pretrained transformer and persist them in SQLite.
- Skip unchanged notes, replace modified notes, and remove deleted notes.
- Retrieve exact cosine-similarity top-k results using custom C++ and a bounded
  min-heap, exposed to Python through pybind11.
- Offer SQLite FTS5 keyword search and reproducible correctness, performance,
  and retrieval-quality comparisons.

Planned stack: Python 3.11+, C++17, pybind11, sentence-transformers (initial
candidate: `all-MiniLM-L6-v2`), SQLite/FTS5, CMake, pytest, and NumPy as a test
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

`search` and `benchmark` remain placeholders and return exit code 1. Invalid CLI
usage returns 2; operational errors return 1; successful indexing returns 0.

Verified on macOS arm64 with Python 3.11.17 and SQLite 3.53.1. Offline tests and a
separate cached real-model smoke test passed. The existing C++ diagnostic still
works. Ubuntu has not been tested. See the [roadmap](docs/roadmap.md) for results.

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
