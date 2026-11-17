# ResearchSeek

ResearchSeek is a local-first CLI project for semantic and keyword search over
research notes. It is an independent software engineering portfolio project.

Status: Stage 2 foundations. Document discovery, normalized reads, deterministic
chunking, and transactional SQLite/FTS5 storage are implemented as Python APIs.
The CLI remains a scaffold; indexing, retrieval, and benchmarking are not implemented.

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
There are no runtime dependencies yet. The `dev` extra installs pytest.

Python edits are available immediately. After C++ or build-configuration changes,
rerun `python -m pip install -e '.[dev]'` to rebuild the extension. The diagnostic
should print `201703` (C++17). Tests run offline and require no embedding model.

The `index`, `search`, and `benchmark` commands currently provide help and return
exit code 1 when invoked without `--help`, explaining that they are not implemented.
Invalid CLI usage returns 2. The full command examples in the requirements are
for later stages.

Verified on macOS arm64 with Python 3.11.17 and SQLite 3.53.1: all 109 tests passed,
including the 17 Stage 1 tests. The existing C++ diagnostic still works. Ubuntu has
not been tested. See the [roadmap](docs/roadmap.md) for verification details.

## Document and storage APIs

- `discover_documents(root, database_path=None)` finds `.txt` and `.md` files in
  root-relative order, excluding symlinks and the specified database/sidecars.
- `read_document(root, path)` hashes original bytes, reads strict UTF-8, normalizes
  CRLF/CR to LF, and records byte size and mtime. Detected changes during reading
  raise an error.
- `chunk_text(text, tokenizer, config)` returns source slices with character and
  line positions. Defaults are 200 tokens including special tokens and up to 32
  content tokens of overlap. The offline whitespace tokenizer is a stub; a real
  transformer tokenizer will replace it in Stage 3.
- `Storage` holds corpus configuration, document metadata, chunks, a reserved
  embeddings table, and a trigger-maintained FTS5 index. Configure and modify it
  inside `with storage.transaction():`; an escaping exception rolls back the run.

The five notes under `samples/` cover nested paths, Unicode, long passages, and
LF/CRLF/CR line endings. Tests exercise discovery, chunk positions, persistence,
replacement, deletion, and rollback. No model is loaded or embeddings generated.
See [architecture](docs/architecture.md) for chunking rules and transaction semantics.
