# ResearchSeek

ResearchSeek is a planned local-first CLI for semantic and keyword search over
research notes. It is an independent software engineering portfolio project.

Status: Stage 0 (planning). There is no implementation or benchmark data yet.

Intended MVP features:

- Recursively index UTF-8 `.txt` and `.md` notes into deterministic passages.
- Embed passages with a pretrained transformer and persist them in SQLite.
- Skip unchanged notes, replace modified notes, and remove deleted notes.
- Retrieve exact cosine-similarity top-k results using custom C++ and a bounded
  min-heap, exposed to Python through pybind11.
- Offer SQLite FTS5 keyword search and reproducible correctness, performance,
  and retrieval-quality comparisons.

Proposed stack: Python 3.11+, C++17, pybind11, sentence-transformers (initial
candidate: `all-MiniLM-L6-v2`), SQLite/FTS5, CMake, pytest, and NumPy as a test
reference. Development targets macOS, with validation on Ubuntu Linux. No cloud
infrastructure, web interface, approximate search, or external vector database is planned.

Design documents:

- [Requirements and proposed CLI](docs/requirements.md)
- [Architecture and storage design](docs/architecture.md)
- [Stages and exit criteria](docs/roadmap.md)

The next stage covers packaging, the native build, and offline test setup.
Implementation is pending Stage 1 approval; the CLI examples are not runnable yet.
