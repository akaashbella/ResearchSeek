from pathlib import Path
import shutil
import sqlite3
import struct
import subprocess
import sys

import numpy as np
import pytest

from researchseek import embeddings
from researchseek.indexing import index_corpus
from researchseek.search import QueryError, keyword_expression, search
from researchseek.storage import Storage


@pytest.fixture
def corpus(tmp_path, fake_embedder):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("alpha beta common\n\nsecond paragraph\n")
    (root / "b.txt").write_text("gamma delta common\n")
    (root / "研究.md").write_text("OR NOT AND NEAR café 研究 unicode\n")
    path = tmp_path / "index #1.sqlite3"
    index_corpus(root, path, fake_embedder)
    return root, path, fake_embedder


def test_semantic_query_encoding_native_ranking_and_positions(corpus, monkeypatch):
    from researchseek import _native
    root, path, embedder = corpus
    query = (root / "a.md").read_text()
    calls = []
    original = _native.top_k
    def spy(matrix, vector, ids, k):
        calls.append((matrix.shape, vector.shape, ids.tolist(), k))
        return original(matrix, vector, ids, k)
    monkeypatch.setattr(_native, "top_k", spy)
    embedder.calls.clear()
    result = search(path, query, embedder=embedder, top_k=100)
    assert len(result) == 3 and result[0].path == "a.md"
    assert result[0].score == pytest.approx(1)
    assert result[0].score_type == "cosine_similarity" and result[0].mode == "semantic"
    assert [item.rank for item in result] == [1, 2, 3]
    assert embedder.calls == [([query], 32)]
    assert calls[0][:2] == ((3, 3), (3,)) and calls[0][3] == 100
    assert calls[0][2] == sorted(calls[0][2])
    for item in result:
        assert item.text == (root / item.path).read_text()
        assert item.start_line == 1
        assert item.end_line == item.text[:-1].count("\n") + 1
    assert search(path, query, embedder=embedder, top_k=100) == result


def test_exact_ties_and_k(tmp_path, fake_embedder):
    root = tmp_path / "notes"
    root.mkdir()
    for name in ("c.md", "a.md", "b.md"):
        (root / name).write_text("identical passage")
    path = tmp_path / "ties.sqlite3"
    index_corpus(root, path, fake_embedder)
    for mode in ("semantic", "keyword"):
        result = search(path, "identical passage", mode=mode, embedder=fake_embedder, top_k=100)
        assert [item.chunk_id for item in result] == sorted(item.chunk_id for item in result)
        assert len(search(path, "identical passage", mode=mode, embedder=fake_embedder, top_k=1)) == 1


def test_recorded_model_loaded_once_without_download(corpus, monkeypatch):
    _, path, embedder = corpus
    loads = []
    def load(model_id, **kwargs):
        loads.append((model_id, kwargs))
        return embedder
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", load)
    assert search(path, "query", cache_dir=Path("missing-cache"))
    assert loads == [(embedder.model_id, {"revision": embedder.model_revision,
                     "cache_dir": Path("missing-cache"), "allow_download": False})]


@pytest.mark.parametrize("change", ["model", "revision", "dimension", "tokenizer", "packages"])
def test_incompatible_query_embedder_rejected(corpus, change):
    _, path, embedder = corpus
    if change == "model": embedder.model_id = "another-model"
    elif change == "revision": embedder.model_revision = "b" * 40
    elif change == "dimension": embedder.dimension = 2
    elif change == "tokenizer": embedder.tokenizer.identity = "other-tokenizer"
    else: embedder.package_versions = {"test-embedder": "different"}
    embedder.calls.clear()
    with pytest.raises(ValueError, match="incompatible"):
        search(path, "query", embedder=embedder)
    assert embedder.calls == []


def test_missing_cache_does_not_break_keyword(corpus, monkeypatch):
    _, path, _ = corpus
    def unavailable(*args, **kwargs):
        raise embeddings.EmbeddingError("Model missing; provision with --download-model")
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", unavailable)
    with pytest.raises(embeddings.EmbeddingError, match="provision"):
        search(path, "query")
    assert search(path, "alpha", mode="keyword")


@pytest.mark.parametrize("matrix", [np.array([[np.nan, 0, 0]]), np.zeros((1, 3)),
    np.ones((1, 4)), np.ones((2, 3)), np.array([[3, 4, 0]])])
def test_invalid_query_vectors(corpus, monkeypatch, matrix):
    _, path, embedder = corpus
    monkeypatch.setattr(embedder, "encode", lambda *args, **kwargs: matrix)
    with pytest.raises(ValueError):
        search(path, "query", embedder=embedder)


def test_query_token_limit(corpus):
    _, path, embedder = corpus
    with pytest.raises(QueryError, match="token limit"):
        search(path, "word " * 300, embedder=embedder)


def test_query_line_endings_use_existing_preprocessing(corpus):
    _, path, embedder = corpus
    embedder.calls.clear()
    search(path, "alpha\r\nbeta\rcommon", embedder=embedder)
    assert embedder.calls == [(["alpha\nbeta\ncommon"], 32)]


@pytest.mark.parametrize("query,expected", [("alpha", {"a.md"}), ("alpha gamma", {"a.md", "b.txt"}),
    ('"alpha": (gamma)*', {"a.md", "b.txt"}), ("OR", {"研究.md"}), ("NOT AND NEAR", {"研究.md"}),
    ("cafe", {"研究.md"}), ("café 研究", {"研究.md"}), ("unmatchedxyz", set()),
    ('" OR 1=1 --', {"研究.md"})])
def test_literal_keyword_queries(corpus, query, expected):
    _, path, _ = corpus
    results = search(path, query, mode="keyword")
    assert {item.path for item in results} == expected
    assert all(item.score_type == "fts5_bm25" and item.mode == "keyword" for item in results)
    assert [(item.score, item.chunk_id) for item in results] == sorted((item.score, item.chunk_id) for item in results)


@pytest.mark.parametrize("query", ["...", '"(){}*:^+-', "💡🚀", "\x00"])
def test_no_searchable_keyword_terms(corpus, query):
    with pytest.raises(QueryError, match="searchable terms"):
        search(corpus[1], query, mode="keyword")


def test_keyword_expression_is_quoted_deduplicated_unicode61():
    assert keyword_expression('OR alpha alpha "CAFÉ"') == '"alpha" OR "cafe" OR "or"'
    assert keyword_expression("cafe\u0301") == '"cafe"'


def test_bm25_better_scores_come_first(tmp_path, fake_embedder):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "a.md").write_text("term term term")
    (root / "b.md").write_text("term with several other unrelated words here")
    path = tmp_path / "scores.sqlite3"
    index_corpus(root, path, fake_embedder)
    results = search(path, "term", mode="keyword")
    assert [item.path for item in results] == ["a.md", "b.md"]
    assert results[0].score < results[1].score < 0


def test_keyword_does_not_import_models_or_native(corpus):
    script = '''
import importlib.abc, sys
class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'researchseek.embeddings', 'researchseek.retrieval', 'researchseek._native',
                         'sentence_transformers', 'transformers', 'torch'}:
            raise AssertionError('Forbidden keyword-search import: ' + fullname)
sys.meta_path.insert(0, NoModelImports())
from researchseek.search import search
assert search(sys.argv[1], 'alpha', mode='keyword')
from researchseek.cli import main
assert main(['search', 'alpha', '--mode', 'keyword', '--db', sys.argv[1], '--json']) == 0
'''
    result = subprocess.run([sys.executable, "-c", script, str(corpus[1])], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["semantic", "keyword"])
def test_updates_and_deletions(corpus, mode):
    root, path, embedder = corpus
    (root / "a.md").write_text("replacement vocabulary")
    (root / "b.txt").unlink()
    index_corpus(root, path, embedder)
    results = search(path, "replacement vocabulary", mode=mode, top_k=100, embedder=embedder)
    assert results[0].path == "a.md" and results[0].text == "replacement vocabulary"
    assert all(item.path != "b.txt" and "alpha" not in item.text for item in results)
    assert search(path, "alpha", mode="keyword") == []


@pytest.mark.parametrize("mode", ["semantic", "keyword"])
def test_search_does_not_write_database(corpus, mode):
    _, path, embedder = corpus
    before, modified = path.read_bytes(), path.stat().st_mtime_ns
    search(path, "alpha", mode=mode, embedder=embedder)
    assert path.read_bytes() == before and path.stat().st_mtime_ns == modified
    with Storage(path, readonly=True) as store:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.connection.execute("DELETE FROM chunks")


@pytest.mark.parametrize("mode", ["semantic", "keyword"])
def test_missing_database_not_created(tmp_path, mode):
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(FileNotFoundError):
        search(path, "query", mode=mode)
    assert not path.exists()


@pytest.mark.parametrize("kind", ["empty-file", "invalid-file", "unsupported", "unconfigured"])
def test_invalid_database(tmp_path, kind):
    path = tmp_path / "bad.sqlite3"
    if kind == "unconfigured":
        with Storage(path): pass
    elif kind == "unsupported":
        with sqlite3.connect(path) as connection: connection.execute("PRAGMA user_version=999")
    else: path.write_bytes(b"invalid database" if kind == "invalid-file" else b"")
    before = path.read_bytes()
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        search(path, "query", mode="keyword")
    assert path.read_bytes() == before


def test_corrupt_vectors_semantic_only(corpus):
    _, path, embedder = corpus
    with Storage(path) as store, store.transaction():
        store.connection.execute("UPDATE embeddings SET vector=?", (struct.pack("<fff", np.nan, 0, 0),))
    with pytest.raises(ValueError, match="embedding"):
        search(path, "query", embedder=embedder)
    assert search(path, "alpha", mode="keyword")


def test_missing_metadata_is_rejected(corpus, monkeypatch):
    from researchseek import _native
    _, path, embedder = corpus
    monkeypatch.setattr(_native, "top_k", lambda *args: (np.array([999]), np.array([0.5])))
    with pytest.raises(ValueError, match="metadata"):
        search(path, "query", embedder=embedder)


def test_missing_native_extension(corpus, monkeypatch):
    from researchseek import retrieval
    def unavailable(name):
        raise ImportError("missing native extension")
    monkeypatch.setattr(retrieval, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="Native retrieval is unavailable"):
        search(corpus[1], "query", embedder=corpus[2])


def test_invalid_source_positions(corpus):
    _, path, _ = corpus
    with Storage(path) as store, store.transaction():
        store.connection.execute("UPDATE chunks SET end_line=end_line+1")
    with pytest.raises(ValueError, match="source positions"):
        search(path, "alpha", mode="keyword")


def test_fts_missing_support_is_actionable(corpus, monkeypatch):
    actual_connect = sqlite3.connect
    def missing(*args, **kwargs):
        if args[0] == ":memory:": raise sqlite3.OperationalError("no such module: fts5")
        return actual_connect(*args, **kwargs)
    monkeypatch.setattr(sqlite3, "connect", missing)
    with pytest.raises(RuntimeError, match="FTS5"):
        search(corpus[1], "alpha", mode="keyword")


def test_semantic_snapshot_during_writer_commit(corpus, monkeypatch):
    from researchseek import _native
    root, path, embedder = corpus
    with Storage(path) as store: store.connection.execute("PRAGMA journal_mode=WAL")
    original = _native.top_k
    def writing_top_k(*args):
        (root / "a.md").write_text("new committed contents")
        index_corpus(root, path, embedder)
        return original(*args)
    monkeypatch.setattr(_native, "top_k", writing_top_k)
    results = search(path, "alpha beta common\n\nsecond paragraph\n", embedder=embedder)
    assert results[0].text == "alpha beta common\n\nsecond paragraph\n"
    with Storage(path) as store: assert store.chunks()[0]["text"] == "new committed contents"


def test_keyword_snapshot_during_writer_commit(corpus, monkeypatch):
    from researchseek import search as search_module
    root, path, embedder = corpus
    with Storage(path) as store: store.connection.execute("PRAGMA journal_mode=WAL")
    original = search_module._passage
    def writing_lookup(*args):
        (root / "a.md").write_text("new committed contents")
        index_corpus(root, path, embedder)
        return original(*args)
    monkeypatch.setattr(search_module, "_passage", writing_lookup)
    assert search(path, "alpha", mode="keyword")[0].text.startswith("alpha")


def test_sample_results_match_source_positions(tmp_path, fake_embedder):
    root = tmp_path / "samples"
    shutil.copytree(Path(__file__).resolve().parents[1] / "samples", root)
    path = tmp_path / "samples.sqlite3"
    index_corpus(root, path, fake_embedder)
    for mode in ("semantic", "keyword"):
        results = search(path, "transaction", mode=mode, embedder=fake_embedder)
        assert results
        with Storage(path, readonly=True) as store:
            for result in results:
                row = store.connection.execute("SELECT * FROM chunks WHERE id=?", (result.chunk_id,)).fetchone()
                source = (root / result.path).read_text()
                assert result.text == row["text"] == source[row["start_char"]:row["end_char"]]
                assert result.start_line == source[:row["start_char"]].count("\n") + 1


@pytest.mark.parametrize("query", ["", "  ", "\n\t"])
def test_invalid_query(corpus, query):
    with pytest.raises(QueryError): search(corpus[1], query)


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_k(corpus, k):
    with pytest.raises(QueryError): search(corpus[1], "alpha", top_k=k)


def test_empty_index_avoids_model_load(tmp_path, fake_embedder, monkeypatch):
    root = tmp_path / "empty"
    root.mkdir()
    path = tmp_path / "empty.sqlite3"
    index_corpus(root, path, fake_embedder)
    def forbidden(*args, **kwargs): raise AssertionError("Unnecessary model load")
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", forbidden)
    assert search(path, "query") == search(path, "query", mode="keyword") == []
