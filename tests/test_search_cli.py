import json
import pytest

from researchseek import embeddings
from researchseek.cli import main
from researchseek.indexing import index_corpus


@pytest.fixture
def corpus(tmp_path, fake_embedder, monkeypatch):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("alpha beta research passage\n")
    path = tmp_path / "index.sqlite3"
    index_corpus(root, path, fake_embedder)
    calls = []
    def load(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return fake_embedder
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", load)
    return root, path, fake_embedder, calls


@pytest.mark.parametrize("mode", ["semantic", "keyword"])
def test_search_json_schema(corpus, capsys, mode):
    _, path, _, calls = corpus
    assert main(["search", "alpha", "--db", str(path), "--mode", mode, "--json"]) == 0
    output = capsys.readouterr()
    assert output.err == ""
    body = json.loads(output.out)
    assert set(body) == {"query", "mode", "top_k", "results"}
    assert body["query"] == "alpha" and body["mode"] == mode and body["top_k"] == 5
    result = body["results"][0]
    assert set(result) == {"rank", "chunk_id", "path", "text", "start_line", "end_line", "mode", "score", "score_type"}
    assert result["path"] == "note.md" and result["start_line"] == result["end_line"] == 1
    assert result["rank"] == 1 and isinstance(result["chunk_id"], int) and isinstance(result["score"], float)
    assert result["score_type"] == ("cosine_similarity" if mode == "semantic" else "fts5_bm25")
    assert len(calls) == (1 if mode == "semantic" else 0)


def test_default_mode_db_and_custom_cache(corpus, tmp_path, monkeypatch, capsys):
    _, path, _, calls = corpus
    target = tmp_path / "researchseek.sqlite3"
    path.rename(target)
    monkeypatch.chdir(tmp_path)
    assert main(["search", "query", "--cache-dir", "custom-cache", "--top-k", "1", "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["mode"] == "semantic" and body["top_k"] == 1
    assert calls[0][1]["allow_download"] is False
    assert str(calls[0][1]["cache_dir"]) == "custom-cache"


def test_human_output(corpus, capsys):
    assert main(["search", "alpha", "--mode", "keyword", "--db", str(corpus[1])]) == 0
    output = capsys.readouterr()
    assert "1. note.md:1-1 fts5_bm25=" in output.out
    assert "alpha beta research passage" in output.out and output.err == ""


@pytest.mark.parametrize("json_output", [False, True])
def test_empty_results_are_success(corpus, capsys, json_output):
    args = ["search", "nomatchxyz", "--mode", "keyword", "--db", str(corpus[1])]
    assert main(args + (["--json"] if json_output else [])) == 0
    output = capsys.readouterr()
    assert output.err == ""
    if json_output:
        assert json.loads(output.out)["results"] == []
    else:
        assert output.out.strip() == "No matches."


@pytest.mark.parametrize("arguments", [[""], [" \n "], ["alpha", "--top-k", "0"],
    ["alpha", "--top-k", "-2"], ["alpha", "--top-k", "bad"], ["alpha", "--mode", "hybrid"],
    ["alpha", "--download-model"], []])
def test_invalid_search_usage(arguments):
    with pytest.raises(SystemExit) as error:
        main(["search", *arguments])
    assert error.value.code == 2


def test_no_keyword_terms_is_usage_error(corpus):
    with pytest.raises(SystemExit) as error:
        main(["search", "...", "--mode", "keyword", "--db", str(corpus[1])])
    assert error.value.code == 2


def test_missing_database_is_operational_failure_without_creation(tmp_path, capsys):
    path = tmp_path / "missing.sqlite3"
    assert main(["search", "alpha", "--db", str(path), "--json"]) == 1
    output = capsys.readouterr()
    assert output.out == "" and "does not exist" in output.err
    assert not path.exists()


def test_missing_cache_stderr_and_keyword_survives(corpus, monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise embeddings.EmbeddingError("Missing model cache; use index --download-model")
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", missing)
    assert main(["search", "alpha", "--db", str(corpus[1]), "--json"]) == 1
    output = capsys.readouterr()
    assert output.out == "" and "Missing model cache" in output.err
    assert main(["search", "alpha", "--db", str(corpus[1]), "--mode", "keyword"]) == 0


def test_model_mismatch_is_operational(corpus, capsys):
    corpus[2].tokenizer.identity = "incompatible"
    assert main(["search", "alpha", "--db", str(corpus[1])]) == 1
    assert "incompatible" in capsys.readouterr().err


def test_too_long_query_is_usage_error(corpus):
    with pytest.raises(SystemExit) as error:
        main(["search", "word " * 300, "--db", str(corpus[1])])
    assert error.value.code == 2


def test_unexpected_programming_error_is_not_swallowed(corpus, monkeypatch):
    import researchseek.search as search_module
    def bug(*args, **kwargs):
        raise KeyError("unexpected bug")
    monkeypatch.setattr(search_module, "search", bug)
    with pytest.raises(KeyError, match="unexpected bug"):
        main(["search", "query", "--db", str(corpus[1])])
