import json
import sqlite3

import pytest

from researchseek.cli import main
from researchseek.embeddings import DEFAULT_MODEL, DEFAULT_REVISION, EmbeddingError
from researchseek import embeddings


@pytest.fixture
def cli_model(monkeypatch, fake_embedder):
    fake_embedder.model_id = DEFAULT_MODEL
    fake_embedder.model_revision = DEFAULT_REVISION
    calls = []
    def load(model, **kwargs):
        calls.append((model, kwargs))
        return fake_embedder
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", load)
    return fake_embedder, calls


def test_cli_initial_unchanged_and_json(tmp_path, cli_model, capsys):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("A test note")
    path = tmp_path / "corpus.sqlite3"
    args = ["index", str(root), "--db", str(path)]
    assert main(args) == 0
    assert "Added: 1" in capsys.readouterr().out
    cli_model[0].calls.clear()
    assert main([*args, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["unchanged"] == 1 and result["chunks"] == 1
    assert result["elapsed_seconds"] >= 0
    assert cli_model[0].calls == []
    assert cli_model[1][-1][1]["revision"] == DEFAULT_REVISION
    assert cli_model[1][-1][1]["allow_download"] is False


def test_default_db_download_flag_and_rebuild(tmp_path, cli_model, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "notes"
    root.mkdir()
    (root / "note.md").write_text("A test note")
    assert main(["index", str(root), "--download-model"]) == 0
    assert (tmp_path / "researchseek.sqlite3").exists()
    assert cli_model[1][-1][1]["allow_download"] is True
    assert main(["index", str(root), "--rebuild", "--json"]) == 0
    capsys.readouterr()


@pytest.mark.parametrize("error", [EmbeddingError("Model missing; use --download-model"),
                                  sqlite3.OperationalError("database locked"), ValueError("incompatible model")])
def test_operational_errors(tmp_path, monkeypatch, capsys, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", fail)
    assert main(["index", str(tmp_path), "--db", str(tmp_path / "error.sqlite3")]) == 1
    output = capsys.readouterr()
    assert str(error) in output.err and output.out == ""


def test_missing_directory_precedes_model_loading(tmp_path, cli_model, capsys):
    assert main(["index", str(tmp_path / "missing")]) == 1
    assert cli_model[1] == []
    assert "root" in capsys.readouterr().err


def test_bad_utf8_cli_error(tmp_path, cli_model, capsys):
    (tmp_path / "bad.txt").write_bytes(b"\xff")
    assert main(["index", str(tmp_path), "--db", str(tmp_path / "bad.sqlite3")]) == 1
    assert "Invalid UTF-8" in capsys.readouterr().err


@pytest.mark.parametrize("option,value", [("--batch-size", "0"), ("--max-tokens", "0"),
                                        ("--overlap", "-1"), ("--batch-size", "bad")])
def test_invalid_cli_options(tmp_path, option, value):
    with pytest.raises(SystemExit) as error:
        main(["index", str(tmp_path), option, value])
    assert error.value.code == 2


def test_invalid_token_overlap_is_usage_error(tmp_path, cli_model):
    with pytest.raises(SystemExit) as error:
        main(["index", str(tmp_path), "--max-tokens", "10", "--overlap", "8"])
    assert error.value.code == 2
