from pathlib import Path
import shutil
import json

import numpy as np
import pytest

from researchseek.chunking import chunk_text
from researchseek.embeddings import DEFAULT_MODEL, DEFAULT_REVISION, SentenceTransformerEmbedder
from researchseek.indexing import index_corpus
from researchseek.storage import Storage
from researchseek.search import search
from researchseek.cli import main


@pytest.mark.real_model
def test_cached_real_model_indexing(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    embedder = SentenceTransformerEmbedder()
    assert embedder.model_id == DEFAULT_MODEL and embedder.model_revision == DEFAULT_REVISION
    assert embedder.dimension == 384 and embedder.tokenizer.max_tokens == 256
    assert embedder.tokenizer.special_tokens == 2
    source = "  "+ "\n\n".join(["Tokenization preserves naïve café and 研究 offsets. " * 40] * 2)
    chunks = chunk_text(source, embedder.tokenizer)
    assert chunks == chunk_text(source, embedder.tokenizer)
    assert all(chunk.text == source[chunk.start_char:chunk.end_char] for chunk in chunks)
    assert all(embedder.tokenizer.count_tokens(chunk.text) <= 200 for chunk in chunks)
    root = tmp_path / "samples"
    shutil.copytree(Path(__file__).resolve().parents[1] / "samples", root)
    path = tmp_path / "real.sqlite3"
    first = index_corpus(root, path, embedder)
    assert first.added == 5 and first.chunks > 5
    with Storage(path) as store:
        assert len(store.embeddings()) == first.chunks
        assert all(vector.shape == (384,) for vector in store.embeddings().values())
        assert all(np.isclose(np.linalg.norm(vector), 1, atol=1e-5) for vector in store.embeddings().values())
        before_ids = [row["id"] for row in store.chunks()]
    original_encode = embedder.encode
    def no_encode(*args, **kwargs):
        raise AssertionError("Unchanged indexing must not call the encoder")
    monkeypatch.setattr(embedder, "encode", no_encode)
    second = index_corpus(root, path, embedder)
    assert second.unchanged == 5
    with Storage(path) as store:
        assert [row["id"] for row in store.chunks()] == before_ids
    monkeypatch.setattr(embedder, "encode", original_encode)
    (root / "retrieval.md").write_text("Revised retrieval note about database transactions.")
    (root / "unicode.TXT").unlink()
    changed = index_corpus(root, path, embedder)
    assert changed.updated == changed.deleted == 1 and changed.unchanged == 3
    with Storage(path) as store:
        assert len(store.embeddings()) == changed.chunks
    for mode in ("semantic", "keyword"):
        results = search(path, "database transactions", mode=mode)
        assert results
        with Storage(path, readonly=True) as store:
            for result in results:
                row = store.connection.execute("SELECT * FROM chunks WHERE id=?", (result.chunk_id,)).fetchone()
                source = (root / result.path).read_text()
                assert result.text == source[row["start_char"]:row["end_char"]]
                assert result.start_line == source[:row["start_char"]].count("\n") + 1
                assert result.end_line == source[:row["end_char"] - 1].count("\n") + 1
        for _ in range(2):
            assert main(["search", "database transactions", "--db", str(path), "--mode", mode, "--json"]) == 0
            response = json.loads(capsys.readouterr().out)
            assert response["mode"] == mode and response["results"]
            assert response["results"][0]["chunk_id"] == results[0].chunk_id
