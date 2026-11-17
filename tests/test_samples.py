from pathlib import Path
import shutil

from researchseek.chunking import WhitespaceTokenizer, chunk_text
from researchseek.documents import discover_documents, read_document
from researchseek.storage import CorpusConfig, Storage


def test_sample_read_chunk_store_roundtrip(tmp_path):
    root = tmp_path / "samples"
    shutil.copytree(Path(__file__).resolve().parents[1] / "samples", root)
    assert b"\r\n" in (root / "unicode.TXT").read_bytes()
    legacy = (root / "systems/line-endings.txt").read_bytes()
    assert b"\r" in legacy and b"\n" not in legacy
    assert b"\r" not in (root / "retrieval.md").read_bytes()
    files = discover_documents(root)
    assert len(files) == 5
    assert files == discover_documents(root)
    tokenizer = WhitespaceTokenizer()
    prepared = []
    for path in files:
        document = read_document(root, path)
        assert "\r" not in document.text
        chunks = chunk_text(document.text, tokenizer)
        assert chunks == chunk_text(document.text, tokenizer)
        for chunk in chunks:
            assert document.text[chunk.start_char:chunk.end_char] == chunk.text
            assert chunk.start_line == document.text[:chunk.start_char].count("\n") + 1
            assert chunk.end_line == document.text[:chunk.end_char - 1].count("\n") + 1
            assert tokenizer.count_tokens(chunk.text) <= 200
        prepared.append((document, chunks))
    assert any(len(chunks) > 1 for _, chunks in prepared)
    with Storage(tmp_path / "sample.sqlite3") as database:
        with database.transaction():
            database.configure(CorpusConfig(str(root.resolve())))
            for document, chunks in prepared:
                database.put_document(document, chunks)
        assert len(database.documents()) == 5
        assert len(database.chunks()) == sum(len(chunks) for _, chunks in prepared)
        assert database.connection.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?", ("rollback",)
        ).fetchall()
        for document, chunks in prepared:
            stored = database.get_document(document.path)
            assert stored["content_hash"] == document.content_hash
            assert [row["text"] for row in database.chunks(stored["id"])] == [c.text for c in chunks]
        assert database.connection.execute("PRAGMA foreign_key_check").fetchall() == []
