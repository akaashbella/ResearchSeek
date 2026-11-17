import hashlib
import os
from pathlib import Path

import pytest

from researchseek import documents
from researchseek.documents import DocumentError, discover_documents, read_document


def test_discovery(tmp_path):
    names = ["z.MD", "nested/α.txt", "nested/a.TXT", "b.md", "skip.csv"]
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("note", encoding="utf-8")
    (tmp_path / "empty").mkdir()
    (tmp_path / "linked.md").symlink_to(tmp_path / "b.md")
    (tmp_path / "linked-dir").symlink_to(tmp_path / "nested", target_is_directory=True)
    expected = ["b.md", "nested/a.TXT", "nested/α.txt", "z.MD"]
    for _ in range(2):
        assert [path.relative_to(tmp_path).as_posix() for path in discover_documents(tmp_path)] == expected


def test_database_excluded_even_with_document_extension(tmp_path):
    database = tmp_path / "database.md"
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(str(database) + suffix).write_bytes(b"database")
    (tmp_path / "note.txt").write_text("note")
    assert discover_documents(tmp_path, database) == [tmp_path / "note.txt"]


def test_empty_directory(tmp_path):
    assert discover_documents(tmp_path) == []


@pytest.mark.parametrize("kind", ["missing", "file", "symlink"])
def test_invalid_root(tmp_path, kind):
    root = tmp_path / "root"
    if kind == "file":
        root.write_text("not a directory")
    elif kind == "symlink":
        root.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(DocumentError, match="root"):
        discover_documents(root)


def test_scan_failure_is_not_an_empty_corpus(tmp_path, monkeypatch):
    def broken_walk(root, *, followlinks, onerror):
        onerror(PermissionError("denied"))
    monkeypatch.setattr(documents.os, "walk", broken_walk)
    with pytest.raises(DocumentError, match="Cannot scan"):
        discover_documents(tmp_path)


def test_read_preserves_content_and_original_hash(tmp_path):
    path = tmp_path / "研究.MD"
    raw = "  α\r\n\rβ\n\tend  \r\n".encode()
    path.write_bytes(raw)
    document = read_document(tmp_path, path)
    assert document.path == "研究.MD"
    assert document.text == "  α\n\nβ\n\tend  \n"
    assert document.content_hash == hashlib.sha256(raw).hexdigest()
    assert document.size_bytes == len(raw)
    assert document.mtime_ns == path.stat().st_mtime_ns


def test_equal_size_and_timestamp_do_not_determine_hash(tmp_path):
    path = tmp_path / "note.txt"
    path.write_bytes(b"first")
    first = read_document(tmp_path, path)
    path.write_bytes(b"other")
    os.utime(path, ns=(first.mtime_ns, first.mtime_ns))
    second = read_document(tmp_path, Path("note.txt"))
    assert first.size_bytes == second.size_bytes
    assert first.mtime_ns == second.mtime_ns
    assert first.content_hash != second.content_hash


def test_invalid_utf8(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"valid\xff")
    with pytest.raises(DocumentError, match=r"Invalid UTF-8.*bad.txt.*byte 5"):
        read_document(tmp_path, Path("bad.txt"))


@pytest.mark.parametrize("kind", ["symlink", "parent-symlink", "outside", "traversal", "unsupported", "missing"])
def test_invalid_document_paths(tmp_path, kind):
    (tmp_path / "note.txt").write_text("note")
    (tmp_path / "directory").mkdir()
    (tmp_path / "linked.txt").symlink_to(tmp_path / "note.txt")
    (tmp_path / "linked-dir").symlink_to(tmp_path / "directory", target_is_directory=True)
    (tmp_path / "note.csv").write_text("unsupported")
    paths = {"symlink": Path("linked.txt"), "parent-symlink": Path("linked-dir/note.txt"),
             "outside": tmp_path.parent / "note.txt", "traversal": Path("../note.txt"),
             "unsupported": Path("note.csv"), "missing": Path("missing.txt")}
    with pytest.raises(DocumentError):
        read_document(tmp_path, paths[kind])


def test_file_modified_during_read(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_bytes(b"before")
    real_fstat = documents.os.fstat
    calls = 0

    def changing_fstat(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.write_bytes(b"modified during read")
        return real_fstat(descriptor)

    monkeypatch.setattr(documents.os, "fstat", changing_fstat)
    with pytest.raises(DocumentError, match="changed while reading"):
        read_document(tmp_path, path)


def test_file_replaced_during_read(tmp_path, monkeypatch):
    path = tmp_path / "note.txt"
    path.write_bytes(b"before")
    real_fstat = documents.os.fstat
    calls = 0

    def replacing_fstat(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            other = tmp_path / "replacement"
            other.write_bytes(b"before")
            other.replace(path)
        return real_fstat(descriptor)

    monkeypatch.setattr(documents.os, "fstat", replacing_fstat)
    with pytest.raises(DocumentError, match="changed while reading"):
        read_document(tmp_path, path)
