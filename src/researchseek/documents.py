"""Document discovery and consistent UTF-8 reads."""

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat


class DocumentError(RuntimeError):
    """A corpus cannot be scanned or a document cannot be read consistently."""


@dataclass(frozen=True)
class Document:
    path: str
    text: str
    content_hash: str
    size_bytes: int
    mtime_ns: int


def _validate_root(root: Path) -> Path:
    root = Path(os.path.abspath(root.expanduser()))
    if root.is_symlink() or not root.is_dir():
        raise DocumentError(f"Corpus root must be an existing, non-symlink directory: {root}")
    return root


def discover_documents(root: Path, database_path: Path | None = None) -> list[Path]:
    """Return supported regular files sorted by root-relative POSIX path."""
    root = _validate_root(root)
    excluded = set()
    if database_path is not None:
        database = Path(database_path).resolve()
        excluded = {Path(str(database) + suffix) for suffix in ("", "-wal", "-shm", "-journal")}

    def scan_error(error: OSError) -> None:
        raise error

    files = []
    try:
        for directory, subdirs, names in os.walk(root, followlinks=False, onerror=scan_error):
            subdirs[:] = [name for name in subdirs if not (Path(directory) / name).is_symlink()]
            for name in names:
                path = Path(directory) / name
                if path.suffix.lower() not in {".txt", ".md"}:
                    continue
                mode = path.lstat().st_mode
                if stat.S_ISREG(mode) and path.resolve() not in excluded:
                    files.append(path)
    except OSError as error:
        raise DocumentError(f"Cannot scan {root}: {error}") from error
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def read_document(root: Path, path: Path) -> Document:
    """Read once, hash original bytes, and reject ordinary changes during the read."""
    root = _validate_root(root)
    path = Path(path)
    path = path if path.is_absolute() else root / path
    try:
        relative = path.relative_to(root)
        if ".." in relative.parts:
            raise ValueError
    except ValueError as error:
        raise DocumentError(f"Document is outside corpus root: {path}") from error
    try:
        for parent in path.parents:
            if parent == root:
                break
            if parent.is_symlink():
                raise DocumentError(f"Symlink directory is not supported: {parent}")
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.suffix.lower() not in {".txt", ".md"}:
            raise DocumentError(f"Not a supported regular document: {path}")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read()
            after = os.fstat(stream.fileno())
        current = path.lstat()
        if not (_signature(before) == _signature(opened) == _signature(after) == _signature(current)):
            raise DocumentError(f"Document changed while reading: {path}")
        if len(raw) != after.st_size:
            raise DocumentError(f"Document changed while reading: {path}")
        text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeDecodeError as error:
        raise DocumentError(f"Invalid UTF-8 in {path} at byte {error.start}") from error
    except OSError as error:
        raise DocumentError(f"Cannot read {path}: {error}") from error
    return Document(relative.as_posix(), text, hashlib.sha256(raw).hexdigest(),
                    len(raw), after.st_mtime_ns)
