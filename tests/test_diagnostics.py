import sqlite3

import pytest

from researchseek.diagnostics import check_fts5


def test_fts5_available():
    check_fts5()


def test_missing_fts5(monkeypatch):
    class ConnectionWithoutFTS5:
        def execute(self, statement):
            raise sqlite3.OperationalError("no such module: fts5")

        def close(self):
            pass

    monkeypatch.setattr(sqlite3, "connect", lambda database: ConnectionWithoutFTS5())
    with pytest.raises(RuntimeError, match="FTS5-enabled SQLite"):
        check_fts5()
