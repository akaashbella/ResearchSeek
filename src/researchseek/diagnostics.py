"""Environment checks needed by the planned storage layer."""

import sqlite3
from contextlib import closing


def check_fts5() -> None:
    """Raise RuntimeError if this Python's SQLite cannot create an FTS5 table."""
    try:
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(text)")
    except sqlite3.OperationalError as error:
        raise RuntimeError(
            "SQLite FTS5 is unavailable. Use a Python build with FTS5-enabled SQLite."
        ) from error
