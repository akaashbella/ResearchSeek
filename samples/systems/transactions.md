# Transaction boundaries

An indexing run changes document metadata, chunks, vectors, and keyword entries
together. Committing each document separately can expose an incomplete corpus
when a later file fails to decode. One transaction makes the previous committed
state available again after a rollback.

SQLite foreign keys remove dependent rows when a document is deleted. FTS5 needs
separate triggers because its inverted index is not an ordinary child table.
Deletion tests should issue MATCH queries; reading an external-content FTS table
alone does not prove that its inverted index is synchronized.
