# Retrieval notes

Cosine similarity compares vector direction rather than magnitude. Normalizing
embeddings allows a dot product to represent cosine similarity, apart from
floating-point rounding. A zero vector has no defined direction.

An exact search scans every passage vector. A bounded heap retains the best k
candidates without sorting the full corpus. Equal scores need a stable secondary
ordering so repeated queries produce the same results.

Keyword search remains useful for identifiers such as sqlite3 and O(log k).
A semantic model may help with paraphrases, but relevance needs labeled queries
and measured comparisons.
