import numpy as np


def cosine_reference(matrix, query, ids, k):
    vectors = matrix.astype(np.float64)
    query64 = query.astype(np.float64)
    scores = (vectors @ query64) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query64))
    order = np.lexsort((ids, -scores))[:min(k, len(ids))]
    return ids[order], scores[order]
