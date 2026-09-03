# Ref (cosine similarity): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
#
# Kept separate from rag/vector_store.py so the similarity metric (the
# actual "which reference is more relevant" ranking logic) can be reused
# or swapped independently of how/where chunks are stored.

import numpy as np


def cosine_similarity_scores(
    query_embedding: list[float], embeddings: list[list[float]]
) -> list[float]:
    """
    Score each vector in `embeddings` against `query_embedding`.

    Assumes every vector is already unit-normalized (see
    embeddings/client.py's normalize_embeddings=True), which makes cosine
    similarity equivalent to a plain dot product — cheaper than recomputing
    norms on every call.
    """

    # Return an empty list when there are no embeddings to score.
    if not embeddings:
        return []

    # Convert the query embedding from a Python list into a NumPy array.
    query_vec = np.array(query_embedding)

    # Convert all embeddings into a 2D NumPy array, where each row is a vector.
    matrix = np.array(embeddings)

    # The @ operator performs matrix multiplication. Here, it computes the dot product
    # between the query vector and every embedding. Since all vectors are unit-normalized,
    # the dot product equals cosine similarity. Convert the resulting NumPy array back
    # into a regular Python list.
    return (matrix @ query_vec).tolist()


def rank_chunks(items: list, scores: list[float], top_k: int) -> list[tuple]:
    """
    Pair arbitrary items with their similarity scores and return the
    top_k, sorted descending by score.

    Generic over `items` (kept untyped on purpose) so it works for text
    chunks, whole sources, or anything else scored by similarity —
    vector_store.py is the only current caller.
    """

    if len(items) != len(scores):
        raise ValueError("items and scores must have the same length.")

    ranked = sorted(zip(items, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]
