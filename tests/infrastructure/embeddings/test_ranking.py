import pytest

from app.infrastructure.embeddings.ranking import cosine_similarity_scores, rank_chunks


def test_cosine_similarity_scores_identical_vector_scores_highest():
    scores = cosine_similarity_scores(
        [1.0, 0.0], [[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]]
    )

    assert scores[0] == max(scores)


def test_cosine_similarity_scores_empty_embeddings_returns_empty():
    assert cosine_similarity_scores([1.0, 0.0], []) == []


def test_rank_chunks_returns_top_k_sorted_descending():
    items = ["a", "b", "c", "d"]
    scores = [0.1, 0.9, 0.5, 0.3]

    ranked = rank_chunks(items, scores, top_k=2)

    assert ranked == [("b", 0.9), ("c", 0.5)]


def test_rank_chunks_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        rank_chunks(["a", "b"], [0.1], top_k=1)
