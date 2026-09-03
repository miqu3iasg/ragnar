# Ref (FAISS vs ChromaDB vs pgvector comparison): https://benchmark.vectorview.ai/vectordbs.html
#
# Starts as a plain in-memory structure (list of chunks + embeddings,
# compared via cosine similarity) instead of FAISS/ChromaDB/pgvector.
# At this project's current scale (a handful of sources fetched per
# question, not a persistent corpus kept across requests) an index of a
# few hundred vectors doesn't need approximate nearest-neighbor search;
# an exact O(n) scan is fast enough. The class exposes only `index` and
# `search`, so swapping in a real vector store later doesn't require
# touching any of its callers (see the note in the roadmap doc about this
# exact interface staying stable across that swap).

from dataclasses import dataclass

from app.infrastructure.embeddings.ranking import cosine_similarity_scores, rank_chunks


@dataclass
class IndexedChunk:
    """A chunk of source text, paired with its embedding and the page it came from."""

    text: str
    embedding: list[float]
    source_url: str
    source_title: str


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: list[IndexedChunk] = []

    def index(self, chunks: list[IndexedChunk]) -> None:
        self._chunks.extend(chunks)

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[IndexedChunk, float]]:
        """
        Return the top_k indexed chunks most similar to the query
        embedding, as (chunk, similarity_score) pairs, sorted descending.

        The actual scoring/ranking logic lives in embeddings/ranking.py;
        this method only owns storage and delegates the "which chunk is
        more relevant" decision to it.
        """

        if not self._chunks:
            return []

        scores = cosine_similarity_scores(
            query_embedding, [chunk.embedding for chunk in self._chunks]
        )
        return rank_chunks(self._chunks, scores, top_k)

    def __len__(self) -> int:
        return len(self._chunks)
