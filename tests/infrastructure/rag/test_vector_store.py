from app.infrastructure.rag.vector_store import IndexedChunk, InMemoryVectorStore


def _chunk(text: str, embedding: list[float]) -> IndexedChunk:
    return IndexedChunk(
        text=text,
        embedding=embedding,
        source_url="https://example.com",
        source_title="Example",
    )


def test_search_returns_most_similar_chunk_first():
    store = InMemoryVectorStore()
    store.index(
        [
            _chunk("about cats", [1.0, 0.0]),
            _chunk("about dogs", [0.0, 1.0]),
        ]
    )

    results = store.search(query_embedding=[1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0][0].text == "about cats"
    assert results[0][1] == 1.0


def test_search_respects_top_k():
    store = InMemoryVectorStore()
    store.index([_chunk(f"chunk {i}", [1.0, 0.0]) for i in range(10)])

    results = store.search(query_embedding=[1.0, 0.0], top_k=3)

    assert len(results) == 3


def test_search_on_empty_store_returns_empty_list():
    store = InMemoryVectorStore()

    assert store.search(query_embedding=[1.0, 0.0]) == []


def test_len_reflects_indexed_chunk_count():
    store = InMemoryVectorStore()
    store.index([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])

    assert len(store) == 2
