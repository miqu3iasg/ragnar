import pytest

from app.infrastructure.rag.chunking import _ENCODING, chunk_text


def test_chunk_text_splits_long_text_into_multiple_chunks():
    text = "word " * 1000
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) > 0 for chunk in chunks)


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=300, overlap=50)

    assert len(chunks) == 1


def test_chunk_text_empty_string_returns_empty_list():
    assert chunk_text("", chunk_size=300, overlap=50) == []


def test_chunk_text_overlap_preserves_shared_tokens():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    tokens_0 = _ENCODING.encode(chunks[0])
    tokens_1 = _ENCODING.encode(chunks[1])

    # The last `overlap` tokens of chunk 0 should reappear at the start of chunk 1.
    assert tokens_0[-20:] == tokens_1[:20]


def test_chunk_text_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_chunk_text_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=0, overlap=0)
