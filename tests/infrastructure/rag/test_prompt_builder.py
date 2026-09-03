from app.infrastructure.rag.prompt_builder import build_rag_prompt
from app.infrastructure.rag.vector_store import IndexedChunk


def _ranked_chunk(text, url="https://example.com", title="Example", score=0.9):
    return (
        IndexedChunk(text=text, embedding=[0.0], source_url=url, source_title=title),
        score,
    )


def test_build_rag_prompt_includes_question_and_sources():
    prompt = build_rag_prompt(
        "What is RAG?", [_ranked_chunk("RAG combines retrieval and generation.")]
    )

    assert "What is RAG?" in prompt
    assert "RAG combines retrieval and generation." in prompt
    assert "[1]" in prompt


def test_build_rag_prompt_numbers_sources_in_order():
    chunks = [_ranked_chunk("first chunk"), _ranked_chunk("second chunk")]

    prompt = build_rag_prompt("question", chunks)

    assert prompt.index("[1]") < prompt.index("[2]")
    assert prompt.index("first chunk") < prompt.index("second chunk")


def test_build_rag_prompt_with_no_sources_still_includes_question():
    prompt = build_rag_prompt("question with no sources", [])

    assert "question with no sources" in prompt
    assert "none retrieved" in prompt
