# Ref:
# - RAG prompt / grounding patterns: https://www.promptingguide.ai/techniques/rag
#
# This is the only place in the codebase that formats "retrieved context +
# question" into the final prompt sent to the LLM. Centralizing it here
# means the citation/grounding instructions are defined once, instead of
# being duplicated (and drifting) anywhere else that needs to build a RAG
# prompt.

from app.infrastructure.rag.vector_store import IndexedChunk

_SYSTEM_INSTRUCTION = (
    "You are a research assistant. Answer the user's question using ONLY "
    "the information in the numbered sources below. If the sources do not "
    "contain enough information to answer, say so explicitly instead of "
    "guessing. When you use information from a source, cite it inline "
    "using its number in square brackets, e.g. [1]. Do not fabricate "
    "sources or facts not present in the context."
)


def build_rag_prompt(
    question: str, ranked_chunks: list[tuple[IndexedChunk, float]]
) -> str:
    """
    Assemble the final prompt: grounding instructions + numbered sources +
    the original question.

    ranked_chunks is expected to already be sorted and trimmed to top_k by
    the caller (vector_store.search), this function only formats what
    it's given; it doesn't decide how many sources to include.
    """

    if not ranked_chunks:
        # No sources retrieved; fall back to a plain question so the model
        # can still answer from its own knowledge (per _SYSTEM_INSTRUCTION's
        # "say so explicitly" clause it should flag the lack of sources).
        return (
            f"{_SYSTEM_INSTRUCTION}\n\nSources:\n(none retrieved)\n\n"
            f"Question: {question}"
        )

    sources_block = "\n\n".join(
        f"[{i}] Source: {chunk.source_title} ({chunk.source_url})\n{chunk.text}"
        for i, (chunk, _score) in enumerate(ranked_chunks, start=1)
    )

    return f"{_SYSTEM_INSTRUCTION}\n\nSources:\n{sources_block}\n\nQuestion: {question}"
