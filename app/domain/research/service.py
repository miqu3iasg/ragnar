# Refs:
# - RAG prompt / grounding patterns: https://www.promptingguide.ai/techniques/rag
# - Chunking strategies for RAG: https://www.pinecone.io/learn/chunking-strategies/
# - OpenAI function/tool calling guide: https://platform.openai.com/docs/guides/function-calling
# - asyncio.gather: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
# - openai Python SDK exception hierarchy: https://github.com/openai/openai-python/blob/main/src/openai/_exceptions.py
# - tenacity (retry policy this module's calls eventually run through,
#   defined in infrastructure/llm/client.py): https://tenacity.readthedocs.io/en/latest/
#
# This module owns the whole research flow: ask the model whether it needs
# to search, and if it does, run the retrieval pipeline (search -> fetch ->
# extract -> chunk -> embed -> rank) and hand the model back a grounded
# prompt to answer from. It's also still the translation boundary between
# infrastructure-level failures (openai SDK exceptions, search/embedding
# failures) and domain-level exceptions the API layer (api/routes/
# research.py + api/exception_handlers.py) knows how to map to HTTP status
# codes.

import asyncio
import json
import logging

# Ref (this exception hierarchy): https://github.com/openai/openai-python/
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.domain.research.answer import Answer
from app.domain.research.exceptions import (
    EmptyQuestionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
    RetrievalUnavailableError,
)
from app.domain.research.question import Question
from app.domain.research.source import Source
from app.infrastructure.embeddings.client import (
    EmbeddingModelUnavailableError,
    embed_text,
    embed_texts,
)
from app.infrastructure.llm.client import get_completion_with_tools
from app.infrastructure.llm.tools import AVAILABLE_TOOLS
from app.infrastructure.rag.chunking import chunk_text
from app.infrastructure.rag.prompt_builder import build_rag_prompt
from app.infrastructure.rag.vector_store import IndexedChunk, InMemoryVectorStore
from app.infrastructure.search.client import SearchResult, search_web
from app.infrastructure.search.extractor import extract_page_text

logger = logging.getLogger(__name__)

# How many chunks (pooled across every fetched source) go into the final
# RAG prompt. Kept small so the prompt stays within a reasonable token
# budget for a free-tier model, while still giving it a few independent
# sources to cross-reference.
TOP_K_CHUNKS = 5


async def _call_llm_with_tools(messages: list[dict], tools: list[dict] | None):
    """
    Wraps get_completion_with_tools, translating SDK-level exceptions into
    domain exceptions. Shared by both LLM calls in ask_research_question
    below (the initial call, which may request a tool, and the follow-up
    call after RAG context has been injected) so this mapping exists in
    exactly one place.

    Ref (RateLimitError IS-A APIStatusError, which is why it must be
    caught in its own except clause BEFORE the generic APIStatusError one
    below — a Python except clause matches the first compatible type it
    sees): https://docs.python.org/3/tutorial/errors.html#handling-exceptions
    """

    try:
        return await get_completion_with_tools(messages, tools=tools)
    except RateLimitError as exc:
        # Ref: https://tenacity.readthedocs.io/en/latest/
        # By the time we get here, tenacity's @retry in client.py has
        # already exhausted and reraised the original exception, so this
        # is the final rate limit failure, not the first one.
        logger.warning(f"Rate limit persisted after retries exhausted: {exc}")
        raise LLMRateLimitError("Provider rate limit exceeded.") from exc
    except (APIConnectionError, APITimeoutError) as exc:
        logger.error(f"Connection to LLM provider failed: {exc}")
        raise LLMUnavailableError("Could not reach the LLM provider.") from exc
    except APIStatusError as exc:
        logger.error(
            f"LLM provider returned an error (status={exc.status_code}): {exc}"
        )
        raise LLMUnavailableError(
            f"LLM provider returned an error (status {exc.status_code})."
        ) from exc
    except Exception:
        # Same rationale as the previous single-call version of this
        # module: don't wrap unknown exceptions in a domain exception,
        # since that would hide the real cause from logs/monitoring.
        logger.exception("Unexpected error during LLM call.")
        raise


async def _process_source(result: SearchResult) -> list[tuple[str, SearchResult]]:
    """
    Fetch, extract, and chunk a single search result. Returns a list of
    (chunk_text, source) pairs, so each chunk carries the source it came
    from without a second lookup later.

    Returns an empty list if the page couldn't be fetched, had no
    extractable content, or produced no chunks — callers treat all three
    the same way: skip this source.
    """
    page_text = await extract_page_text(str(result.url))
    if page_text is None:
        return []

    # Ref: chunking strategy background — https://www.pinecone.io/learn/chunking-strategies/
    chunks = chunk_text(page_text)
    return [(chunk, result) for chunk in chunks]


def _deduplicate(
    pairs: list[tuple[str, SearchResult]],
) -> list[tuple[str, SearchResult]]:
    """
    Drop chunks whose text is an exact match (ignoring case and
    whitespace differences) of one already seen, keeping the first
    occurrence.

    Different pages often repeat the exact same paragraph — boilerplate,
    syndicated content, a quote picked up by multiple outlets — and
    without deduping, top_k can end up mostly filled with near-identical
    chunks instead of a diverse set of sources. This is intentionally a
    cheap exact-match check, not a similarity-based near-duplicate
    detector (that would need something like MinHash or embedding-space
    clustering — out of scope here); it catches the common case without
    adding another model or library to the pipeline.
    """
    seen: set[str] = set()
    deduped = []
    for chunk, result in pairs:
        normalized = " ".join(chunk.split()).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append((chunk, result))
    return deduped


async def _run_search_tool(query: str) -> list[tuple[IndexedChunk, float]]:
    """
    Execute the full retrieval pipeline for one search query: search the
    web, fetch and extract each page, chunk it, embed everything, and
    return the top_k chunks ranked by similarity to the query.

    Ref (asyncio.gather — why sources are fetched concurrently instead of
    one at a time): https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
    With up to 5 sources per query and each fetch independently bounded
    by extractor.py's own timeout, doing this sequentially would multiply
    worst-case latency by the number of sources for no benefit, since the
    sources don't depend on each other.

    Any single source failing (fetch error, no extractable content) is
    skipped rather than failing the whole pipeline — a partial set of
    sources is still useful; failing outright over one bad page isn't.

    A genuine search or embedding failure, on the other hand — the
    provider being down, not one bad page — is not swallowed; it's
    translated into RetrievalUnavailableError so the API layer can
    surface it distinctly from an LLM failure (see api/exception_handlers.py).
    """
    try:
        results = await search_web(query)
    except Exception as exc:
        logger.exception(f"Web search failed for query: {query!r}")
        raise RetrievalUnavailableError("Could not complete the web search.") from exc

    if not results:
        return []

    per_source_pairs = await asyncio.gather(
        *(_process_source(result) for result in results)
    )
    pairs = [pair for source_pairs in per_source_pairs for pair in source_pairs]
    if not pairs:
        return []

    deduped_pairs = _deduplicate(pairs)
    texts = [chunk for chunk, _result in deduped_pairs]

    try:
        chunk_embeddings = await embed_texts(texts)
    except EmbeddingModelUnavailableError as exc:
        logger.error(f"Embedding model unavailable during retrieval: {exc}")
        raise RetrievalUnavailableError(
            "Could not generate embeddings for retrieved content."
        ) from exc

    store = InMemoryVectorStore()
    store.index(
        [
            IndexedChunk(
                text=chunk,
                embedding=embedding,
                source_url=str(result.url),
                source_title=result.title,
            )
            for (chunk, result), embedding in zip(deduped_pairs, chunk_embeddings)
        ]
    )

    if len(store) == 0:
        return []

    try:
        query_embedding = await embed_text(query)
    except EmbeddingModelUnavailableError as exc:
        logger.error(f"Embedding model unavailable during retrieval: {exc}")
        raise RetrievalUnavailableError(
            "Could not generate an embedding for the search query."
        ) from exc

    # Ref (cosine similarity ranking): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    return store.search(query_embedding, top_k=TOP_K_CHUNKS)


async def ask_research_question(question: Question) -> Answer:
    # A question that is only whitespace should be treated the same as an
    # empty question, not silently sent to the LLM.
    question_text = question.question_text.strip()
    if not question_text:
        raise EmptyQuestionError("Question content must not be empty.")

    messages = [{"role": "user", "content": question_text}]
    message = await _call_llm_with_tools(messages, tools=AVAILABLE_TOOLS)

    # The model decided the question doesn't need external sources (see
    # tools.py's search_web description), answer directly, no RAG.
    if not message.tool_calls:
        answer_text = message.content
        if not answer_text or not answer_text.strip():
            logger.error(
                f"LLM provider returned an empty response for question: {question_text!r}"
            )
            raise LLMResponseError("The model did not return a usable response.")
        return Answer(question_id=question.id, answer_text=answer_text, sources=[])

    # The model asked to search. Only the first tool call is executed,
    # AVAILABLE_TOOLS currently exposes a single tool (search_web), so a
    # well-behaved model shouldn't request more than one per turn.
    # Ref: https://platform.openai.com/docs/guides/function-calling
    tool_call = message.tool_calls[0]
    try:
        # Ref: json.loads / JSONDecodeError — https://docs.python.org/3/library/json.html
        arguments = json.loads(tool_call.function.arguments)
        search_query = arguments["query"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error(
            f"Model returned malformed tool call arguments: {tool_call.function.arguments!r}"
        )
        raise LLMResponseError(
            "The model requested a search with invalid arguments."
        ) from exc

    ranked_chunks = await _run_search_tool(search_query)
    rag_prompt = build_rag_prompt(question_text, ranked_chunks)

    # Ref: https://platform.openai.com/docs/guides/function-calling
    # Standard tool-calling turn shape: echo the assistant's tool call,
    # supply a "tool" role message with the result, then a fresh "user"
    # message carrying the actual RAG-formatted context + question.
    follow_up_messages = messages + [
        {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": f"Retrieved {len(ranked_chunks)} relevant source excerpts.",
        },
        {"role": "user", "content": rag_prompt},
    ]

    # tools=[] here: this is the grounded, final-answer turn, we don't
    # want the model requesting another search instead of answering.
    final_message = await _call_llm_with_tools(follow_up_messages, tools=[])

    answer_text = final_message.content
    if not answer_text or not answer_text.strip():
        logger.error(
            f"LLM provider returned an empty final answer for question: {question_text!r}"
        )
        raise LLMResponseError("The model did not return a usable response.")

    sources = [
        Source(
            url=chunk.source_url,
            title=chunk.source_title,
            content=chunk.text,
            relevance_score=score,
        )
        for chunk, score in ranked_chunks
    ]

    return Answer(question_id=question.id, answer_text=answer_text, sources=sources)
