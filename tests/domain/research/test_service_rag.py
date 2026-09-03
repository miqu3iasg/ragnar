# Refs:
# - pytest-asyncio markers: https://pytest-asyncio.readthedocs.io/en/stable/reference/markers/index.html
# - unittest.mock.AsyncMock (needed because service.py awaits these calls): https://docs.python.org/3/library/unittest.mock.html#unittest.mock.AsyncMock
# - unittest.mock, "Where to patch": https://docs.python.org/3/library/unittest.mock.html#where-to-patch
# - AsyncMock.side_effect as a list (plays return values in call order): https://docs.python.org/3/library/unittest.mock.html#unittest.mock.Mock.side_effect
# - types.SimpleNamespace (used to fake the openai SDK's message objects without depending on the real SDK classes): https://docs.python.org/3/library/types.html#types.SimpleNamespace
# - OpenAI function/tool calling message shape (what a real message with tool_calls looks like): https://platform.openai.com/docs/guides/function-calling
# - asyncio.gather concurrency testing pattern (why the elapsed-time assertion below is a legitimate way to prove parallelism): https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
# - time.monotonic (correct clock for measuring elapsed time, unaffected by system clock adjustments): https://docs.python.org/3/library/time.html#time.monotonic

import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.domain.research.answer import Answer
from app.domain.research.exceptions import LLMResponseError, RetrievalUnavailableError
from app.domain.research.question import Question
from app.domain.research.service import ask_research_question
from app.infrastructure.embeddings.client import EmbeddingModelUnavailableError
from app.infrastructure.search.client import SearchResult


def _tool_call_message(query: str, tool_call_id: str = "call_1") -> SimpleNamespace:
    # Shape mirrors the openai SDK's assistant message when the model
    # requests a tool call: content is typically empty/None, and
    # tool_calls carries the function name + JSON-encoded arguments.
    # Ref: https://platform.openai.com/docs/guides/function-calling
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=tool_call_id,
                function=SimpleNamespace(
                    name="search_web",
                    # Ref: json.dumps — https://docs.python.org/3/library/json.html
                    arguments=json.dumps({"query": query}),
                ),
            )
        ],
    )


def _final_message(content: str | None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=None)


def _source(url: str, title: str) -> SearchResult:
    # Ref: pydantic model construction — https://docs.pydantic.dev/latest/concepts/models/
    return SearchResult(url=url, title=title, snippet="")


def _patch_llm(*, tool_call_message, final_message):
    # The first call decides whether to search; the second answers using
    # the RAG-formatted prompt. Both go through get_completion_with_tools,
    # so a single AsyncMock with a side_effect list plays them in order.
    return patch(
        "app.domain.research.service.get_completion_with_tools",
        new=AsyncMock(side_effect=[tool_call_message, final_message]),
    )


@pytest.fixture
def question() -> Question:
    # Ref: https://docs.pydantic.dev/latest/concepts/fields/#default-values
    return Question(question_text="what is the capital of france?")


async def test_ask_runs_full_retrieval_pipeline_when_model_requests_a_search(question):
    source_a = _source("https://a.example.com/page", "Source A")
    source_b = _source("https://b.example.com/page", "Source B")

    extract_by_url = {
        str(source_a.url): "Paris is the capital of France.",
        str(source_b.url): "Berlin is the capital of Germany.",
    }

    with _patch_llm(
        tool_call_message=_tool_call_message("capital of france"),
        final_message=_final_message("Paris [1]."),
    ):
        with (
            patch(
                "app.domain.research.service.search_web",
                new=AsyncMock(return_value=[source_a, source_b]),
            ),
            patch(
                "app.domain.research.service.extract_page_text",
                new=AsyncMock(side_effect=lambda url: extract_by_url[url]),
            ),
            patch(
                "app.domain.research.service.embed_texts",
                new=AsyncMock(return_value=[[1.0, 0.0], [0.0, 1.0]]),
            ),
            patch(
                "app.domain.research.service.embed_text",
                new=AsyncMock(return_value=[1.0, 0.0]),
            ),
        ):
            answer = await ask_research_question(question)

    assert isinstance(answer, Answer)
    assert answer.answer_text == "Paris [1]."
    assert len(answer.sources) == 2
    # answer.sources[i].url is a pydantic HttpUrl (Source.url's field
    # type), not a plain str, so both sides need str() to compare equal.
    returned_urls = {str(source.url) for source in answer.sources}
    assert returned_urls == {str(source_a.url), str(source_b.url)}


async def test_ask_deduplicates_identical_chunks_across_sources(question):
    source_a = _source("https://a.example.com/page", "Source A")
    source_b = _source("https://b.example.com/page", "Mirror of Source A")

    # Same text, different sources — a common case for syndicated or
    # mirrored content.
    identical_text = "Paris is the capital of France."
    extract_by_url = {
        str(source_a.url): identical_text,
        str(source_b.url): identical_text,
    }

    embed_texts_mock = AsyncMock(return_value=[[1.0, 0.0]])

    with _patch_llm(
        tool_call_message=_tool_call_message("capital of france"),
        final_message=_final_message("Paris."),
    ):
        with (
            patch(
                "app.domain.research.service.search_web",
                new=AsyncMock(return_value=[source_a, source_b]),
            ),
            patch(
                "app.domain.research.service.extract_page_text",
                new=AsyncMock(side_effect=lambda url: extract_by_url[url]),
            ),
            patch("app.domain.research.service.embed_texts", new=embed_texts_mock),
            patch(
                "app.domain.research.service.embed_text",
                new=AsyncMock(return_value=[1.0, 0.0]),
            ),
        ):
            answer = await ask_research_question(question)

    # Only one distinct chunk should ever reach embed_texts, even though
    # two sources produced text.
    embed_texts_mock.assert_awaited_once_with([identical_text])
    # And only the first source that produced it is kept.
    assert len(answer.sources) == 1
    # answer.sources[0].url is a pydantic HttpUrl, not a plain str — see
    # the same note in the test above.
    assert str(answer.sources[0].url) == str(source_a.url)


async def test_ask_fetches_sources_concurrently_not_sequentially(question):
    source_a = _source("https://a.example.com/page", "A")
    source_b = _source("https://b.example.com/page", "B")
    source_c = _source("https://c.example.com/page", "C")

    # Each fake fetch takes 100ms. If _run_search_tool fetched sequentially,
    # three sources would take >= 300ms; run concurrently (asyncio.gather,
    # ref: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather),
    # it should take roughly one fetch's worth of time.
    async def slow_extract(url):
        await asyncio.sleep(0.1)
        return f"content from {url}"

    with _patch_llm(
        tool_call_message=_tool_call_message("some query"),
        final_message=_final_message("Answer."),
    ):
        with (
            patch(
                "app.domain.research.service.search_web",
                new=AsyncMock(return_value=[source_a, source_b, source_c]),
            ),
            patch(
                "app.domain.research.service.extract_page_text",
                new=AsyncMock(side_effect=slow_extract),
            ),
            patch(
                "app.domain.research.service.embed_texts",
                new=AsyncMock(
                    side_effect=lambda texts: [[float(i)] for i in range(len(texts))]
                ),
            ),
            patch(
                "app.domain.research.service.embed_text",
                new=AsyncMock(return_value=[0.0]),
            ),
        ):
            # Ref: time.monotonic — https://docs.python.org/3/library/time.html#time.monotonic
            start = time.monotonic()
            await ask_research_question(question)
            elapsed = time.monotonic() - start

    # Generous upper bound (well under the 300ms a sequential
    # implementation would need) to avoid flakiness on a loaded CI box,
    # while still failing loudly if concurrency regresses to sequential.
    assert elapsed < 0.25


async def test_ask_handles_empty_search_results_gracefully(question):
    with _patch_llm(
        tool_call_message=_tool_call_message("obscure query"),
        final_message=_final_message("I don't have enough information to answer that."),
    ):
        with (
            patch(
                "app.domain.research.service.search_web",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "app.domain.research.service.extract_page_text", new=AsyncMock()
            ) as extract_mock,
            patch(
                "app.domain.research.service.embed_texts", new=AsyncMock()
            ) as embed_texts_mock,
            patch(
                "app.domain.research.service.embed_text", new=AsyncMock()
            ) as embed_text_mock,
        ):
            answer = await ask_research_question(question)

    assert answer.sources == []
    # None of the downstream retrieval steps should run over an empty result set.
    extract_mock.assert_not_called()
    embed_texts_mock.assert_not_called()
    embed_text_mock.assert_not_called()


async def test_ask_rejects_malformed_tool_call_arguments(question):
    malformed_message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="search_web", arguments="not valid json"),
            )
        ],
    )

    with patch(
        "app.domain.research.service.get_completion_with_tools",
        new=AsyncMock(return_value=malformed_message),
    ):
        with pytest.raises(LLMResponseError):
            await ask_research_question(question)


async def test_ask_translates_search_failure_into_retrieval_unavailable(question):
    with _patch_llm(
        tool_call_message=_tool_call_message("some query"),
        final_message=_final_message("unused"),
    ):
        with patch(
            "app.domain.research.service.search_web",
            new=AsyncMock(side_effect=RuntimeError("Tavily is down")),
        ):
            with pytest.raises(RetrievalUnavailableError):
                await ask_research_question(question)


async def test_ask_translates_embedding_failure_into_retrieval_unavailable(question):
    source_a = _source("https://a.example.com/page", "A")

    with _patch_llm(
        tool_call_message=_tool_call_message("some query"),
        final_message=_final_message("unused"),
    ):
        with (
            patch(
                "app.domain.research.service.search_web",
                new=AsyncMock(return_value=[source_a]),
            ),
            patch(
                "app.domain.research.service.extract_page_text",
                new=AsyncMock(return_value="Paris is the capital of France."),
            ),
            patch(
                "app.domain.research.service.embed_texts",
                new=AsyncMock(
                    side_effect=EmbeddingModelUnavailableError("model unavailable")
                ),
            ),
        ):
            with pytest.raises(RetrievalUnavailableError):
                await ask_research_question(question)


async def test_ask_continues_when_one_source_fetch_raises_unexpectedly(question):
    # extractor.py returns None for known failure modes (timeout, HTTP
    # error, no extractable content), but a programming bug in the
    # extractor — or a malformed URL that httpx surfaces as an
    # unexpected exception type — must NOT take down the whole request.
    # The contract is "partial sources still produce a useful answer".
    # Refs:
    # - asyncio.gather + per-task exception: this depends on the
    #   in-process gather in service.py raising per-task and the
    #   _process_source wrapper swallowing it to an empty list.
    #   https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
    good_source = _source("https://good.example.com/page", "Good")
    bad_source = _source("https://bad.example.com/page", "Bad")

    async def flaky_extract(url):
        if "bad" in url:
            raise RuntimeError("simulated extractor bug")
        return "Paris is the capital of France."

    with _patch_llm(
        tool_call_message=_tool_call_message("capital of france"),
        final_message=_final_message("Paris [1]."),
    ):
        with (
            patch(
                "app.domain.research.service.search_web",
                new=AsyncMock(return_value=[good_source, bad_source]),
            ),
            patch(
                "app.domain.research.service.extract_page_text",
                new=AsyncMock(side_effect=flaky_extract),
            ),
            patch(
                "app.domain.research.service.embed_texts",
                new=AsyncMock(return_value=[[1.0, 0.0]]),
            ),
            patch(
                "app.domain.research.service.embed_text",
                new=AsyncMock(return_value=[1.0, 0.0]),
            ),
        ):
            answer = await ask_research_question(question)

    # The good source must survive, the bad source must be excluded.
    assert answer.answer_text == "Paris [1]."
    assert len(answer.sources) == 1
    assert str(answer.sources[0].url) == str(good_source.url)


async def test_ask_sends_no_sources_block_to_llm_when_all_sources_fail(question):
    # All sources raise — the model should still be invoked with the
    # question, but with the "(none retrieved)" branch of the prompt
    # (see prompt_builder.py). LLMResponseError if the model then
    # returns empty content, but we test the more useful case: the
    # model answers honestly from its own knowledge.
    bad_source = _source("https://bad.example.com/page", "Bad")

    async def always_fails(_url):
        raise RuntimeError("simulated extractor bug")

    with _patch_llm(
        tool_call_message=_tool_call_message("capital of france"),
        final_message=_final_message("I could not retrieve any sources."),
    ):
        with (
            patch(
                "app.domain.research.service.search_web",
                new=AsyncMock(return_value=[bad_source]),
            ),
            patch(
                "app.domain.research.service.extract_page_text",
                new=AsyncMock(side_effect=always_fails),
            ),
            patch(
                "app.domain.research.service.embed_texts",
                new=AsyncMock(),
            ),
            patch(
                "app.domain.research.service.embed_text",
                new=AsyncMock(),
            ),
        ):
            answer = await ask_research_question(question)

    # No sources survived, so Answer.sources must be empty even though
    # the model was called.
    assert answer.answer_text == "I could not retrieve any sources."
    assert answer.sources == []


async def test_ask_final_answer_sources_match_prompt_citation_order(question):
    # The prompt template numbers sources [1], [2], ... and instructs the
    # model to cite them inline. Today, Answer.sources is populated from
    # the ranked top_k (in cosine-similarity order), and the prompt is
    # built from that same list, so the numbering in the prompt and the
    # order of Answer.sources must agree. This test pins that contract:
    # if a future refactor decouples "what the model sees" from "what
    # gets returned", citations like "[2]" in the answer would silently
    # point at the wrong source.
    source_a = _source("https://a.example.com/page", "Source A")
    source_b = _source("https://b.example.com/page", "Source B")

    # Force source_a's text to embed to a vector with higher similarity
    # than source_b's, so source_a is the [1] in the prompt.
    extract_by_url = {
        str(source_a.url): "Paris is the capital of France.",
        str(source_b.url): "Berlin is the capital of Germany.",
    }

    with _patch_llm(
        tool_call_message=_tool_call_message("capital of france"),
        final_message=_final_message("Paris [1]."),
    ):
        with (
            patch(
                "app.domain.research.service.search_web",
                new=AsyncMock(return_value=[source_a, source_b]),
            ),
            patch(
                "app.domain.research.service.extract_page_text",
                new=AsyncMock(side_effect=lambda url: extract_by_url[url]),
            ),
            patch(
                "app.domain.research.service.embed_texts",
                # source_a → [1,0,0], source_b → [0,1,0]; query → [1,0,0].
                new=AsyncMock(return_value=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            ),
            patch(
                "app.domain.research.service.embed_text",
                new=AsyncMock(return_value=[1.0, 0.0, 0.0]),
            ),
        ):
            answer = await ask_research_question(question)

    # The highest-similarity chunk is source_a, so it must be sources[0]
    # and correspond to "[1]" in the prompt.
    assert len(answer.sources) == 2
    assert str(answer.sources[0].url) == str(source_a.url)
    assert str(answer.sources[1].url) == str(source_b.url)
    # The model cited "[1]" and source_a is at index 0.
    assert "Paris [1]." in answer.answer_text


async def test_ask_full_happy_path_sources_align_end_to_end(question):
    # End-to-end cross-layer integration test: every boundary is mocked
    # at its public surface (search_web, extract_page_text, embed_texts,
    # embed_text, get_completion_with_tools), and the assertion checks
    # that the chunk text the user sees as "the answer's sources"
    # actually matches the chunk text the model saw in the prompt.
    # This is the test most likely to catch a "prompt/sources drift"
    # bug introduced by a future refactor.
    source = _source("https://example.com/article", "Example")

    with _patch_llm(
        tool_call_message=_tool_call_message("the example query"),
        final_message=_final_message("Based on the source, X is true [1]."),
    ):
        with (
            patch(
                "app.domain.research.service.search_web",
                new=AsyncMock(return_value=[source]),
            ),
            patch(
                "app.domain.research.service.extract_page_text",
                new=AsyncMock(return_value="The article body. " * 50),
            ),
            patch(
                "app.domain.research.service.embed_texts",
                new=AsyncMock(return_value=[[1.0, 0.0]]),
            ),
            patch(
                "app.domain.research.service.embed_text",
                new=AsyncMock(return_value=[1.0, 0.0]),
            ),
        ):
            answer = await ask_research_question(question)

    assert answer.answer_text == "Based on the source, X is true [1]."
    assert len(answer.sources) == 1
    assert str(answer.sources[0].url) == str(source.url)
    assert answer.sources[0].title == "Example"
    # relevance_score must survive the round-trip from the vector store.
    assert answer.sources[0].relevance_score is not None
    assert answer.sources[0].relevance_score > 0
