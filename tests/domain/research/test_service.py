from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.domain.research.answer import Answer
from app.domain.research.exceptions import (
    EmptyQuestionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
)
from app.domain.research.question import Question
from app.domain.research.service import ask_research_question

# Ref: https://pytest-asyncio.readthedocs.io/en/stable/reference/markers/index.html
# This marks every test in the module as asyncio so we don't have to decorate each async test function one by one.
pytestmark = pytest.mark.asyncio

_FAKE_URL = "https://openrouter.ai/api/v1/chat/completions"


# Ref: https://github.com/openai/openai-python/blob/main/src/openai/_exceptions.py
# These helpers build fake errors the same way the openai SDK raises them for real, using
# actual httpx Request and Response objects instead of plain mocks, since the SDK reads
# things like status_code and headers straight off those objects.
def _make_rate_limit_error() -> RateLimitError:
    request = httpx.Request("POST", _FAKE_URL)
    response = httpx.Response(status_code=429, request=request, json={})
    return RateLimitError("rate limited", response=response, body={})


def _make_connection_error() -> APIConnectionError:
    # This error has no response attached at all, since it represents a request that
    # never got one back, like a DNS failure or a refused connection.
    request = httpx.Request("POST", _FAKE_URL)
    return APIConnectionError(request=request)


def _make_timeout_error() -> APITimeoutError:
    # This subclasses APIConnectionError and has the same shape, just a different
    # cause, and service.py catches both of them with the same except clause.
    request = httpx.Request("POST", _FAKE_URL)
    return APITimeoutError(request=request)


def _make_status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", _FAKE_URL)
    response = httpx.Response(
        status_code=status_code, request=request, json={"error": {"message": "boom"}}
    )
    return APIStatusError("provider error", response=response, body={})


@pytest.fixture
def question() -> Question:
    # Ref: https://docs.pydantic.dev/latest/concepts/fields/#default-values
    # Pydantic fills in id, status and created_at automatically, so this fixture only
    # needs to set question_text, which is padded with spaces on purpose so the stripping test below can reuse it too.
    return Question(question_text="  what is the capital of france?  ")


def _patch_get_completion(**kwargs) -> patch:
    # Refs:
    # - https://docs.python.org/3/library/unittest.mock.html#where-to-patch
    # - https://docs.python.org/3/library/unittest.mock.html#unittest.mock.AsyncMock
    #
    # We patch get_completion where service.py imported it instead of where it's defined in
    # infrastructure/llm/client.py, because patching the definition site would leave service.py's
    # own reference untouched and the real function would still run.
    #
    # AsyncMock is needed here instead of a plain MagicMock because service.py awaits this call.
    return patch("app.domain.research.service.get_completion", new=AsyncMock(**kwargs))


@pytest.mark.parametrize(
    "question_text",
    ["", "   ", "\n\t  "],
    ids=["empty", "spaces", "mixed-whitespace"],
)
# Ref: https://docs.pytest.org/en/stable/how-to/parametrize.html
# This covers a fully empty string, a string with only spaces and a string with mixed
# whitespace characters in a single test instead of writing three nearly identical ones.
async def test_ask_rejects_blank_question_without_calling_llm(question_text):
    blank_question = Question(question_text=question_text)

    with _patch_get_completion(return_value="should never be reached") as mocked:
        # Ref: https://docs.pytest.org/en/stable/how-to/assertions.html#assertions-about-expected-exceptions
        with pytest.raises(EmptyQuestionError):
            await ask_research_question(blank_question)

        # If validation ever gets reordered to run after the LLM call, this assertion fails
        # loudly instead of the bug hiding behind a test that only checked the exception type.
        mocked.assert_not_called()


async def test_ask_returns_answer_built_from_completion(question):
    with _patch_get_completion(return_value="Paris."):
        answer = await ask_research_question(question)

    assert isinstance(answer, Answer)
    assert answer.answer_text == "Paris."
    assert answer.question_id == question.id
    assert answer.sources == []


async def test_ask_sends_stripped_question_text_to_llm(question):
    # The question fixture carries leading and trailing whitespace on purpose,
    # so the provider should only ever see the trimmed text.
    with _patch_get_completion(return_value="Paris.") as mocked:
        await ask_research_question(question)

    # Ref: https://docs.python.org/3/library/unittest.mock.html#unittest.mock.AsyncMock.assert_awaited_once_with
    mocked.assert_awaited_once_with("what is the capital of france?")


async def test_ask_translates_rate_limit_error(question):
    # RateLimitError is a subclass of APIStatusError, so service.py MUST catch it in an
    # except clause that comes before the generic APIStatusError clause; otherwise a 429
    # would silently fall into the APIStatusError branch and come out as LLMUnavailableError
    # instead of LLMRateLimitError. This test alone can't prove the except order is correct
    # (that's covered by test_ask_translates_rate_limit_error_takes_precedence_over_status_error
    # below), it only proves today's behavior is the right one.
    with _patch_get_completion(side_effect=_make_rate_limit_error()):
        with pytest.raises(LLMRateLimitError):
            await ask_research_question(question)


async def test_ask_translates_rate_limit_error_takes_precedence_over_status_error(
    question,
):
    # Same setup as test_ask_translates_rate_limit_error, but this one exists specifically to
    # guard the except-clause ORDER in service.py. Because RateLimitError IS-A APIStatusError,
    # swapping the two except clauses (or merging them) would make this raise LLMUnavailableError
    # instead, but the test above would keep passing as long as *some* exception with "rate
    # limit" semantics came out - it doesn't fail loudly on that specific regression. Asserting
    # the exact type here, plus a status_code that only makes sense as a rate limit, closes that gap.
    error = _make_rate_limit_error()
    assert isinstance(
        error, APIStatusError
    )  # sanity check on the inheritance this test relies on

    with _patch_get_completion(side_effect=error):
        with pytest.raises(LLMRateLimitError) as exc_info:
            await ask_research_question(question)

    assert not isinstance(exc_info.value, LLMUnavailableError)


@pytest.mark.parametrize(
    "make_error",
    [_make_connection_error, _make_timeout_error],
    ids=["connection-error", "timeout-error"],
)
# service.py catches APIConnectionError and APITimeoutError with a single except clause, so
# this test proves both errors really land there instead of assuming it from just one of them.
async def test_ask_translates_network_level_errors(question, make_error):
    with _patch_get_completion(side_effect=make_error()):
        with pytest.raises(LLMUnavailableError):
            await ask_research_question(question)


@pytest.mark.parametrize("status_code", [500, 502, 503])
async def test_ask_translates_server_status_errors(question, status_code):
    with _patch_get_completion(side_effect=_make_status_error(status_code)):
        # The match argument also proves the status code survives into the translated error
        # message, which is handy for debugging without needing the raw logs.
        with pytest.raises(LLMUnavailableError, match=str(status_code)):
            await ask_research_question(question)


@pytest.mark.parametrize("status_code", [400, 404, 422])
# service.py's except APIStatusError clause is generic: it doesn't branch on status_code, so
# a 4xx from the provider (e.g. we sent a malformed request, or an invalid model name) is
# translated into LLMUnavailableError exactly the same way a 5xx is. That's arguably a smell -
# "the provider is unavailable" isn't really true when the problem is on our side of the
# request - but it's today's actual, intentional behavior. This test exists to document that
# choice explicitly, so if someone later decides 4xx deserves its own domain exception, they
# have to consciously change this test instead of discovering the gap by accident.
async def test_ask_translates_client_status_errors_as_unavailable(
    question, status_code
):
    with _patch_get_completion(side_effect=_make_status_error(status_code)):
        with pytest.raises(LLMUnavailableError, match=str(status_code)):
            await ask_research_question(question)


async def test_ask_reraises_unexpected_exception_unchanged(question):
    with _patch_get_completion(side_effect=ValueError("unexpected")):
        with pytest.raises(ValueError, match="unexpected"):
            await ask_research_question(question)


@pytest.mark.parametrize(
    "content",
    ["", "   \n  ", None],
    ids=["empty-string", "whitespace-only", "none"],
)
# Ref: https://platform.openai.com/docs/api-reference/chat/object
# None is a documented possibility for message.content, for example when finish_reason
# comes back as content_filter, so it's tested here alongside the empty and whitespace cases.
async def test_ask_rejects_unusable_content(question, content):
    with _patch_get_completion(return_value=content):
        with pytest.raises(LLMResponseError):
            await ask_research_question(question)


# Ref: https://docs.python.org/3/library/exceptions.html#exception-context
# service.py re-raises every translated exception with "raise ... from exc", which sets
# __cause__ to the original openai SDK exception. This matters in production: without it,
# whoever reads the traceback (logs, Sentry, etc.) sees only "LLMRateLimitError: Provider
# rate limit exceeded." with no way back to the real RateLimitError/APIStatusError that
# caused it. None of the tests above check this, since they only assert on exception type
# and message, so a stray refactor that drops "from exc" wouldn't be caught anywhere else.
@pytest.mark.parametrize(
    ("make_error", "expected_domain_exception"),
    [
        (_make_rate_limit_error, LLMRateLimitError),
        (_make_connection_error, LLMUnavailableError),
        (_make_timeout_error, LLMUnavailableError),
        (lambda: _make_status_error(503), LLMUnavailableError),
    ],
    ids=["rate-limit", "connection-error", "timeout-error", "server-status-error"],
)
async def test_ask_chains_original_sdk_exception_as_cause(
    question, make_error, expected_domain_exception
):
    original_error = make_error()

    with _patch_get_completion(side_effect=original_error):
        with pytest.raises(expected_domain_exception) as exc_info:
            await ask_research_question(question)

    assert exc_info.value.__cause__ is original_error
