# Refs:
#   pytest: https://docs.pytest.org/en/stable/
#   pytest usage: https://docs.pytest.org/en/stable/how-to/usage.html
#   pytest-asyncio: https://pytest-asyncio.readthedocs.io/en/latest/
#   httpx.Response/Request, used to build realistic fake HTTP errors: https://www.python-httpx.org/api/
import time

import httpx
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.infrastructure.llm.client import (
    get_completion,
    is_retryable,
    _extract_retry_after,
    _headers_from_json_body,
)
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
)


def _fake_request() -> httpx.Request:
    return httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")


def _fake_response(
    status_code: int, headers: dict | None = None, json_body: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=_fake_request(),
        headers=headers or {},
        json=json_body if json_body is not None else {"error": {"message": "error"}},
    )


# Ref: https://pytest-asyncio.readthedocs.io/en/latest/how-to-guides/markers.html
@pytest.mark.asyncio
async def test_get_completion_returns_model_text():
    # We are mocking the objects here to simulate what the OpenAI/OpenRouter API would actually return.
    # Since `completion.choices[0].message.content` is a chain of nested attributes, we use `MagicMock`,
    # which accepts any attribute without requiring us to define an actual class for it.
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "Simulated response by AI"

    # Refs:
    #   where to patch: https://realpython.com/python-mock-library/#knowing-where-to-patch
    #   patch documentation: https://docs.python.org/3/library/unittest.mock-examples.html#mocking-classes
    #   AsyncMock: https://docs.python.org/3/library/unittest.mock.html#unittest.mock.AsyncMock
    #
    # patch() temporarily replaces the specified object, ONLY during the "with" block. Note the path
    # used: it is not "openai.AsyncOpenAI", but rather where the client is USED within the
    # infrastructure.llm.client module.
    with patch("app.infrastructure.llm.client.client") as mock_client:
        # We need to use AsyncMock here because a simple MagicMock would return itself,
        # instead of "awaitable" and the `await` inside get_completion would failed.
        mock_client.chat.completions.create = AsyncMock(return_value=fake_completion)

        result = await get_completion("What is the captal of Brazil?")

        assert result == "Simulated response by AI"
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["messages"][0]["content"] == "What is the captal of Brazil?"


# is_retryable decides which errors trigger a retry. Only transient/rate-limit errors should;
# permanent client errors (4xx other than 429) must fail fast instead of wasting retry budget.
@pytest.mark.parametrize(
    "exc, expected",
    [
        (RateLimitError("rate limited", response=_fake_response(429), body=None), True),
        (APIConnectionError(request=_fake_request()), True),
        (APITimeoutError(request=_fake_request()), True),
        (
            APIStatusError(
                "too many requests", response=_fake_response(429), body=None
            ),
            True,
        ),
        (APIStatusError("bad request", response=_fake_response(400), body=None), False),
        (
            APIStatusError("server error", response=_fake_response(500), body=None),
            False,
        ),
        (ValueError("unrelated error"), False),
    ],
)
def test_is_retryable(exc, expected):
    assert is_retryable(exc) is expected


# _extract_retry_after is the most fragile piece of this module: it parses real HTTP headers,
# falls back to the OpenRouter-specific body.error.metadata.headers shape, and normalizes
# X-RateLimit-Reset whether it's epoch ms, epoch s, or a plain duration.
def test_extract_retry_after_from_standard_http_header():
    response = _fake_response(429, headers={"retry-after": "2.5"})
    exc = APIStatusError("rate limited", response=response, body=None)

    assert _extract_retry_after(exc) == 2.5


def test_extract_retry_after_from_json_body_metadata_when_no_http_header():
    # Ref: github.com/BerriAI/litellm/issues/9035
    body = {
        "error": {
            "message": "rate limited",
            "metadata": {"headers": {"Retry-After": "3"}},
        }
    }
    response = _fake_response(429, json_body=body)
    exc = APIStatusError("rate limited", response=response, body=body)

    assert _extract_retry_after(exc) == 3.0


def test_extract_retry_after_prefers_real_http_headers_over_body():
    body = {"error": {"metadata": {"headers": {"Retry-After": "999"}}}}
    response = _fake_response(429, headers={"retry-after": "1"}, json_body=body)
    exc = APIStatusError("rate limited", response=response, body=body)

    # Real HTTP headers must take precedence over the ones embedded in the body.
    assert _extract_retry_after(exc) == 1.0


def test_extract_retry_after_handles_epoch_milliseconds_reset():
    future_ms = str(int((time.time() + 5) * 1000))
    response = _fake_response(429, headers={"x-ratelimit-reset": future_ms})
    exc = APIStatusError("rate limited", response=response, body=None)

    wait = _extract_retry_after(exc)
    assert wait is not None
    # Tolerance for the time elapsed between building the header and this assert.
    assert 4.0 <= wait <= 5.5


def test_extract_retry_after_returns_none_when_no_headers_present():
    exc = APIStatusError("rate limited", response=_fake_response(429), body=None)
    assert _extract_retry_after(exc) is None


def test_headers_from_json_body_ignores_malformed_shapes():
    class FakeExc(Exception):
        body = {"error": "not-a-dict"}

    assert _headers_from_json_body(FakeExc()) == {}


# Boundary tests for the X-RateLimit-Reset heuristic in _extract_retry_after.
# The heuristic has three regimes (epoch ms, epoch s, plain duration) and
# edge cases in each: a value just below 1e9 must NOT be parsed as epoch
# seconds, a value of exactly 1e9 must be, and a negative duration must
# be clamped to zero. These tests pin those boundaries so a future
# "off-by-one" refactor doesn't quietly turn a real bug into "wait 0s".
def test_extract_retry_after_x_ratelimit_reset_below_one_billion_is_duration():
    # 999_999_999 < 1e9 → must be parsed as a plain duration in seconds,
    # not as an epoch-second timestamp. Today this is exactly 31 years
    # before unix epoch, which would be a wildly negative wait, so this
    # test catches the threshold bug specifically.
    response = _fake_response(429, headers={"x-ratelimit-reset": "999999999"})
    exc = APIStatusError("rate limited", response=response, body=None)

    assert _extract_retry_after(exc) == 999999999.0


def test_extract_retry_after_x_ratelimit_reset_at_one_billion_is_epoch_seconds():
    # Exactly at the 1e9 threshold → must be parsed as epoch seconds,
    # which yields a clearly negative wait that gets clamped to 0.
    response = _fake_response(429, headers={"x-ratelimit-reset": "1000000000"})
    exc = APIStatusError("rate limited", response=response, body=None)

    wait = _extract_retry_after(exc)
    assert wait is not None
    assert wait == 0.0


def test_extract_retry_after_x_ratelimit_reset_negative_value_is_clamped():
    # A provider bug or signed integer underflow could give us a
    # negative duration. Must clamp to 0, not return the negative value
    # (which tenacity would forward to asyncio.sleep — silent breakage).
    response = _fake_response(429, headers={"x-ratelimit-reset": "-5"})
    exc = APIStatusError("rate limited", response=response, body=None)

    assert _extract_retry_after(exc) == 0.0


def test_extract_retry_after_x_ratelimit_reset_unparseable_returns_none():
    # Header value isn't a valid float (rare but possible if a proxy
    # corrupts it) — must return None, not raise. The caller falls back
    # to exponential backoff.
    response = _fake_response(429, headers={"x-ratelimit-reset": "not-a-number"})
    exc = APIStatusError("rate limited", response=response, body=None)

    assert _extract_retry_after(exc) is None


def test_extract_retry_after_http_date_string_returns_none():
    # Per RFC 7231 Retry-After can be either a delta-seconds or an
    # HTTP-date. This implementation only handles delta-seconds; HTTP-date
    # is documented as "not handled here for simplicity". If a future
    # contributor removes that comment without adding date parsing, this
    # test fails loudly.
    response = _fake_response(
        429, headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}
    )
    exc = APIStatusError("rate limited", response=response, body=None)

    assert _extract_retry_after(exc) is None


# End-to-end retry behavior of get_completion. asyncio.sleep is patched so these tests don't
# actually wait through real backoff delays.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_error",
    [
        lambda: RateLimitError("rate limited", response=_fake_response(429), body=None),
        lambda: APIConnectionError(request=_fake_request()),
        lambda: APITimeoutError(request=_fake_request()),
    ],
)
async def test_get_completion_retries_transient_errors_then_succeeds(make_error):
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "Recovered after retry"

    with (
        patch("app.infrastructure.llm.client.client") as mock_client,
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[make_error(), make_error(), fake_completion]
        )

        result = await get_completion("Ping")

        assert result == "Recovered after retry"
        assert mock_client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_get_completion_does_not_retry_non_retryable_error():
    non_retryable = APIStatusError(
        "bad request", response=_fake_response(400), body=None
    )

    with (
        patch("app.infrastructure.llm.client.client") as mock_client,
        patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        mock_client.chat.completions.create = AsyncMock(side_effect=non_retryable)

        with pytest.raises(APIStatusError):
            await get_completion("Ping")

        # Non-transient error: fails on the first attempt, no wasted wait.
        mock_client.chat.completions.create.assert_called_once()
        mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_get_completion_reraises_after_exhausting_retries():
    persistent_error = APITimeoutError(request=_fake_request())

    with (
        patch("app.infrastructure.llm.client.client") as mock_client,
        patch("asyncio.sleep", new=AsyncMock()),
    ):
        mock_client.chat.completions.create = AsyncMock(side_effect=persistent_error)

        with pytest.raises(APITimeoutError):
            await get_completion("Ping")

        # stop_after_attempt(6): should have tried exactly 6 times before giving up,
        # and reraise=True means the original exception type must propagate, not
        # a generic Tenacity RetryError.
        assert mock_client.chat.completions.create.call_count == 6


@pytest.mark.asyncio
async def test_get_completion_waits_according_to_retry_after_header():
    body = {"error": {"metadata": {"headers": {"Retry-After": "7"}}}}
    error_with_retry_after = RateLimitError(
        "rate limited", response=_fake_response(429, json_body=body), body=body
    )
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "ok"

    with (
        patch("app.infrastructure.llm.client.client") as mock_client,
        patch("asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[error_with_retry_after, fake_completion]
        )

        await get_completion("Ping")

        # The wait must honor the provider-suggested value (7s), not the random
        # exponential backoff fallback.
        mock_sleep.assert_awaited_once_with(7.0)


@pytest.mark.asyncio
async def test_get_completion_forwards_extra_headers():
    # HTTP-Referer and X-OpenRouter-Title are sent for OpenRouter rankings; a regression
    # here wouldn't break functionality but would silently drop that metadata.
    # Ref: https://openrouter.ai/docs/api_reference/overview
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "ok"

    with patch("app.infrastructure.llm.client.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=fake_completion)

        await get_completion("Ping")

        _, kwargs = mock_client.chat.completions.create.call_args
        assert "HTTP-Referer" in kwargs["extra_headers"]
        assert "X-OpenRouter-Title" in kwargs["extra_headers"]


@pytest.mark.asyncio
async def test_get_completion_uses_configured_model():
    # Ref: infrastructure.llm.config.OPENROUTER_MODEL
    from app.infrastructure.llm.config import OPENROUTER_MODEL

    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "ok"

    with patch("app.infrastructure.llm.client.client") as mock_client:
        mock_client.chat.completions.create = AsyncMock(return_value=fake_completion)

        await get_completion("Ping")

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == OPENROUTER_MODEL
