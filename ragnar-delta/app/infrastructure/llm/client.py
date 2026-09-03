# Refs:
#   OpenRouter API reference: https://openrouter.ai/docs/api_reference/overview
#   OpenAI Python client (error types used below): https://github.com/openai/openai-python/
#   OpenRouter rate limits: https://openrouter.ai/docs/api_reference/limits
#   OpenAI function calling / tool_calls message shape:
#   https://platform.openai.com/docs/guides/function-calling
#
# Retry-with-backoff lives in this module because it is the network boundary
# between our server and OpenRouter, and therefore the layer most exposed to
# transient failures (timeouts, connection drops, rate limiting).

import logging

# Error types: https://github.com/openai/openai-python
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

# Tenacity: https://tenacity.readthedocs.io/en/latest/
# Examples: https://github.com/jd/tenacity
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

# Environment variables are centralized in config.py to avoid repeated
# os.getenv calls scattered across the codebase.
from app.infrastructure.llm.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)

logger = logging.getLogger(__name__)

# max_retries=0 disables the OpenAI SDK's built-in retry mechanism.
# Leaving it enabled would stack two independent backoff strategies
# (the SDK's and Tenacity's), producing unpredictable wait times and
# obscuring the actual retry behavior.
client = AsyncOpenAI(
    base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, max_retries=0
)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError, APITimeoutError)):
        return True
    return bool(isinstance(exc, APIStatusError) and exc.status_code == 429)


def _headers_from_json_body(exc: BaseException) -> dict:
    """
    Extract rate-limit headers from the error's JSON body.

    OpenRouter occasionally embeds these headers under
    error.metadata.headers instead of returning them as actual
    HTTP response headers. See: github.com/BerriAI/litellm/issues/9035

    Expected shape:
    {
        "error": {
            "message": "...",
            "code": 429,
            "metadata": {
                "headers": {
                    "X-RateLimit-Limit": "80",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "1741305600000"
                }
            }
        }
    }
    """

    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return {}

    error = body.get("error")
    if not isinstance(error, dict):
        return {}

    metadata = error.get("metadata")
    if not isinstance(metadata, dict):
        return {}

    headers = metadata.get("headers")
    if not isinstance(headers, dict):
        return {}

    # Normalize keys to lowercase to match the convention used by actual HTTP headers.
    return {k.lower(): v for k, v in headers.items()}


def _extract_retry_after(exc: BaseException) -> float | None:
    """Resolve the provider-suggested wait time, in seconds, if available."""

    # Standard path: actual HTTP response headers (per OpenAI SDK docs).
    response = getattr(exc, "response", None)
    headers = dict(getattr(response, "headers", {})) if response is not None else {}

    # Fallback: headers embedded in the JSON body, as observed with OpenRouter.
    if not headers:
        headers = _headers_from_json_body(exc)
    else:
        # Merge both sources, giving precedence to actual HTTP headers.
        headers = {**_headers_from_json_body(exc), **headers}

    if not headers:
        return None

    # Standard HTTP Retry-After header, in seconds.
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except ValueError:
            pass  # May be an HTTP-date string; not handled here for simplicity.

    # X-RateLimit-Reset can be an epoch timestamp in milliseconds or seconds,
    # depending on the provider. Both cases are handled below.
    reset = headers.get("x-ratelimit-reset")
    if reset is not None:
        try:
            reset_value = float(reset)
        except ValueError:
            return None

        # Heuristic: sufficiently large values indicate epoch milliseconds.
        if reset_value >= 1e12:
            import time

            wait_seconds = (reset_value / 1000) - time.time()
        elif reset_value >= 1e9:
            # Epoch seconds.
            import time

            wait_seconds = reset_value - time.time()
        else:
            # Already expressed as a duration in seconds.
            wait_seconds = reset_value

        return max(wait_seconds, 0.0)

    return None


def wait_with_retry_after(fallback_wait):
    """Prefer the provider's suggested Retry-After; fall back to exponential backoff."""

    def wait_func(retry_state: RetryCallState) -> float:
        exc = retry_state.outcome.exception()
        if exc is not None:
            retry_after = _extract_retry_after(exc)
            if retry_after is not None:
                logger.warning(
                    f"Rate limit: waiting {retry_after:.1f}s (suggested by the provider)"
                )
                return retry_after

        return fallback_wait(retry_state)

    return wait_func


# Defined once and reused by both get_completion and get_completion_with_tools,
# so the retry/backoff policy lives in exactly one place instead of being
# redefined per function (and risking drift between the two).
_retry_policy = retry(
    retry=retry_if_exception(is_retryable),
    wait=wait_with_retry_after(wait_random_exponential(multiplier=1, max=60)),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


async def _create_completion(messages: list[dict], tools: list[dict] | None = None):
    """
    Low-level call to the chat completions endpoint.

    Returns the raw message object (not just its text), since tool-calling
    callers need to inspect message.tool_calls, not just message.content.
    get_completion (below) narrows this down to .content for callers that
    don't care about tools.
    """

    kwargs = {}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    # Response schema ref: https://developers.openai.com/api/reference/resources/chat
    completion = await client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
            "X-OpenRouter-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
        },
        model=OPENROUTER_MODEL,
        messages=messages,
        **kwargs,
    )

    return completion.choices[0].message


@_retry_policy
async def get_completion(content: str):
    """
    Simple single-turn completion, no tools involved.

    Kept as-is (existing tests target this function) for callers that
    just need a plain question-in, answer-out call.
    """

    message = await _create_completion([{"role": "user", "content": content}])

    # Only the generated text is relevant to callers; the rest of the
    # message object is discarded here.
    return message.content


@_retry_policy
async def get_completion_with_tools(
    messages: list[dict], tools: list[dict] | None = None
):
    """
    Multi-turn completion with optional tool definitions.

    Returns the raw assistant message so the caller (domain/research/
    service.py) can branch on whether the model answered directly
    (message.content) or requested a tool call (message.tool_calls) —
    unlike get_completion, this can't collapse the result to plain text,
    since the caller needs to see the tool_calls to act on them.
    """

    return await _create_completion(messages, tools=tools)
