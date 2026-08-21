# Refs:
#   OpenRouter API reference: https://openrouter.ai/docs/api_reference/overview
#   OpenAI Python client (error types used below): https://github.com/openai/openai-python/
#   OpenRouter rate limits: https://openrouter.ai/docs/api_reference/limits
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


# Async usage ref: https://github.com/openai/openai-python#async-usage
@retry(
    retry=retry_if_exception(is_retryable),
    wait=wait_with_retry_after(wait_random_exponential(multiplier=1, max=60)),
    stop=stop_after_attempt(6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
async def get_completion(content: str):
    # Response schema ref: https://developers.openai.com/api/reference/resources/chat
    completion = await client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
            "X-OpenRouter-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
        },
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
    )

    # Only the generated text is relevant to callers; the rest of the
    # response object is discarded here.
    return completion.choices[0].message.content
