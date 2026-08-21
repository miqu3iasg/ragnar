# This module owns exactly one responsibility of translate infrastructure level
# failures (openai SDK exceptions) into domain-level exceptions that the API
# layer (api/routes/research.py) knows how to map to HTTP status codes.
import logging

# openai SDK exception hierarchy ref: https://github.com/openai/openai-python/
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.domain.research.question import Question
from app.domain.research.answer import Answer

from app.domain.research.exceptions import (
    EmptyQuestionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
)

from app.infrastructure.llm.client import get_completion

logger = logging.getLogger(__name__)


async def ask_research_question(question: Question) -> Answer:
    # A question that is only whitespace should be treated the same as an
    # empty question, not silently sent to the LLM.
    question_text = question.question_text.strip()
    if not question_text:
        raise EmptyQuestionError("Question content must not be empty.")

    try:
        answer_text = await get_completion(question_text)
    except RateLimitError as exc:
        # ref: https://tenacity.readthedocs.io/en/latest/
        #
        # By the time we get here, tenacity's @retry in client.py has already
        # exhausted and reraised the original exception, so this is the final
        # rate limit failure, not the first one.
        logger.warning(f"Rate limit persisted after retries exhausted: {exc}")
        raise LLMRateLimitError("Provider rate limit exceeded.") from exc
    except (APIConnectionError, APITimeoutError) as exc:
        # Network-level failures like DNS resolution, connection refused, or
        # the request timing out before a response was received at all.
        logger.error(f"Connection to LLM provider failed: {exc}")
        raise LLMUnavailableError("Could not reach the LLM provider.") from exc
    except APIStatusError as exc:
        # Any other non-2xx response not already caught as RateLimitError
        # above (e.g. 500, 502, 503 from OpenRouter or the upstream model).
        logger.error(
            f"LLM provider returned an error (status={exc.status_code}): {exc}"
        )
        raise LLMUnavailableError(
            f"LLM provider returned an error (status {exc.status_code})."
        ) from exc
    except Exception:
        # Catch all safety net like anything not in the four branches above
        # (a bug in our own code, an unexpected SDK exception type, an SDK
        # version bump that changes behavior) previously propagated as an
        # unlogged bare exception, surfacing to the client as a generic,
        # untraceable 500. Log with full context here, then re-raise
        # unchanged; we deliberately do NOT wrap this in a domain
        # exception, because we don't know what it is; wrapping it would
        # hide the real cause from logs/monitoring.
        logger.exception(
            f"Unexpected error while processing question: {question_text!r}"
        )
        raise

    # The call succeeded wiht no exception, but the content itself may still be
    # unusable.
    #
    # Known cause: OpenRouter/OpenAI can return an empty string
    # when finish_reason is "content_filter" or occasionally "length".
    # Silently returning Answer(answer_text="") here would produce a 200 OK
    # response that looks successful but carries no useful information,
    # worse than an explicit error, because it fails silently downstream.
    if not answer_text or not answer_text.strip():
        logger.error(
            f"LLM provider returned a empty response for question: {question_text!r}"
        )
        raise LLMResponseError("The model did not return a usable response.")

    return Answer(question_id=question.id, answer_text=answer_text, sources=[])
