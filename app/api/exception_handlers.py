# Refs:
#   - https://fastapi.tiangolo.com/tutorial/handling-errors/#install-custom-exception-handlers
#   - https://www.starlette.io/exceptions/
# Global FastAPI exception handlers.
#
# This module is the single translation boundary between domain exceptions
# (app.domain.research.exceptions) and HTTP responses. Before this file
# existed, api/routes/research.py had a try/except block repeating this
# mapping inline — see the OBS comment that used to live there. Moving it
# here means new routes that call ask_research_question (or any future
# service that raises ResearchError subclasses) get this mapping for free,
# without repeating try/except.
#
# How FastAPI/Starlette picks a handler: when an exception is raised,
# Starlette walks the exception's MRO (its own class, then parent classes,
# up to Exception) and uses the first class in that chain that has a
# registered handler. That's why ResearchError (the base class) can be
# registered as a catch-all fallback below: any future ResearchError
# subclass we forget to register explicitly still gets *a* handler
# instead of leaking as an unhandled 500 with no detail.

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domain.research.exceptions import (
    EmptyQuestionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
    ResearchError,
)

logger = logging.getLogger(__name__)


async def empty_question_handler(
    request: Request, exc: EmptyQuestionError
) -> JSONResponse:
    """
    The user sent an empty (or whitespace-only) question. Client error,
    not worth logging above INFO, since this is expected user input,
    not a system failure.
    """

    logger.info(f"Rejecting empty question: {exc}")

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)},
    )


async def llm_rate_limit_handler(
    request: Request, exc: LLMRateLimitError
) -> JSONResponse:
    """
    Handle rate-limit failures that survived client.py's retry-with-backoff.

    By the time this handler runs, tenacity has already exhausted its
    retries (see infrastructure/llm/client.py), so this is the final,
    non-recoverable rate limit failure for this request, not the first
    attempt. Returning 429 (instead of a generic 500) lets the caller
    apply its own backoff/retry logic on their side.
    Ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/429
    """

    logger.warning(f"Returning 429 to client after LLM rate limit: {exc}")

    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": str(exc)},
    )


async def llm_unavailable_handler(
    request: Request, exc: LLMUnavailableError
) -> JSONResponse:
    """
    Connection failure, timeout, or 5xx from the provider, after
    retries are exhausted. 503 signals "try again later", distinct from
    502 (llm_response_handler below), where the provider *did* answer.
    """

    logger.error(f"Returning 503 to client, LLM provider unavailable: {exc}")

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


async def llm_response_handler(request: Request, exc: LLMResponseError) -> JSONResponse:
    """
    The provider responded successfully but the content was unusable
    (empty string, content filter, etc — see LLMResponseError docstring).

    502 Bad Gateway, not 503: the upstream call itself succeeded, we're
    signaling that what it returned isn't something we can hand back to
    the client, not that the upstream is down.
    Ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/502
    """

    logger.error(f"Returning 502 to client, unusable LLM response: {exc}")

    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": str(exc)},
    )


async def research_error_fallback_handler(
    request: Request, exc: ResearchError
) -> JSONResponse:
    """
    Catch-all for any ResearchError subclass that doesn't have a more
    specific handler registered above (see ResearchError's own docstring
    in exceptions.py, which documents this exact intent).

    This only fires for a *future* exception type we add to exceptions.py
    and forget to register a specific handler for — every subclass that
    exists today (EmptyQuestionError, LLMRateLimitError,
    LLMUnavailableError, LLMResponseError) has its own handler above,
    which Starlette matches before falling back to this one.

    500 here, since we don't know the specific failure mode, unlike the
    handlers above, this isn't a status code we can justify semantically.
    """

    logger.exception(f"Unhandled ResearchError subtype reached fallback: {exc}")

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred while processing the question."
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all domain-exception -> HTTP response mappings.

    Call once from main.py, right after creating the FastAPI app instance.
    Registration order doesn't affect matching (Starlette resolves by
    walking the raised exception's MRO, not by insertion order), but
    specific-to-general reads better for anyone skimming this function.
    """

    app.add_exception_handler(EmptyQuestionError, empty_question_handler)
    app.add_exception_handler(LLMRateLimitError, llm_rate_limit_handler)
    app.add_exception_handler(LLMUnavailableError, llm_unavailable_handler)
    app.add_exception_handler(LLMResponseError, llm_response_handler)
    # Fallback last, both for readability and because it's the least specific.
    app.add_exception_handler(ResearchError, research_error_fallback_handler)
