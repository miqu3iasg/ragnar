# Refs:
# - FastAPI custom exception handlers: https://fastapi.tiangolo.com/tutorial/handling-errors/#install-custom-exception-handlers
# - Starlette exception handling (matching is done by walking the raised
# exception's MRO, not by insertion order): https://www.starlette.io/exceptions/
# - pytest parametrize: https://docs.pytest.org/en/stable/how-to/parametrize.html
#
# These tests exercise app/api/exception_handlers.py in isolation, with no
# dependency on api/routes/research.py or the domain service. Each test wires
# a throwaway FastAPI app with a single route that raises the exception under
# test, so a failure here points directly at the handler mapping, not at
# routing or mocking that lives elsewhere.

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.exception_handlers import register_exception_handlers
from app.domain.research.exceptions import (
    EmptyQuestionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
    ResearchError,
)


def _make_client(exc_to_raise: Exception) -> TestClient:
    # Disposable app, built fresh per test so raising a different exception
    # never leaks state (e.g. handler registration order) between test cases.
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    async def boom():
        raise exc_to_raise

    return TestClient(app)


# Each of these exceptions has its own handler registered in
# register_exception_handlers, and each handler owns the status code +
# message pairing documented on the exception class itself (see
# domain/research/exceptions.py). This parametrize just confirms the wiring
# actually produces the response the docstrings promise.
@pytest.mark.parametrize(
    "exc, expected_status",
    [
        (EmptyQuestionError("Question content must not be empty."), 400),
        (LLMRateLimitError("Provider rate limit exceeded."), 429),
        (LLMUnavailableError("Could not reach the LLM provider."), 503),
        (LLMResponseError("The model did not return a usable response."), 502),
    ],
)
def test_specific_handlers_return_expected_status_and_detail(exc, expected_status):
    client = _make_client(exc)

    response = client.get("/boom")

    assert response.status_code == expected_status
    assert response.json() == {"detail": str(exc)}


def test_llm_rate_limit_handler_forwards_retry_after_header():
    # When the underlying exception carries a provider-suggested wait
    # time, the 429 response must surface it as a Retry-After HTTP
    # header so the caller can pace their own retries. Without this,
    # the provider's hint was thrown away at the boundary.
    exc = LLMRateLimitError("Provider rate limit exceeded.", retry_after=12.5)

    client = _make_client(exc)

    response = client.get("/boom")

    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "12"


def test_llm_rate_limit_handler_omits_retry_after_when_not_provided():
    # If the exception doesn't carry a wait time, no header is set.
    # Asserting absence (rather than None/empty) is the contract — a
    # spurious Retry-After would mislead callers into backing off
    # unnecessarily.
    exc = LLMRateLimitError("Provider rate limit exceeded.")

    client = _make_client(exc)

    response = client.get("/boom")

    assert response.status_code == 429
    assert "Retry-After" not in response.headers


def test_research_error_fallback_handles_unregistered_subclass():
    # Simulates exactly what ResearchError's own docstring warns about: a
    # future subclass gets added to exceptions.py but nobody registers a
    # specific handler for it. Starlette resolves this by walking the raised
    # exception's class hierarchy (SomeFutureResearchError -> ResearchError)
    # until it finds a registered handler, which is why the fallback still
    # fires here instead of leaking as an undetailed, unhandled 500.
    # Ref: https://www.starlette.io/exceptions/
    class SomeFutureResearchError(ResearchError):
        pass

    client = _make_client(SomeFutureResearchError("unexpected domain failure"))

    response = client.get("/boom")

    assert response.status_code == 500
    # Deliberately not asserting the raw exception message here: the
    # fallback handler intentionally returns a generic message instead of
    # str(exc), since by definition we don't know what kind of failure a
    # future, unregistered subclass represents.
    assert response.json() == {
        "detail": "An unexpected error occurred while processing the question."
    }


def test_non_research_error_is_not_caught_by_our_handlers():
    # Ref: https://www.starlette.io/exceptions/#errors-and-handled-exceptions
    #
    # Guards against handlers being registered too broadly. If
    # register_exception_handlers ever accidentally caught something on the
    # base Exception class, it would start swallowing unrelated bugs in our
    # own code as generic domain responses, exactly what ResearchError's
    # docstring says this fallback should NOT do. A plain ValueError should
    # propagate unhandled.
    #
    # TestClient re-raises unhandled server exceptions by default (instead
    # of turning them into a 500 response), which is what lets us assert on
    # pytest.raises here instead of a response status code.
    client = _make_client(ValueError("this is not a ResearchError"))

    with pytest.raises(ValueError):
        client.get("/boom")
