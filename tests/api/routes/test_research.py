# Refs:
# - FastAPI testing with TestClient: https://fastapi.tiangolo.com/tutorial/testing/
# - AsyncMock: https://docs.python.org/3/library/unittest.mock.html#unittest.mock.AsyncMock
# - monkeypatch: https://docs.pytest.org/en/stable/how-to/monkeypatch.html
# - where to patch (same principle used in tests/infrastructure/llm/test_client.py
# for infrastructure.llm.client.client): https://realpython.com/python-mock-library/#knowing-where-to-patch
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import app.api.routes.research as research_route
from app.domain.research.answer import Answer
from app.domain.research.exceptions import (
    EmptyQuestionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
)
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def question_payload() -> dict:
    return {"question_text": "O que é fotossíntese?"}


def test_ask_question_success(client, question_payload, monkeypatch):
    # We mock at the Answer level, not by hitting the real service, since
    # this test only cares whether the route wires the response correctly
    # end to end (status, body shape), not whether the LLM call itself
    # works, that's the service/client layer's job to prove.
    fake_answer = Answer(
        question_id="3b925b8e-2e42-4a5d-bcfe-a5194dac0109",  # It must be an id with a valid uuid format.
        answer_text=(
            "Photosynthesis is the process by which plants convert light into energy."
        ),
        sources=[],
    )
    mock_ask = AsyncMock(return_value=fake_answer)
    monkeypatch.setattr(research_route, "ask_research_question", mock_ask)

    response = client.post("/ask", json=question_payload)

    assert response.status_code == 200
    body = response.json()
    assert body["answer_text"] == fake_answer.answer_text
    assert body["sources"] == []
    # Confirms the route actually delegated to the service, not that it
    # returned 200 for some unrelated reason (e.g. a stale cached response).
    mock_ask.assert_awaited_once()


# Each of these is raised by the mocked service and must reach the client as
# the status code owned by api/exception_handlers.py. This is what actually
# proves the handler wiring registered in main.py works end to end, since
# test_exception_handlers.py only proves the handler functions are correct
# in isolation, not that they're reachable from a real request through /ask.
@pytest.mark.parametrize(
    "exc, expected_status",
    [
        (EmptyQuestionError("Question content must not be empty."), 400),
        (LLMRateLimitError("Provider rate limit exceeded."), 429),
        (LLMUnavailableError("Could not reach the LLM provider."), 503),
        (LLMResponseError("The model did not return a usable response."), 502),
    ],
)
def test_ask_question_domain_errors_map_to_expected_http_status(
    client, question_payload, monkeypatch, exc, expected_status
):
    mock_ask = AsyncMock(side_effect=exc)
    monkeypatch.setattr(research_route, "ask_research_question", mock_ask)

    response = client.post("/ask", json=question_payload)

    assert response.status_code == expected_status
    assert response.json() == {"detail": str(exc)}


def test_ask_question_missing_question_text_returns_422(client):
    # Ref: https://fastapi.tiangolo.com/tutorial/handling-errors/#override-the-default-exception-handlers
    #
    # This never reaches ask_research_question at all: request validation
    # happens first, via the Question Pydantic model, before our route body
    # even runs. Worth covering explicitly since it's the first line of
    # defense against malformed input and nothing else here exercises it.
    response = client.post("/ask", json={})

    assert response.status_code == 422


def test_ask_question_question_text_exceeding_max_length_returns_422(client):
    # Defense-in-depth: the max_length on Question.question_text (set in
    # app/domain/research/question.py) rejects overlong payloads at the
    # Pydantic layer, before any LLM or retrieval work happens. A request
    # at exactly max_length is accepted; one over it is rejected.
    overlong = "a" * 4001

    response = client.post("/ask", json={"question_text": overlong})

    assert response.status_code == 422


def test_ask_question_empty_string_question_text_returns_422(client):
    # An empty string fails Question's min_length=1 validator before
    # reaching the service layer's whitespace-stripping empty-question
    # check. Both paths reject, but the Pydantic layer rejects earlier
    # and with a more specific error class.
    response = client.post("/ask", json={"question_text": ""})

    assert response.status_code == 422


def test_ask_question_unexpected_exception_is_not_swallowed(
    client, question_payload, monkeypatch
):
    # Mirrors service.py's own "catch-all safety net" comment: a bug in our
    # code, or any exception type outside the ResearchError hierarchy, must
    # NOT get silently mapped to one of our domain status codes. It should
    # propagate as an unhandled error instead, same principle as
    # test_non_research_error_is_not_caught_by_our_handlers in
    # test_exception_handlers.py, but verified here through the real app and
    # the real route, not a throwaway one.
    mock_ask = AsyncMock(side_effect=RuntimeError("something we didn't anticipate"))
    monkeypatch.setattr(research_route, "ask_research_question", mock_ask)

    with pytest.raises(RuntimeError):
        client.post("/ask", json=question_payload)
