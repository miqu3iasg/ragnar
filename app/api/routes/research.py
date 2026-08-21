# This layer's only job is translating domain exceptions into HTTP status
# codes. It should never contain business logic because that lives in service.py.
#
# OBS: this try/except-per-route approach is intentional for now. The
# roadmap's next relevant item ("Criar um exception handler no FastAPI para
# erros de rate limit") will replace this with FastAPI's
# @app.exception_handler(...) mechanism, registered once in main.py, so new
# routes don't need to repeat this block.
#
# Ref: https://fastapi.tiangolo.com/tutorial/handling-errors/#install-custom-exception-handlers

import logging

from app.domain.research.answer import Answer
from app.domain.research.exceptions import (
    EmptyQuestionError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
)
from app.domain.research.question import Question
from app.domain.research.service import ask_research_question
from fastapi import APIRouter, HTTPException, status

logger = logging.getLogger(__name__)

router = APIRouter()


# Ref: https://fastapi.tiangolo.com/tutorial/response-model/#response-model-parameter
# reponse_model = specifies the format of the endpoint response
@router.post("/ask", response_model=Answer)
async def ask_question(question: Question) -> Answer:
    try:
        return await ask_research_question(question)
    except EmptyQuestionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except LLMRateLimitError as exc:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except LLMResponseError as exc:
        # Ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/502
        #
        # Provider responded but the content was unusable, then we raise 502 Bad Gateway.
        #
        # Distinct from 503. The upstream call itself succeeded, but what it
        # returned isn't something we can hand back to the client.
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
