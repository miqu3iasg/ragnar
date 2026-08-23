# This layer's only job is exposing the /ask endpoint and delegating to
# the domain service. It used to also translate domain exceptions into
# HTTP status codes via a try/except block (see git history), but that
# responsibility now lives entirely in api/exception_handlers.py,
# registered once in main.py. Every ResearchError subclass raised by
# ask_research_question propagates uncaught from here and is picked up
# by Starlette's exception middleware.
#
# Ref: https://fastapi.tiangolo.com/tutorial/handling-errors/#install-custom-exception-handlers
from app.domain.research.answer import Answer
from app.domain.research.question import Question
from app.domain.research.service import ask_research_question
from fastapi import APIRouter

router = APIRouter()


# Ref: https://fastapi.tiangolo.com/tutorial/response-model/#response-model-parameter
# response_model = specifies the format of the endpoint response
@router.post("/ask", response_model=Answer)
async def ask_question(question: Question) -> Answer:
    return await ask_research_question(question)
