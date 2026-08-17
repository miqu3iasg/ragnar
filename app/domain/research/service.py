from domain.research.question import Question
from domain.research.answer import Answer
from infrastructure.llm.client import get_completion


async def ask_research_question(question: Question) -> Answer:
    completion_content = get_completion(question.content)

    answer = Answer(question_id=question.id, sources=[], content=completion_content)

    return answer
