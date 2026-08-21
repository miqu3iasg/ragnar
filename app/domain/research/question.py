# ref:
# https://fastapi.tiangolo.com/#alternative-api-docs
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class QuestionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# We can declare request objects and validate them using standard python types thanks to pidantic (BaseModel)
# This works like zod
class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    question_text: str
    status: QuestionStatus = QuestionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
