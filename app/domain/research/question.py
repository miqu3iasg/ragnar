# ref:
# https://fastapi.tiangolo.com/#alternative-api-docs
from datetime import datetime
from enum import StrEnum
from uuid import uuid, uuid4

from pydantic import BaseModel, Field


class QuestionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# We can declare request objects and validate them using standard python types thanks to pidantic (BaseModel)
# This works like zod
class Question(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    content: str
    status: QuestionStatus = QuestionStatus.PENDING
    create_at: datetime = Field(default_factory=datetime.now)
