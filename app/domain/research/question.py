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
    # Length caps exist as a defense-in-depth input-validation layer:
    # min_length blocks the validator accepting a totally empty payload
    # that downstream code might forget to re-check, max_length caps the
    # worst-case prompt size sent to the LLM (and the size of a single
    # chunk that flows through the retrieval pipeline) at request time.
    # The exact values were picked to be generous enough for real research
    # questions while still blocking obvious abuse (multi-megabyte
    # paste-ins); see TESTING.md for the rationale.
    question_text: str = Field(min_length=1, max_length=4000)
    status: QuestionStatus = QuestionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
