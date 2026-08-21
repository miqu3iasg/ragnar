from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ConfigDict, HttpUrl


class Source(BaseModel):
    # ref: https://pydantic.dev/docs/validation/dev/api/pydantic/config/
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    title: str
    content: str


# ref: https://pydantic.dev/docs/validation/latest/get-started/
# Pydantic automatically handles initialization based on the fields you declare
class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    question_id: UUID
    sources: list[Source] = Field(default_factory=list)
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
