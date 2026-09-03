from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

# Source used to be redefined here as a second, drifted copy of the class
# in source.py (missing relevance_score). That meant service.py built
# Source instances from source.py's class, but Answer's own `sources`
# field validated against this file's separate class of the same name —
# two distinct class objects, so Pydantic rejected every real Source
# instance as "not an instance of Source". Importing the canonical class
# instead of duplicating it is what fixes that.
# ref: https://pydantic.dev/docs/validation/dev/api/pydantic/config/
from app.domain.research.source import Source


# ref: https://pydantic.dev/docs/validation/latest/get-started/
# Pydantic automatically handles initialization based on the fields you declare
class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    question_id: UUID
    sources: list[Source] = Field(default_factory=list)
    answer_text: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
