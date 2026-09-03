from pydantic import BaseModel, ConfigDict, HttpUrl


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: HttpUrl
    title: str
    content: str
    # Cosine similarity score (0-1) between this chunk's embedding and the
    # search query's embedding (see embeddings/ranking.py). Optional and
    # defaulting to none keeps this backward-compatible with any Answer
    # built without going through the RAG retrieval pipeline (e.g. the
    # model answering directly, no tool call involved).
    relevance_score: float | None = None
