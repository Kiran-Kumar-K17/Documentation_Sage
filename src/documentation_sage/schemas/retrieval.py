from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """
    Represents a chunk returned by a retriever.
    """

    chunk_id: str

    document_id: str | None = None

    content: str

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )

    score: float

    rank: int