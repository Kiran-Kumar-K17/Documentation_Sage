from typing import Any
from pydantic import BaseModel, Field

class Document(BaseModel):
     """
    Represents a complete source document before chunking.
    """
     document_id: str
     content: str
     metadata: dict[str,Any] = Field(default_factory=dict)

class Chunk(BaseModel):
     """
    Represents a chunk of a source document.
    """
     chunk_id: str
     document_id: str
     content: str
     chunk_index: int
     metadata: dict[str,Any] = Field(default_factory=dict)