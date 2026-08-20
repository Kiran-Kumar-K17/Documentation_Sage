from pathlib import Path

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """
    Configuration for document chunking.
    """

    chunk_size: int = 1000
    chunk_overlap: int = 200


class EmbeddingConfig(BaseModel):
    """
    Configuration for embedding generation.
    """

    model_name: str = "all-MiniLM-L6-v2"

    batch_size: int = 32


class VectorStoreConfig(BaseModel):
    """
    Configuration for the vector database.
    """

    collection_name: str = "Text_Documents"

    persist_directory: Path = Path("data/vector_store")


class AppConfig(BaseModel):
    """
    Main application configuration.
    """

    chunking: ChunkingConfig = Field(
        default_factory=ChunkingConfig
    )

    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig
    )

    vector_store: VectorStoreConfig = Field(
        default_factory=VectorStoreConfig
    )