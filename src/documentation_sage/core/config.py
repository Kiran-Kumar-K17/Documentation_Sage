import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32


class VectorStoreConfig(BaseModel):
    persist_directory: str = "data/vector_store"
    collection_name: str = "Text_Documents"


class GenerationConfig(BaseModel):
    model_name: str = os.getenv("LLM_MODEL", "")
    api_key: str = os.getenv("GROQ_API_KEY", "")
    temperature: float = 0.2
    max_tokens: int = 1000


# ADD THIS
class RerankerConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class AppConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()

    embedding: EmbeddingConfig = EmbeddingConfig()

    vector_store: VectorStoreConfig = VectorStoreConfig()

    generation: GenerationConfig = GenerationConfig()

    reranker: RerankerConfig = RerankerConfig()
