from abc import ABC, abstractmethod

from documentation_sage.schemas.documents import Chunk
from documentation_sage.schemas.retrieval import RetrievedChunk


class VectorStore(ABC):

    @abstractmethod
    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """
        Store chunks and their corresponding embeddings.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Search for the most similar chunks.
        """
        pass
