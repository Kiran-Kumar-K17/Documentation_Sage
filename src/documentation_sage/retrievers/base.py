from abc import ABC, abstractmethod

from documentation_sage.schemas.retrieval import RetrievedChunk


class Retriever(ABC):

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        pass
