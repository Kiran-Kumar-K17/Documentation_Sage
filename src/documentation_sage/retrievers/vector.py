from documentation_sage.embeddings.base import EmbeddingProvider
from documentation_sage.retrievers.base import Retriever
from documentation_sage.schemas.retrieval import RetrievedChunk
from documentation_sage.vectorstores.base import VectorStore


class VectorRetriever(Retriever):

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:

        query_embedding = self.embedder.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results
