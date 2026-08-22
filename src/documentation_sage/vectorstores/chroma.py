import chromadb

from documentation_sage.schemas.documents import Chunk
from documentation_sage.core.config import VectorStoreConfig
from documentation_sage.vectorstores.base import VectorStore
from documentation_sage.schemas.retrieval import RetrievedChunk


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        config: VectorStoreConfig,
    ) -> None:
        self.config = config

        self.client = chromadb.PersistentClient(path=config.persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=config.collection_name
        )

    def exists(self) -> bool:
        """
        Check whether the vector store contains embeddings.
        """
        return self.collection.count() > 0

    def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        batch_size: int = 1000,
    ) -> None:

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings.")

        total_chunks = len(chunks)

        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)

            batch_chunks = chunks[start:end]
            batch_embeddings = embeddings[start:end]

            ids = [chunk.chunk_id for chunk in batch_chunks]

            documents = [chunk.content for chunk in batch_chunks]

            metadatas = [
                {
                    **chunk.metadata,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in batch_chunks
            ]

            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            print(f"Added chunks {start + 1}-{end} " f"of {total_chunks}")

    def search(
        self,
        query_embedding,
        top_k,
    ) -> list[RetrievedChunk]:

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        retrieved_chunks: list[RetrievedChunk] = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for rank, (
            chunk_id,
            document,
            metadata,
            distance,
        ) in enumerate(
            zip(
                ids,
                documents,
                metadatas,
                distances,
            ),
            start=1,
        ):

            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk_id,
                document_id=metadata.get("document_id"),
                content=document,
                metadata=metadata,
                score=float(distance),
                rank=rank,
            )

            retrieved_chunks.append(retrieved_chunk)

        return retrieved_chunks
