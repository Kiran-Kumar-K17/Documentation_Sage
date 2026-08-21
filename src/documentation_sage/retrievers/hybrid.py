from documentation_sage.retrievers.vector import VectorRetriever
from documentation_sage.retrievers.bm25 import BM25Retriever
from documentation_sage.schemas.retrieval import RetrievedChunk


class HybridRetriever:

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:

        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        fused_scores: dict[str, float] = {}

        chunks_by_id: dict[str, RetrievedChunk] = {}

        # Process vector results
        for result in vector_results:

            chunk_id = result.chunk_id

            chunks_by_id[chunk_id] = result

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (
                self.rrf_k + result.rank
            )

        # Process BM25 results
        for result in bm25_results:

            chunk_id = result.chunk_id

            chunks_by_id[chunk_id] = result

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1 / (
                self.rrf_k + result.rank
            )

        # Sort chunks by fused score
        sorted_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )

        # Create final results
        final_results: list[RetrievedChunk] = []

        for rank, chunk_id in enumerate(
            sorted_chunk_ids[:top_k],
            start=1,
        ):

            original_chunk = chunks_by_id[chunk_id]

            final_chunk = RetrievedChunk(
                chunk_id=original_chunk.chunk_id,
                document_id=original_chunk.document_id,
                content=original_chunk.content,
                metadata=original_chunk.metadata,
                score=fused_scores[chunk_id],
                rank=rank,
            )

            final_results.append(final_chunk)

        return final_results
