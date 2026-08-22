from documentation_sage.retrievers.vector import VectorRetriever
from documentation_sage.retrievers.bm25 import BM25Retriever
from documentation_sage.schemas.retrieval import RetrievedChunk


class HybridRetriever:
    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        candidate_multiplier: int = 10,
    ) -> None:
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:

        # Retrieve more candidates from each retriever
        candidate_k = top_k * self.candidate_multiplier

        print(
            f"\nHybrid retrieval:"
            f"\n  Final top_k: {top_k}"
            f"\n  Candidates per retriever: {candidate_k}"
        )

        # Vector search
        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        # BM25 search
        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_k,
        )

        fused_scores: dict[str, float] = {}
        chunks_by_id: dict[str, RetrievedChunk] = {}

        # ------------------------------------------
        # Process vector results
        # ------------------------------------------

        for result in vector_results:
            chunk_id = result.chunk_id

            chunks_by_id[chunk_id] = result

            rrf_score = 1 / (self.rrf_k + result.rank)

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + rrf_score

        # ------------------------------------------
        # Process BM25 results
        # ------------------------------------------

        for result in bm25_results:
            chunk_id = result.chunk_id

            # Keep the chunk if it is new.
            # If it already exists, metadata/content
            # should be identical because it is the same chunk.
            chunks_by_id[chunk_id] = result

            rrf_score = 1 / (self.rrf_k + result.rank)

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + rrf_score

        # ------------------------------------------
        # Sort by RRF score
        # ------------------------------------------

        sorted_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )

        # ------------------------------------------
        # Build final results
        # ------------------------------------------

        final_results: list[RetrievedChunk] = []

        for rank, chunk_id in enumerate(
            sorted_chunk_ids[:top_k],
            start=1,
        ):
            chunk = chunks_by_id[chunk_id]

            final_results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=fused_scores[chunk_id],
                    rank=rank,
                )
            )

        return final_results
