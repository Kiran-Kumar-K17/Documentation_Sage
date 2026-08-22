import time

from documentation_sage.generation.context import build_context
from documentation_sage.generation.base import BaseGenerator
from documentation_sage.rerankers.cross_encoder import CrossEncoderReranker
from documentation_sage.retrievers.hybrid import HybridRetriever


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: BaseGenerator,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def query(
        self,
        query: str,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 3,
    ) -> tuple[str, dict[str, float]]:

        total_start = time.perf_counter()

        # 1. Retrieve documents
        retrieval_start = time.perf_counter()

        retrieved_chunks = self.retriever.retrieve(
            query=query,
            top_k=retrieval_top_k,
        )

        retrieval_time = time.perf_counter() - retrieval_start

        # 2. Rerank documents
        rerank_start = time.perf_counter()

        reranked_chunks = self.reranker.rerank(
            query=query,
            chunks=retrieved_chunks,
            top_k=rerank_top_k,
        )

        rerank_time = time.perf_counter() - rerank_start

        # 3. Build context
        context_start = time.perf_counter()

        context = build_context(
            chunks=reranked_chunks,
            max_context_chars=8000,
            max_chunk_chars=2000,
        )

        context_time = time.perf_counter() - context_start

        # 4. Generate answer
        generation_start = time.perf_counter()

        answer = self.generator.generate(
            query=query,
            context=context,
        )

        generation_time = time.perf_counter() - generation_start

        # 5. Extract unique sources
        sources = []

        for chunk in reranked_chunks:
            source_file = chunk.metadata.get("source_file")

            if source_file and source_file not in sources:
                sources.append(source_file)

        sources = sources[:3]

        # 6. Add sources
        if sources:
            answer += "\n\nSources:\n"

            for source in sources:
                answer += f"- {source}\n"

        total_time = time.perf_counter() - total_start

        metrics = {
            "retrieval_time": retrieval_time,
            "rerank_time": rerank_time,
            "context_time": context_time,
            "generation_time": generation_time,
            "total_time": total_time,
        }

        return answer.strip(), metrics
