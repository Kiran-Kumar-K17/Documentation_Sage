from documentation_sage.generation.context import build_context
from documentation_sage.generation.ollama_generator import OllamaGenerator
from documentation_sage.rerankers.cross_encoder import CrossEncoderReranker
from documentation_sage.retrievers.hybrid import HybridRetriever


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: OllamaGenerator,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def query(
        self,
        query: str,
        retrieval_top_k: int = 10,
        rerank_top_k: int = 5,
    ) -> str:

        # 1. Retrieve documents
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            top_k=retrieval_top_k,
        )

        # 2. Rerank documents
        reranked_chunks = self.reranker.rerank(
            query=query,
            chunks=retrieved_chunks,
            top_k=rerank_top_k,
            score_threshold=0.0,
        )
        if not reranked_chunks:
            return "I couldn't find the answer in the provided documentation."

        # 3. Collect unique source files

        # 4. Build context
        context, used_chunks = build_context(
            chunks=reranked_chunks,
            max_context_chars=8000,
            max_chunk_chars=2000,
        )
        sources = []

        for chunk in used_chunks:
            source_file = chunk.metadata.get("source_file")

            if source_file and source_file not in sources:
                sources.append(source_file)

        # 5. Generate answer
        answer = self.generator.generate(
            query=query,
            context=context,
        )

        # 6. Add citations
        if sources:
            answer += "\n\nSources:\n"

            for source in sources:
                answer += f"- {source}\n"

        return answer
