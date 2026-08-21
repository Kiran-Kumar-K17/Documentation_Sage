from documentation_sage.generation.context import build_context
from documentation_sage.generation.groq_generator import GroqGenerator
from documentation_sage.rerankers.cross_encoder import CrossEncoderReranker
from documentation_sage.retrievers.hybrid import HybridRetriever


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        generator: GroqGenerator,
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
        )

        # 3. Build context
        context = build_context(
            chunks=reranked_chunks,
        )

        # 4. Generate answer
        answer = self.generator.generate(
            query=query,
            context=context,
        )

        return answer
