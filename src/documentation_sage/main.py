from pathlib import Path

from documentation_sage.core.config import AppConfig

from documentation_sage.ingestion.loader import TextDocumentLoader
from documentation_sage.chunking.recursive import RecursiveChunker

from documentation_sage.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)
from documentation_sage.vectorstores.chroma import ChromaVectorStore
from documentation_sage.retrievers.vector import VectorRetriever
from documentation_sage.retrievers.bm25 import BM25Retriever
from documentation_sage.retrievers.hybrid import HybridRetriever

from documentation_sage.rerankers.cross_encoder import (
    CrossEncoderReranker,
)
from documentation_sage.generation.ollama_generator import OllamaGenerator
from documentation_sage.pipelines.rag import RAGPipeline
from documentation_sage.retrievers.index import BM25IndexManager


def create_rag_pipeline() -> RAGPipeline:
    """
    Initialize and return the complete RAG pipeline.
    """

    config = AppConfig()

    index_manager = BM25IndexManager()

    if index_manager.exists():

        print("Loading saved BM25 chunks...")

        chunks = index_manager.load()

        print(f"Loaded chunks: {len(chunks)}")

    else:

        print("Loading documents...")

        loader = TextDocumentLoader()

        documents = loader.load_directory(Path("data/python"))

        print(f"Loaded documents: {len(documents)}")

        print("Creating chunks...")

        chunker = RecursiveChunker(config.chunking)

        chunks = chunker.split(documents)

        print(f"Created chunks: {len(chunks)}")

        print("Saving BM25 chunks...")

        index_manager.save(chunks)

        print("Initializing embedding model...")

    embedding_model = SentenceTransformerEmbedder(config.embedding)

    print("Connecting to vector store...")

    vector_store = ChromaVectorStore(
        config.vector_store,
    )

    print("Initializing retrievers...")

    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        embedder=embedding_model,
    )

    bm25_retriever = BM25Retriever(
        chunks=chunks,
    )

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    print("Initializing reranker...")

    reranker = CrossEncoderReranker(config.reranker.model_name)

    print("Initializing generator...")

    generator = OllamaGenerator(
        model="phi4-mini:3.8b",
        temperature=0.2,
    )

    print("Building RAG pipeline...")

    rag = RAGPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
    )

    return rag


def main():

    rag = create_rag_pipeline()

    print("\n" + "=" * 60)
    print("Documentation Sage")
    print("Type 'q' to quit.")
    print("=" * 60)

    while True:

        query = input("\nQuestion: ").strip()

        if query.lower() in {
            "q",
        }:
            print("\nGoodbye!")
            break

        if not query:
            continue

        print("\nSearching documentation...\n")

        answer = rag.query(
            query=query,
            retrieval_top_k=10,
            rerank_top_k=3,
        )

        print("=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(answer)


if __name__ == "__main__":
    main()
