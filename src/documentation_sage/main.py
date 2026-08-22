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
from documentation_sage.retrievers.index import BM25IndexManager

from documentation_sage.rerankers.cross_encoder import (
    CrossEncoderReranker,
)

from documentation_sage.generation.ollama_generator import OllamaGenerator
from documentation_sage.pipelines.rag import RAGPipeline


def create_rag_pipeline() -> RAGPipeline:
    """
    Initialize and return the complete RAG pipeline.
    """

    config = AppConfig()

    # --------------------------------------------------
    # Load or create BM25 index
    # --------------------------------------------------

    index_manager = BM25IndexManager()

    if index_manager.exists():

        bm25_retriever = index_manager.create_or_load()

    else:
        print("Loading documents...")

        loader = TextDocumentLoader()

        documents = loader.load_directory(Path("data/python"))

        print(f"Loaded documents: {len(documents)}")

        print("Creating chunks...")

        chunker = RecursiveChunker(config.chunking)

        chunks = chunker.split(documents)

        print(f"Created chunks: {len(chunks)}")

        print("Saving BM25 index...")

        bm25_retriever.save(str(index_manager.index_path))

        print("Initializing BM25 retriever...")

        bm25_retriever = BM25Retriever(
            chunks=chunks,
        )

        print("Saving BM25 index...")

        bm25_retriever.save(str(index_manager.index_path))

        print("Initializing BM25 retriever...")

        bm25_retriever = BM25Retriever(
            chunks=chunks,
        )

    # --------------------------------------------------
    # Embedding model
    # --------------------------------------------------

    print("Initializing embedding model...")

    embedding_model = SentenceTransformerEmbedder(config.embedding)

    # --------------------------------------------------
    # Vector store
    # --------------------------------------------------

    print("Connecting to vector store...")

    vector_store = ChromaVectorStore(
        config.vector_store,
    )

    # --------------------------------------------------
    # Vector retriever
    # --------------------------------------------------

    print("Initializing retrievers...")

    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        embedder=embedding_model,
    )

    # --------------------------------------------------
    # Hybrid retriever
    # --------------------------------------------------

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    # --------------------------------------------------
    # Reranker
    # --------------------------------------------------

    print("Initializing reranker...")

    reranker = CrossEncoderReranker(config.reranker.model_name)

    # --------------------------------------------------
    # Generator
    # --------------------------------------------------

    print("Initializing generator...")

    generator = OllamaGenerator(
        model="phi4-mini:3.8b",
        temperature=0.2,
    )

    # --------------------------------------------------
    # RAG Pipeline
    # --------------------------------------------------

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

        if query.lower() == "q":
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

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print(answer)


if __name__ == "__main__":
    main()
