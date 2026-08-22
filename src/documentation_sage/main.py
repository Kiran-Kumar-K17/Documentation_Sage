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

from documentation_sage.generation.ollama_generator import (
    OllamaGenerator,
)

from documentation_sage.pipelines.rag import RAGPipeline


def create_rag_pipeline() -> RAGPipeline:
    """
    Initialize and return the complete RAG pipeline.
    """

    config = AppConfig()

    # ==================================================
    # BM25 INDEX MANAGER
    # ==================================================

    index_manager = BM25IndexManager()

    # ==================================================
    # VECTOR STORE
    # ==================================================

    print("Connecting to vector store...")

    vector_store = ChromaVectorStore(config.vector_store)

    # ==================================================
    # CHECK EXISTING DATA
    # ==================================================

    bm25_exists = index_manager.exists()
    vector_exists = vector_store.exists()

    print(f"BM25 index exists: {bm25_exists}")
    print(f"Vector database exists: {vector_exists}")

    # ==================================================
    # LOAD DOCUMENTS AND CREATE CHUNKS
    #
    # Only needed if BM25 or Vector DB is missing
    # ==================================================

    chunks = None

    if not bm25_exists or not vector_exists:

        print("\nLoading documents...")

        loader = TextDocumentLoader()

        documents = loader.load_directory(Path("data/python"))

        print(f"Loaded documents: {len(documents)}")

        # ----------------------------------------------

        print("Creating chunks...")

        chunker = RecursiveChunker(config.chunking)

        chunks = chunker.split(documents)

        print(f"Created chunks: {len(chunks)}")

    # ==================================================
    # BM25 RETRIEVER
    # ==================================================

    if bm25_exists:

        print("\nLoading saved BM25 index...")

        bm25_retriever = index_manager.create_or_load()

    else:

        print("\nInitializing BM25 retriever...")

        if chunks is None:
            raise RuntimeError("Chunks are required to create the BM25 index.")

        bm25_retriever = BM25Retriever(
            chunks=chunks,
        )

        print("Saving BM25 index...")

        bm25_retriever.save(str(index_manager.index_path))

        print("BM25 index created successfully.")

    # ==================================================
    # EMBEDDING MODEL
    # ==================================================

    print("\nInitializing embedding model...")

    embedding_model = SentenceTransformerEmbedder(config.embedding)

    # ==================================================
    # CREATE VECTOR DATABASE
    # ==================================================

    if vector_exists:

        print("\nVector database already exists.")

        print("Skipping embedding generation.")

    else:

        if chunks is None:
            raise RuntimeError("Chunks are required to create embeddings.")

        print(f"\nGenerating embeddings for " f"{len(chunks)} chunks...")

        texts = [chunk.content for chunk in chunks]

        embeddings = embedding_model.embed_documents(texts)

        print(f"\nGenerated {len(embeddings)} embeddings.")

        # ----------------------------------------------

        print("\nAdding embeddings to vector database...")

        vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
        )

        print("Vector database created successfully.")

    # ==================================================
    # VECTOR RETRIEVER
    # ==================================================

    print("\nInitializing retrievers...")

    vector_retriever = VectorRetriever(
        vector_store=vector_store,
        embedder=embedding_model,
    )

    # ==================================================
    # HYBRID RETRIEVER
    # ==================================================

    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    # ==================================================
    # RERANKER
    # ==================================================

    print("Initializing reranker...")

    reranker = CrossEncoderReranker(config.reranker.model_name)

    # ==================================================
    # GENERATOR
    # ==================================================

    print("Initializing generator...")

    generator = OllamaGenerator(
        model="phi4-mini:3.8b",
        temperature=0.2,
    )

    # ==================================================
    # RAG PIPELINE
    # ==================================================

    print("Building RAG pipeline...")

    rag = RAGPipeline(
        retriever=hybrid_retriever,
        reranker=reranker,
        generator=generator,
    )

    return rag


def main() -> None:

    try:

        rag = create_rag_pipeline()

    except Exception as error:

        print("\nFailed to initialize Documentation Sage.")

        print(f"Error: {error}")

        return

    # ==================================================
    # CLI
    # ==================================================

    print("\n" + "=" * 60)

    print("Documentation Sage")
    print("Type 'q' to quit.")

    print("=" * 60)

    while True:

        try:

            query = input("\nQuestion: ").strip()

            # ------------------------------------------

            if query.lower() == "q":

                print("\nGoodbye!")

                break

            # ------------------------------------------

            if not query:

                continue

            # ------------------------------------------

            print("\nSearching documentation...\n")

            answer, metrics = rag.query(
                query=query,
                retrieval_top_k=30,
                rerank_top_k=5,
            )

            # ------------------------------------------
            # ANSWER
            # ------------------------------------------

            print("=" * 60)

            print("ANSWER")

            print("=" * 60)

            print(answer)

            # ------------------------------------------
            # PERFORMANCE
            # ------------------------------------------

            print("\n" + "=" * 60)

            print("PERFORMANCE")

            print("=" * 60)

            print(f"Retrieval:  " f"{metrics['retrieval_time']:.3f}s")

            print(f"Reranking:  " f"{metrics['rerank_time']:.3f}s")

            print(f"Context:    " f"{metrics['context_time']:.3f}s")

            print(f"Generation: " f"{metrics['generation_time']:.3f}s")

            print(f"Total:      " f"{metrics['total_time']:.3f}s")

        except KeyboardInterrupt:

            print("\n\nGoodbye!")

            break

        except Exception as error:

            print(f"\nUnexpected error: {error}")


if __name__ == "__main__":
    main()
