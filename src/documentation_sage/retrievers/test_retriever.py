from pathlib import Path

from documentation_sage.core.config import AppConfig

from documentation_sage.ingestion.loader import (
    TextDocumentLoader,
)

from documentation_sage.chunking.recursive import (
    RecursiveChunker,
)

from documentation_sage.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)

from documentation_sage.vectorstores.chroma import (
    ChromaVectorStore,
)

from documentation_sage.retrievers.vector import (
    VectorRetriever,
)

from documentation_sage.retrievers.bm25 import (
    BM25Retriever,
)

from documentation_sage.retrievers.hybrid import (
    HybridRetriever,
)

# -------------------------
# Configuration
# -------------------------

config = AppConfig()


# -------------------------
# Load documents
# -------------------------

loader = TextDocumentLoader()

documents = loader.load_directory(Path("data/python"))

print(f"Loaded documents: {len(documents)}")


# -------------------------
# Create chunks
# -------------------------

chunker = RecursiveChunker(config.chunking)

chunks = chunker.split(documents)

print(f"Created chunks: {len(chunks)}")


# -------------------------
# BM25 Retriever
# -------------------------

bm25_retriever = BM25Retriever(chunks=chunks)


# -------------------------
# Vector Retriever
# -------------------------

embedder = SentenceTransformerEmbedder(config.embedding)

vector_store = ChromaVectorStore(config.vector_store)

vector_retriever = VectorRetriever(
    embedder=embedder,
    vector_store=vector_store,
)


# -------------------------
# Hybrid Retriever
# -------------------------

hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
)


# -------------------------
# Query
# -------------------------

query = "How do I handle exceptions in Python?"

results = hybrid_retriever.retrieve(
    query=query,
    top_k=5,
)


# -------------------------
# Display results
# -------------------------

print("\n" + "=" * 70)

print("Query:")
print(query)

print("\nResults:")
print(len(results))


for result in results:

    print("\n" + "=" * 70)

    print("Rank:", result.rank)

    print(
        "Score:",
        result.score,
    )

    print(
        "Source:",
        result.metadata.get("source_file"),
    )

    print(
        "Chunk ID:",
        result.chunk_id,
    )

    print("\nContent:")

    print(result.content[:500])
