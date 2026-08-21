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
from documentation_sage.rerankers.cross_encoder import CrossEncoderReranker

config = AppConfig()

# Load documents
loader = TextDocumentLoader()

documents = loader.load_directory(Path("data/python"))

print(f"Loaded documents: {len(documents)}")


# Create chunks
chunker = RecursiveChunker(config.chunking)

chunks = chunker.split(documents)

print(f"Created chunks: {len(chunks)}")


# Embedding model
embedder = SentenceTransformerEmbedder(config.embedding)


# Vector store
vector_store = ChromaVectorStore(config.vector_store)


# Vector retriever
vector_retriever = VectorRetriever(
    embedder=embedder,
    vector_store=vector_store,
)


# BM25 retriever
bm25_retriever = BM25Retriever(chunks=chunks)


# Hybrid retriever
hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
)


# Reranker
reranker = CrossEncoderReranker()


query = "How do I handle exceptions in Python?"


# Get more candidates first
retrieved_chunks = hybrid_retriever.retrieve(
    query=query,
    top_k=20,
)

print("\n" + "=" * 70)
print("BEFORE RERANKING")
print("=" * 70)

for chunk in retrieved_chunks[:10]:

    print(f"\nRank: {chunk.rank}")

    print(
        "Source:",
        chunk.metadata.get("source_file"),
    )

    print(
        "Score:",
        chunk.score,
    )


# Rerank
results = reranker.rerank(
    query=query,
    chunks=retrieved_chunks,
    top_k=5,
)


print("\n" + "=" * 70)
print("AFTER RERANKING")
print("=" * 70)


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

    print("\nContent:")

    print(result.content[:700])
