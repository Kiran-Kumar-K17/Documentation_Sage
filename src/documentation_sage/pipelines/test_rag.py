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

from documentation_sage.rerankers.cross_encoder import (
    CrossEncoderReranker,
)

from documentation_sage.generation.groq_generator import (
    GroqGenerator,
)

from documentation_sage.pipelines.rag import (
    RAGPipeline,
)

config = AppConfig()


# --------------------------------------------------
# Load documents and create chunks
# --------------------------------------------------

loader = TextDocumentLoader()

documents = loader.load_directory(Path("data/python"))

print(f"Loaded documents: {len(documents)}")


chunker = RecursiveChunker(config.chunking)

chunks = chunker.split(documents)

print(f"Created chunks: {len(chunks)}")


# --------------------------------------------------
# Initialize embedding + vector store
# --------------------------------------------------

embedder = SentenceTransformerEmbedder(config.embedding)

vector_store = ChromaVectorStore(config.vector_store)


# --------------------------------------------------
# Create retrievers
# --------------------------------------------------

vector_retriever = VectorRetriever(
    embedder=embedder,
    vector_store=vector_store,
)


bm25_retriever = BM25Retriever(
    chunks=chunks,
)


hybrid_retriever = HybridRetriever(
    vector_retriever=vector_retriever,
    bm25_retriever=bm25_retriever,
)


# --------------------------------------------------
# Create reranker
# --------------------------------------------------

reranker = CrossEncoderReranker(config.reranker.model_name)


# --------------------------------------------------
# Create generator
# --------------------------------------------------

generator = GroqGenerator(
    api_key=config.generation.api_key,
    model=config.generation.model_name,
    temperature=config.generation.temperature,
)


# --------------------------------------------------
# Create RAG pipeline
# --------------------------------------------------

rag = RAGPipeline(
    retriever=hybrid_retriever,
    reranker=reranker,
    generator=generator,
)


# --------------------------------------------------
# Ask question
# --------------------------------------------------

query = "How do I handle exceptions in Python?"

answer = rag.query(
    query=query,
    retrieval_top_k=10,
    rerank_top_k=5,
)


print("\n" + "=" * 70)

print("QUESTION:")
print(query)

print("\nANSWER:")
print(answer)

print("\n" + "=" * 70)
