from documentation_sage.core.config import AppConfig
from documentation_sage.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)
from documentation_sage.retrievers.vector import VectorRetriever
from documentation_sage.vectorstores.chroma import ChromaVectorStore

config = AppConfig()

embedder = SentenceTransformerEmbedder(config.embedding)

vector_store = ChromaVectorStore(config.vector_store)

retriever = VectorRetriever(
    embedder=embedder,
    vector_store=vector_store,
)

results = retriever.retrieve(
    query="How do I handle exceptions in Python?",
    top_k=5,
)

for result in results:
    print("\n" + "=" * 60)
    print("Rank:", result.rank)
    print("Source:", result.metadata.get("source_file"))
    print("Distance:", result.score)
    print("Content:", result.content[:300])
