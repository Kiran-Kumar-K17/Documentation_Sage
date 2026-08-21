from pathlib import Path

from documentation_sage.core.config import AppConfig
from documentation_sage.ingestion.loader import TextDocumentLoader
from documentation_sage.chunking.recursive import RecursiveChunker
from documentation_sage.retrievers.bm25 import BM25Retriever

config = AppConfig()

loader = TextDocumentLoader()

documents = loader.load_directory(Path("data/python"))

chunker = RecursiveChunker(config.chunking)

chunks = chunker.split(documents)

print(f"Loaded chunks: {len(chunks)}")


retriever = BM25Retriever(chunks=chunks)


results = retriever.retrieve(
    query="How do I handle exceptions in Python?",
    top_k=5,
)


for result in results:

    print("\n" + "=" * 60)

    print("Rank:", result.rank)
    print("Source:", result.metadata.get("source_file"))
    print("BM25 Score:", result.score)

    print("\nContent:")
    print(result.content[:500])
