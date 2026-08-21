from documentation_sage.core.config import AppConfig
from documentation_sage.vectorstores.chroma import ChromaVectorStore
from documentation_sage.schemas.documents import Chunk
from documentation_sage.ingestion.loader import TextDocumentLoader
from documentation_sage.chunking.recursive import RecursiveChunker
from documentation_sage.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)
from pathlib import Path

config = AppConfig()
loader = TextDocumentLoader()
chunker = RecursiveChunker(config.chunking)
embedder = SentenceTransformerEmbedder(config.embedding)
vector_store = ChromaVectorStore(config.vector_store)


query = "How do I handle exceptions in Python?"

query_embedding = embedder.embed_query(query)

results = vector_store.search(
    query_embedding=query_embedding,
    top_k=5,
)

print(f"\nQuery: {query}")
print(f"Results: {len(results)}")

for result in results:
    print("Chunk ID:", result.chunk_id)
    print("Document ID:", result.document_id)
    print("Distance:", result.score)
    print("Source:", result.metadata.get("source_file"))
    print("Chunk index:", result.metadata.get("chunk_index"))
    print("\nContent:")
    print(result.content[:700])


# documents = loader.load_directory(Path("data/python"))

# chunks = chunker.split(documents)

# embeddings = embedder.embed_documents([chunk.content for chunk in chunks])

# vector_store.add(
#     chunks=chunks,
#     embeddings=embeddings,
# )

# query = "How does Python handle exceptions?"

# query_embedding = embedder.embed_query(query)

# results = vector_store.search(
#     query_embedding=query_embedding,
#     top_k=5,
# )
# # print("Vector store initialized")

# print("Collection:", vector_store.collection.name)

# print("Collection count:", vector_store.collection.count())


# vector_store.add(
#     chunks=chunks,
#     embeddings=embeddings,
# )

# print("Collection count:", vector_store.collection.count())
# stored_data = vector_store.collection.get(
#     include=[
#         "documents",
#         "metadatas",
#     ]
# )

# print("\nStored data:")
# print("IDs:", stored_data["ids"])
# print("Documents:", stored_data["documents"])
# print("Metadata:", stored_data["metadatas"])
# query_embedding = [0.1] * 384

# results = vector_store.search(
#     query_embedding=query_embedding,
#     top_k=3,
# )

# print(f"\nResults: {len(results)}")

# for result in results:
#     print("\n---")
#     print("Rank:", result.rank)
#     print("Chunk ID:", result.chunk_id)
#     print("Document ID:", result.document_id)
#     print("Score:", result.score)
#     print("Content:", result.content)
