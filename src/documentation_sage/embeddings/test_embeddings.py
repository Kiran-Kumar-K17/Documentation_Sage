from documentation_sage.core.config import AppConfig

from documentation_sage.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)

config = AppConfig()
embedder = SentenceTransformerEmbedder(config.embedding)


texts = [
    "Python is a programming language.",
    "Lists in Python are mutable.",
    "Dictionaries store key-value pairs.",
]

document_embeddings = embedder.embed_documents(texts)

print(f"Number of document embeddings: {len(document_embeddings)}")
print(f"Embedding dimension: {len(document_embeddings[0])}")


query = "How do Python dictionaries work?"

query_embedding = embedder.embed_query(query)

print(f"\nQuery embedding dimension: {len(query_embedding)}")
print(f"First 5 values: {query_embedding[:5]}")
