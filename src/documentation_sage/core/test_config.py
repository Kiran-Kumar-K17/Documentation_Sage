from documentation_sage.core.config import AppConfig
from documentation_sage.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)

config = AppConfig()

print("Chunking:")
print(config.chunking)

print("\nEmbedding:")
print(config.embedding)

print("\nVector Store:")
print(config.vector_store)

print("Embedding model:", config.embedding.model_name)
print("Batch size:", config.embedding.batch_size)

embedder = SentenceTransformerEmbedder(config.embedding)
