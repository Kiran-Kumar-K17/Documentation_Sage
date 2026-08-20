from documentation_sage.core.config import AppConfig


config = AppConfig()

print("Chunking:")
print(config.chunking)

print("\nEmbedding:")
print(config.embedding)

print("\nVector Store:")
print(config.vector_store)