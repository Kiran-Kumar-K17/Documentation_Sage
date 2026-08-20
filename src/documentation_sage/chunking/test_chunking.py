from pathlib import Path

from documentation_sage.core.config import AppConfig

# from documentation_sage.embeddings.sentence_transformer import (
#     SentenceTransformerEmbedder,
# )
from documentation_sage.ingestion.loader import TextDocumentLoader
from documentation_sage.chunking.recursive import RecursiveChunker

# 1. Load configuration
config = AppConfig()


# 2. Create loader
loader = TextDocumentLoader()


# 3. Load documents
documents = loader.load_directory(Path("data/python"))

print(f"Loaded documents: {len(documents)}")


# 4. Create chunker
chunker = RecursiveChunker(config.chunking)


# 5. Split documents
chunks = chunker.split(documents)

print(f"Created chunks: {len(chunks)}")


# 6. Basic validation
if chunks:

    print("\nFirst chunk")
    print(f"Chunk ID: {chunks[0].chunk_id}")
    print(f"Document ID: {chunks[0].document_id}")
    print(f"Chunk Index: {chunks[0].chunk_index}")
    print(f"Source: {chunks[0].metadata.get('source_file')}")
    print(f"Content preview: {chunks[0].content[:200]}")


# 7. Check unique IDs
chunk_ids = [chunk.chunk_id for chunk in chunks]

print(f"\nUnique chunk IDs: {len(set(chunk_ids))}")
print(f"Total chunk IDs: {len(chunk_ids)}")


if len(set(chunk_ids)) == len(chunk_ids):
    print("All chunk IDs are unique ✅")
else:
    print("Duplicate chunk IDs found ❌")


# embedder = SentenceTransformerEmbedder(config.embedding)
# document_embeddings = embedder.embed_documents([chunk.content for chunk in chunks])
# print(f"Total chunks: {len(chunks)}")
# print(f"Total embeddings: {len(document_embeddings)}")
# print(f"Embedding dimension: {len(document_embeddings[0])}")
