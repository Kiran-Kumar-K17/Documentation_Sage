from documentation_sage.core.config import config
from documentation_sage.ingestion.loader import TextDocumentLoader
from documentation_sage.chunking.recursive import RecursiveChunker
from documentation_sage.embeddings.sentence_transformer import (
    SentenceTransformerEmbedder,
)
from documentation_sage.vectorstores.chroma import ChromaVectorStore
from documentation_sage.retrievers.bm25 import BM25Retriever


def main():
    # 1. Load documents
    print("Loading documents...")

    documents = TextDocumentLoader(config.data_directory).load_directory()

    print(f"Loaded documents: {len(documents)}")

    # 2. Create chunks
    print("Creating chunks...")

    chunks = RecursiveChunker(documents)

    print(f"Created chunks: {len(chunks)}")

    # 3. Initialize embedding model
    print("Initializing embedding model...")

    embedding_model = SentenceTransformerEmbedder()

    # 4. Generate embeddings
    print("Generating embeddings...")

    texts = [chunk.content for chunk in chunks]

    embeddings = embedding_model.embed(texts)

    # 5. Connect to ChromaDB
    print("Connecting to vector store...")

    vector_store = ChromaVectorStore(config.vector_store)

    # 6. Add embeddings and chunks
    print("Saving embeddings to vector store...")

    vector_store.add(
        chunks=chunks,
        embeddings=embeddings,
    )

    # 7. Build BM25 index
    print("Building BM25 index...")

    bm25_retriever = BM25Retriever(
        chunks=chunks,
    )

    # 8. Save BM25 index
    bm25_path = "data/indexes/bm25.pkl"

    bm25_retriever.save(bm25_path)

    print(f"BM25 index saved to: {bm25_path}")

    print("\n" + "=" * 60)
    print("INDEXING COMPLETE!")
    print(f"Documents indexed: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
