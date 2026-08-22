from pathlib import Path

from documentation_sage.schemas.documents import Chunk
from documentation_sage.retrievers.bm25 import BM25Retriever


class BM25IndexManager:
    def __init__(
        self,
        index_path: str = "data/indexes/bm25.pkl",
    ) -> None:
        self.index_path = Path(index_path)

    def exists(self) -> bool:
        return self.index_path.exists()

    def create_or_load(
        self,
        chunks: list[Chunk] | None = None,
    ) -> BM25Retriever:

        # Load existing BM25 index
        if self.exists():
            print("Loading saved BM25 index...")

            return BM25Retriever.load(str(self.index_path))

        # Create a new BM25 index
        if chunks is None:
            raise ValueError("Chunks are required to create a new BM25 index.")

        print("Creating BM25 index...")

        bm25_retriever = BM25Retriever(
            chunks=chunks,
        )

        print("Saving BM25 index...")

        bm25_retriever.save(str(self.index_path))

        return bm25_retriever
