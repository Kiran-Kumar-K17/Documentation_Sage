from pathlib import Path
import pickle

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

    def save(
        self,
        chunks: list[Chunk],
    ) -> None:
        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(self.index_path, "wb") as file:
            pickle.dump(
                chunks,
                file,
            )

    def load(self) -> list[Chunk]:
        with open(self.index_path, "rb") as file:
            chunks = pickle.load(file)

        return chunks

    def create_or_load(
        self,
        chunks: list[Chunk] | None = None,
    ) -> BM25Retriever:

        if self.exists():
            print("Loading BM25 index...")

            saved_chunks = self.load()

            return BM25Retriever(
                chunks=saved_chunks,
            )

        if chunks is None:
            raise ValueError("Chunks are required to create a new BM25 index.")

        print("Creating BM25 index...")

        self.save(chunks)

        return BM25Retriever(
            chunks=chunks,
        )
