import pickle
import re
from pathlib import Path
from nltk.stem.porter import PorterStemmer
from rank_bm25 import BM25Okapi
from documentation_sage.retrievers.base import Retriever
from documentation_sage.schemas.documents import Chunk
from documentation_sage.schemas.retrieval import RetrievedChunk

_stemmer = None


def get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = PorterStemmer()
    return _stemmer


def tokenize(text: str) -> list[str]:
    """Tokenize text with Porter stemming."""
    stemmer = get_stemmer()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [stemmer.stem(token) for token in tokens]


class BM25Retriever(Retriever):
    def __init__(
        self,
        chunks: list[Chunk],
    ):
        self.chunks = chunks

        tokenized_corpus = [tokenize(chunk.content) for chunk in chunks]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self, path: str) -> None:
        path = Path(path)

        # Create the directory if it doesn't exist
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open("wb") as file:
            pickle.dump(
                {
                    "chunks": self.chunks,
                    "bm25": self.bm25,
                },
                file,
            )

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        path = Path(path)

        with path.open("rb") as file:
            data = pickle.load(file)

        instance = cls.__new__(cls)

        instance.chunks = data["chunks"]
        instance.bm25 = data["bm25"]

        return instance

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:

        tokenized_query = tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results: list[RetrievedChunk] = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):
            chunk = self.chunks[index]

            result = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=float(scores[index]),
                rank=rank,
            )

            results.append(result)

        return results
