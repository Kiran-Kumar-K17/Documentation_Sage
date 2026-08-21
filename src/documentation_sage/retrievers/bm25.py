import re
from rank_bm25 import BM25Okapi
from documentation_sage.retrievers.base import Retriever
from documentation_sage.schemas.documents import Chunk
from documentation_sage.schemas.retrieval import RetrievedChunk


def tokenize(text: str) -> list[str]:
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


class BM25Retriever(Retriever):

    def __init__(
        self,
        chunks: list[Chunk],
    ):
        self.chunks = chunks

        tokenized_corpus = [chunk.content.lower().split() for chunk in chunks]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievedChunk]:

        tokenized_query = query.lower().split()

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
