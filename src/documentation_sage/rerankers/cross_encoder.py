from sentence_transformers import CrossEncoder

from documentation_sage.schemas.retrieval import RetrievedChunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:

        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]

        scores = self.model.predict(pairs)

        scored_chunks = list(zip(chunks, scores))

        scored_chunks.sort(
            key=lambda item: float(item[1]),
            reverse=True,
        )

        reranked_chunks: list[RetrievedChunk] = []

        for rank, (chunk, score) in enumerate(
            scored_chunks[:top_k],
            start=1,
        ):

            chunk.score = float(score)
            chunk.rank = rank

            reranked_chunks.append(chunk)

        return reranked_chunks
