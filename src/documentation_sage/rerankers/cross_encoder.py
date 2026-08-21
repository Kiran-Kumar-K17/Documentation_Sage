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
        score_threshold: float = 0.0,
    ) -> list[RetrievedChunk]:

        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]

        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        ranked_chunks = sorted(
            chunks,
            key=lambda chunk: chunk.score,
            reverse=True,
        )

        filtered_chunks = [
            chunk for chunk in ranked_chunks if chunk.score >= score_threshold
        ]

        return filtered_chunks[:top_k]
