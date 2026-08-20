from sentence_transformers import SentenceTransformer
from documentation_sage.embeddings.base import EmbeddingProvider
from documentation_sage.core.config import EmbeddingConfig


class SentenceTransformerEmbedder(EmbeddingProvider):

    def __init__(
        self,
        config: EmbeddingConfig,
    ) -> None:

        self.config = config

        self.model = SentenceTransformer(config.model_name)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        embedding = self.model.encode(
            query,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
        )

        return embedding.tolist()
