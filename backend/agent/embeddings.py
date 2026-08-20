"""
Local embedding model used by the caller-memory system.

The model runs locally on CPU, so embedding does not require an API
call or an external service.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app import config


class EmbeddingModel:
    """Small wrapper around SentenceTransformers."""

    def __init__(self):
        self.model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=config.EMBEDDING_DEVICE,
        )

    def encode(self, text: str) -> list[float]:
        """Convert one text string into a normalized embedding vector."""

        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts."""

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""

        return self.model.get_sentence_embedding_dimension()