from __future__ import annotations

from functools import cached_property


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @cached_property
    def _model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for embedding generation."
            ) from exc
        return SentenceTransformer(self.model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
