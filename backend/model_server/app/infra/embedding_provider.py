import hashlib
import math
import os


class EmbeddingProvider:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if os.getenv("RAG_EMBEDDING_PROVIDER") == "hashing" or self.model_name == "hashing":
            self._model = HashingEmbeddingProvider()
            return self._model
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = HashingEmbeddingProvider()
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        if isinstance(model, HashingEmbeddingProvider):
            return model.embed(texts)
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vector)) for vector in vectors]


class HashingEmbeddingProvider(EmbeddingProvider):
    """Small deterministic fallback for local tests when sentence-transformers is absent."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
