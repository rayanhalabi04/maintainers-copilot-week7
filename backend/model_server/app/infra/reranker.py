from app.domain.rag import RetrievedChunk


class Reranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        raise NotImplementedError


class CrossEncoderReranker(Reranker):
    def __init__(self, model_name: str, enabled: bool = True) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._model = None
        self.last_used = False

    def _load_model(self):
        if not self.enabled:
            return None
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception:
                self.enabled = False
                self._model = None
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        model = self._load_model()
        if model is None or not chunks:
            self.last_used = False
            return chunks
        self.last_used = True
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = model.predict(pairs)
        reranked = []
        for chunk, score in zip(chunks, scores):
            updated = chunk.model_copy(update={"score": float(score)})
            reranked.append(updated)
        return sorted(reranked, key=lambda item: item.score, reverse=True)
