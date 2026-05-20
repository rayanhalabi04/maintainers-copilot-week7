from app.domain.rag import RetrievedChunk


def merge_hybrid_scores(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    alpha: float,
    limit: int,
) -> list[RetrievedChunk]:
    by_id: dict[str, RetrievedChunk] = {}
    for chunk in dense:
        by_id[chunk.chunk_id] = chunk.model_copy(update={"dense_score": chunk.score})
    for chunk in sparse:
        existing = by_id.get(chunk.chunk_id)
        if existing is None:
            by_id[chunk.chunk_id] = chunk.model_copy(update={"sparse_score": chunk.score})
        else:
            by_id[chunk.chunk_id] = existing.model_copy(update={"sparse_score": chunk.score})

    merged = []
    for chunk in by_id.values():
        final_score = alpha * chunk.dense_score + (1 - alpha) * chunk.sparse_score
        merged.append(chunk.model_copy(update={"score": final_score}))
    return sorted(merged, key=lambda item: item.score, reverse=True)[:limit]
