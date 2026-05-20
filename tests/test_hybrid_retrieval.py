from app.domain.rag import RetrievedChunk
from app.services.hybrid_retrieval import merge_hybrid_scores


def chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_type="issue",
        source_id=chunk_id,
        title="Title",
        score=score,
        text="text",
    )


def test_hybrid_score_merging_uses_alpha():
    dense = [chunk("a", 1.0), chunk("b", 0.5)]
    sparse = [chunk("a", 0.0), chunk("b", 1.0)]

    merged = merge_hybrid_scores(dense, sparse, alpha=0.6, limit=2)

    assert [item.chunk_id for item in merged] == ["b", "a"]
    assert merged[0].score == 0.7
    assert merged[1].score == 0.6
