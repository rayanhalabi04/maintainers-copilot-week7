from fastapi.testclient import TestClient

from app.api.rag import get_rag_service
from app.domain.rag import RagQueryResponse, RagSource, RetrievalDebug
from app.main import app


class FakeRagService:
    def query(self, request):
        return RagQueryResponse(
            answer="Based on previous resolved issues, clear stale tokens.",
            sources=[
                RagSource(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    source_type="issue",
                    source_id="101",
                    title="JWTDecodeError during login",
                    url="https://example.test/issues/101",
                    score=0.87,
                    text="Clear stale tokens and align JWT secret.",
                )
            ],
            retrieval_debug=RetrievalDebug(
                original_query=request.question,
                rewritten_query=request.question,
                dense_top_k=20,
                sparse_top_k=20,
                hybrid_top_k=20,
                reranked_top_k=1,
                hybrid_alpha=0.6,
                reranker_enabled=False,
            ),
        )


def test_rag_query_endpoint_with_mocked_service():
    app.dependency_overrides[get_rag_service] = lambda: FakeRagService()
    try:
        response = TestClient(app).post(
            "/rag/query",
            json={"question": "How was the login token error fixed before?", "top_k": 5},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].startswith("Based on previous resolved issues")
    assert payload["sources"][0]["source_type"] == "issue"
