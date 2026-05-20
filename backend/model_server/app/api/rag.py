from fastapi import APIRouter, Depends

from app.domain.rag import RagQueryRequest, RagQueryResponse
from app.services.rag_service import RagService


router = APIRouter(tags=["rag"])

_rag_service = RagService()


def get_rag_service() -> RagService:
    return _rag_service


@router.post("/rag/query", response_model=RagQueryResponse)
def query_rag(
    request: RagQueryRequest,
    service: RagService = Depends(get_rag_service),
) -> RagQueryResponse:
    return service.query(request)
