from fastapi import APIRouter, Depends

from app.domain.summarization import SummarizeRequest, SummarizeResponse
from app.services.summarization_service import SummarizationService


router = APIRouter(tags=["summarization"])

_summarization_service = SummarizationService()


def get_summarization_service() -> SummarizationService:
    return _summarization_service


@router.post("/summarize", response_model=SummarizeResponse)
def summarize_text(
    request: SummarizeRequest,
    service: SummarizationService = Depends(get_summarization_service),
) -> SummarizeResponse:
    return service.summarize(request)