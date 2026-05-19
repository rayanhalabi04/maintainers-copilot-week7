from fastapi import APIRouter, Depends

from app.domain.ner import NerRequest, NerResponse
from app.services.ner_service import NerService


router = APIRouter(tags=["ner"])

_ner_service = NerService()


def get_ner_service() -> NerService:
    return _ner_service


@router.post("/ner", response_model=NerResponse)
def extract_entities(
    request: NerRequest,
    service: NerService = Depends(get_ner_service),
) -> NerResponse:
    return service.extract_entities(request)