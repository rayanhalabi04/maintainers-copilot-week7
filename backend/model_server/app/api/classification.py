from fastapi import APIRouter, Depends

from app.domain.classification import ClassifyRequest, ClassifyResponse
from app.services.classifier_service import ClassifierService


router = APIRouter(tags=["classification"])

_classifier_service = ClassifierService()


def get_classifier_service() -> ClassifierService:
    return _classifier_service


@router.post("/classify", response_model=ClassifyResponse)
def classify_issue(
    request: ClassifyRequest,
    service: ClassifierService = Depends(get_classifier_service),
) -> ClassifyResponse:
    return service.classify(request)