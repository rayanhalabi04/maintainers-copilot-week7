from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.domain.auth import CurrentUser
from app.domain.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


router = APIRouter(tags=["chat"])

_chat_service = ChatService()


def get_chat_service() -> ChatService:
    return _chat_service


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    current_user: CurrentUser = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        response = service.chat(request, current_user=current_user)
    except TypeError as exc:
        if "current_user" not in str(exc):
            raise
        response = service.chat(request)
    trace = dict(response.trace or {})
    trace["authenticated_user"] = {
        "email": current_user.email,
        "role": current_user.role,
    }
    return response.model_copy(update={"trace": trace})
