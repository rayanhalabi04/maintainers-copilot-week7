from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user, require_admin
from app.domain.auth import CurrentUser
from app.domain.memory import (
    AuditLogRecord,
    MemoryRecord,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryWriteRequest,
)
from app.services.memory_service import MemoryService


router = APIRouter(tags=["memory"])

_memory_service = MemoryService()


def get_memory_service() -> MemoryService:
    return _memory_service


@router.post("/memory/write", response_model=MemoryRecord)
def write_memory(
    request: MemoryWriteRequest,
    current_user: CurrentUser = Depends(get_current_user),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemoryRecord:
    return memory_service.write_memory(current_user, request)


@router.get("/memory", response_model=MemorySearchResponse)
def list_memories(
    limit: int = Query(default=10, ge=1, le=100),
    current_user: CurrentUser = Depends(get_current_user),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemorySearchResponse:
    memories = memory_service.list_user_memories(current_user, limit=limit)
    return MemorySearchResponse(memories=memories)


@router.post("/memory/search", response_model=MemorySearchResponse)
def search_memories(
    request: MemorySearchRequest,
    current_user: CurrentUser = Depends(get_current_user),
    memory_service: MemoryService = Depends(get_memory_service),
) -> MemorySearchResponse:
    memories = memory_service.search_user_memories(current_user, request)
    return MemorySearchResponse(memories=memories)


@router.get("/memory/audit", response_model=list[AuditLogRecord])
def list_audit_logs(
    limit: int = Query(default=50, ge=1, le=100),
    current_user: CurrentUser = Depends(require_admin),
    memory_service: MemoryService = Depends(get_memory_service),
) -> list[AuditLogRecord]:
    return memory_service.list_audit_logs(current_user, limit=limit)
