from app.domain.auth import CurrentUser
from app.domain.memory import (
    AuditLogRecord,
    MemoryRecord,
    MemorySearchRequest,
    MemoryWriteRequest,
)
from app.infra.redaction import redact_obj, redact_text
from app.repositories.memory_repository import LocalDemoMemoryRepository
from app.services.audit_service import AuditService


class MemoryService:
    """Explicit local/demo memory service; chat must not auto-write here."""

    def __init__(
        self,
        repository: LocalDemoMemoryRepository | None = None,
        audit_service: AuditService | None = None,
    ) -> None:
        self.repository = repository or LocalDemoMemoryRepository()
        self.audit_service = audit_service

    def write_memory(
        self,
        current_user: CurrentUser,
        request: MemoryWriteRequest,
    ) -> MemoryRecord:
        redacted_metadata = redact_obj(request.metadata)
        memory = self.repository.write_memory(
            user_email=current_user.email,
            text=redact_text(request.text),
            memory_type=request.memory_type,
            metadata=redacted_metadata,
        )
        self.repository.write_audit(
            actor=current_user.email,
            action="write_memory",
            target_type="memory",
            target_id=memory.memory_id,
            metadata=redact_obj(
                {
                    "memory_type": memory.memory_type,
                    "metadata": redacted_metadata,
                }
            ),
        )
        audit_service = self.audit_service or AuditService()
        audit_service.record_event(
            actor=current_user.email,
            action="write_memory",
            target_type="memory",
            target_id=memory.memory_id,
            metadata={
                "memory_type": memory.memory_type,
                "metadata": redacted_metadata,
            },
        )
        return memory

    def list_user_memories(
        self,
        current_user: CurrentUser,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        return self.repository.list_memories(current_user.email, limit=limit)

    def search_user_memories(
        self,
        current_user: CurrentUser,
        request: MemorySearchRequest,
    ) -> list[MemoryRecord]:
        return self.repository.search_memories(
            current_user.email,
            request.query,
            limit=request.limit,
        )

    def list_audit_logs(
        self,
        current_user: CurrentUser,
        limit: int = 50,
    ) -> list[AuditLogRecord]:
        if current_user.role != "admin":
            return []
        return self.repository.list_audit(limit=limit)
