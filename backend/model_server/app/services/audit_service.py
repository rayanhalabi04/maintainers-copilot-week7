from typing import Any

from app.domain.audit import AuditEvent, AuditEventCreate
from app.infra.redaction import redact_obj
from app.repositories.audit_repository import AuditEventRepository


class AuditService:
    def __init__(self, repository: AuditEventRepository | None = None) -> None:
        self.repository = repository or AuditEventRepository()

    def record_event(
        self,
        actor: str,
        action: str,
        target_type: str,
        target_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        return self.repository.write_event(
            AuditEventCreate(
                actor=actor,
                action=action,
                target_type=target_type,
                target_id=target_id,
                metadata=redact_obj(metadata or {}),
            )
        )

    # TODO: Wire role-change and conversation-deletion events when those features
    # gain write endpoints in this codebase.
