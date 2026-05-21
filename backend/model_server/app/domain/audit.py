from typing import Any

from pydantic import BaseModel, Field


class AuditEventCreate(BaseModel):
    actor: str
    action: str
    target_type: str
    target_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditEvent(AuditEventCreate):
    audit_id: str
    timestamp: str
