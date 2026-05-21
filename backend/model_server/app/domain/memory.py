from typing import Any, Literal

from pydantic import BaseModel, Field


MemoryType = Literal["episodic", "semantic", "procedural"]


class MemoryWriteRequest(BaseModel):
    text: str = Field(..., min_length=1)
    memory_type: MemoryType = "episodic"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    memory_id: str
    user_email: str
    memory_type: MemoryType
    text: str
    metadata: dict[str, Any]
    created_at: str


class MemorySearchRequest(BaseModel):
    query: str | None = None
    limit: int = Field(default=10, ge=1, le=100)


class MemorySearchResponse(BaseModel):
    memories: list[MemoryRecord]


class AuditLogRecord(BaseModel):
    audit_id: str
    actor: str
    action: str
    target_type: str
    target_id: str
    timestamp: str
    metadata: dict[str, Any]
