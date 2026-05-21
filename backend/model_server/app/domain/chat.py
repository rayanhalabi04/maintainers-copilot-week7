from typing import Any, Literal

from pydantic import BaseModel, Field


ToolStatus = Literal["ok", "error", "skipped"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    issue_title: str | None = None
    issue_body: str | None = None
    use_rag: bool = True
    top_k: int = Field(default=5, ge=1, le=20)


class ChatToolCall(BaseModel):
    tool_name: str
    status: ToolStatus
    summary: str | None = None
    error: str | None = None


class ChatResponse(BaseModel):
    answer: str
    tool_calls: list[ChatToolCall]
    trace: dict[str, Any] | None = None
