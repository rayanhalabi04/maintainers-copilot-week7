from typing import Any

from pydantic import BaseModel, Field


class WidgetConfig(BaseModel):
    widget_id: str
    theme: dict[str, Any] = Field(default_factory=dict)
    greeting: str
    enabled_tools: list[str] = Field(default_factory=list)
