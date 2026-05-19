from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    title: str = Field(..., min_length=1)
    body: str | None = ""


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str