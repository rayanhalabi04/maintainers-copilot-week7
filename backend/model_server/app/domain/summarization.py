from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max_sentences: int = 3


class SummarizeResponse(BaseModel):
    summary: str