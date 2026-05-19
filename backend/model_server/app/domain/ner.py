from pydantic import BaseModel, Field


class NerRequest(BaseModel):
    text: str = Field(..., min_length=1)


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class NerResponse(BaseModel):
    entities: list[Entity]