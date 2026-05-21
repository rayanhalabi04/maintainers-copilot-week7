from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


SourceType = Literal["doc", "issue"]
SourceTypeFilter = Literal["all", "doc", "issue"]


class RagFilters(BaseModel):
    source_type: SourceType | None = None
    labels: list[str] | None = None
    path: str | None = None
    repo: str | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    resolved_after: datetime | None = None
    resolved_before: datetime | None = None


class RagQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    source_type: SourceTypeFilter | None = None
    filters: RagFilters | None = None


class RagSource(BaseModel):
    chunk_id: str
    document_id: str
    source_type: SourceType
    source_id: str
    title: str
    url: str | None = None
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalDebug(BaseModel):
    original_query: str
    rewritten_query: str
    routed_source_type: SourceTypeFilter = "all"
    query_variants: list[str] = Field(default_factory=list)
    dense_top_k: int
    sparse_top_k: int
    hybrid_top_k: int
    reranked_top_k: int
    hybrid_alpha: float
    reranker_enabled: bool
    reranker_used: bool = False
    reranker_model: str | None = None
    final_chunk_ids: list[str] = Field(default_factory=list)
    reranked_chunk_ids: list[str] = Field(default_factory=list)
    scores: dict[str, dict[str, float]] = Field(default_factory=dict)


class RagQueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[RagSource]
    chunks: list[RagSource] = Field(default_factory=list)
    retrieval_debug: RetrievalDebug
    trace: dict[str, Any] = Field(default_factory=dict)


class RagDocument(BaseModel):
    id: str
    source_type: SourceType
    source_id: str
    title: str
    text: str
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class RagChunk(BaseModel):
    id: str
    document_id: str
    source_type: SourceType
    source_id: str
    title: str
    text: str
    chunk_index: int
    url: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class RetrievedChunk(RagSource):
    dense_score: float = 0.0
    sparse_score: float = 0.0
