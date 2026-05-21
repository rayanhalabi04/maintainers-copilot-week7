from dataclasses import dataclass
from typing import Any

from app.domain.auth import CurrentUser
from app.domain.classification import ClassifyRequest
from app.domain.memory import MemoryWriteRequest
from app.domain.ner import NerRequest
from app.domain.rag import RagQueryRequest
from app.domain.summarization import SummarizeRequest
from app.infra.redaction import redact_obj, redact_text
from app.services.classifier_service import ClassifierService
from app.services.memory_service import MemoryService
from app.services.ner_service import NerService
from app.services.rag_service import RagService
from app.services.summarization_service import SummarizationService


@dataclass
class ToolResult:
    tool_name: str
    status: str
    summary: str | None = None
    error: str | None = None
    data: Any | None = None


class ChatTools:
    def __init__(
        self,
        classifier_service: ClassifierService | None = None,
        ner_service: NerService | None = None,
        summarization_service: SummarizationService | None = None,
        rag_service: RagService | None = None,
        memory_service: MemoryService | None = None,
    ) -> None:
        self._classifier_service = classifier_service
        self._ner_service = ner_service
        self._summarization_service = summarization_service
        self._rag_service = rag_service
        self._memory_service = memory_service

    @property
    def classifier_service(self) -> ClassifierService:
        if self._classifier_service is None:
            self._classifier_service = ClassifierService()
        return self._classifier_service

    @property
    def ner_service(self) -> NerService:
        if self._ner_service is None:
            self._ner_service = NerService()
        return self._ner_service

    @property
    def summarization_service(self) -> SummarizationService:
        if self._summarization_service is None:
            self._summarization_service = SummarizationService()
        return self._summarization_service

    @property
    def rag_service(self) -> RagService:
        if self._rag_service is None:
            self._rag_service = RagService()
        return self._rag_service

    @property
    def memory_service(self) -> MemoryService:
        if self._memory_service is None:
            self._memory_service = MemoryService()
        return self._memory_service

    def classify_issue_tool(
        self,
        text: str,
        title: str | None = None,
        body: str | None = None,
    ) -> ToolResult:
        try:
            request_title = (title or self._fallback_title(text)).strip()
            request_body = body if body is not None else text
            response = self.classifier_service.classify(
                ClassifyRequest(title=request_title, body=request_body)
            )
            summary = redact_text(f"{response.label} ({response.confidence:.2f})")
            return ToolResult("classify_issue", "ok", summary=summary, data=response)
        except Exception:
            return ToolResult(
                "classify_issue",
                "error",
                error="Classifier tool failed.",
            )

    def extract_entities_tool(self, text: str) -> ToolResult:
        try:
            response = self.ner_service.extract_entities(NerRequest(text=text))
            if response.entities:
                labels = sorted({entity.label for entity in response.entities})
                summary = f"{len(response.entities)} entities: {', '.join(labels)}"
            else:
                summary = "No code-shaped entities found."
            redacted_response = response.model_copy(
                update={
                    "entities": [
                        entity.model_copy(update={"text": redact_text(entity.text)})
                        for entity in response.entities
                    ]
                }
            )
            return ToolResult(
                "extract_entities",
                "ok",
                summary=redact_text(summary),
                data=redacted_response,
            )
        except Exception:
            return ToolResult(
                "extract_entities",
                "error",
                error="Entity extraction tool failed.",
            )

    def summarize_issue_tool(self, text: str) -> ToolResult:
        try:
            response = self.summarization_service.summarize(SummarizeRequest(text=text))
            redacted_summary = redact_text(response.summary or "No summary produced.")
            redacted_response = response.model_copy(update={"summary": redacted_summary})
            return ToolResult(
                "summarize_thread",
                "ok",
                summary=redacted_summary,
                data=redacted_response,
            )
        except Exception:
            return ToolResult(
                "summarize_thread",
                "error",
                error="Summarizer tool failed.",
            )

    def rag_search_tool(self, question: str, top_k: int = 5) -> ToolResult:
        try:
            response = self.rag_service.query(RagQueryRequest(question=question, top_k=top_k))
            source_count = len(response.sources)
            redacted_response = self._redact_rag_response(response)
            summary = f"{source_count} sources. {redacted_response.answer}"
            return ToolResult(
                "rag_query",
                "ok",
                summary=redact_text(summary),
                data=redacted_response,
            )
        except Exception:
            return ToolResult(
                "rag_query",
                "error",
                error="RAG search tool failed.",
            )

    def write_memory_tool(
        self,
        current_user: CurrentUser,
        text: str,
        memory_type: str = "episodic",
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        try:
            response = self.memory_service.write_memory(
                current_user,
                MemoryWriteRequest(
                    text=text,
                    memory_type=memory_type,  # type: ignore[arg-type]
                    metadata=metadata or {},
                ),
            )
            return ToolResult(
                "write_memory",
                "ok",
                summary=f"Stored {response.memory_type} memory.",
                data=response,
            )
        except Exception:
            return ToolResult(
                "write_memory",
                "error",
                error="Memory write tool failed.",
            )

    def _fallback_title(self, text: str) -> str:
        first_line = text.strip().splitlines()[0] if text.strip() else "Issue"
        return first_line[:120] or "Issue"

    def _redact_rag_response(self, response):
        sources = [
            source.model_copy(
                update={
                    "title": redact_text(source.title),
                    "url": redact_text(source.url) if source.url else None,
                    "text": redact_text(source.text),
                    "metadata": redact_obj(source.metadata),
                }
            )
            for source in response.sources
        ]
        return response.model_copy(
            update={
                "question": redact_text(response.question),
                "answer": redact_text(response.answer),
                "sources": sources,
                "chunks": sources,
                "trace": redact_obj(response.trace),
            }
        )


_default_tools = ChatTools()


def classify_issue_tool(text: str, title: str | None = None, body: str | None = None) -> ToolResult:
    return _default_tools.classify_issue_tool(text=text, title=title, body=body)


def extract_entities_tool(text: str) -> ToolResult:
    return _default_tools.extract_entities_tool(text)


def summarize_issue_tool(text: str) -> ToolResult:
    return _default_tools.summarize_issue_tool(text)


def rag_search_tool(question: str, top_k: int = 5) -> ToolResult:
    return _default_tools.rag_search_tool(question, top_k)


def write_memory_tool(
    current_user: CurrentUser,
    text: str,
    memory_type: str = "episodic",
    metadata: dict[str, Any] | None = None,
) -> ToolResult:
    return _default_tools.write_memory_tool(current_user, text, memory_type, metadata)
