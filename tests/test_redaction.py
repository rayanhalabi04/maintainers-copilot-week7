import json
from pathlib import Path

from app.domain.auth import CurrentUser
from app.domain.chat import ChatRequest
from app.domain.classification import ClassifyResponse
from app.domain.summarization import SummarizeResponse
from app.infra.redaction import redact_text
from app.repositories.memory_repository import LocalDemoMemoryRepository
from app.repositories.audit_repository import AuditEventRepository
from app.services.audit_service import AuditService
from app.services.memory_service import MemoryService
from app.domain.memory import MemoryWriteRequest
from app.services.chat_service import ChatService
from app.services.chat_tools import ToolResult


GITHUB_TOKEN = "github_pat_FAKESECRET1234567890"
OPENAI_KEY = "sk-FAKESECRET1234567890"
BEARER_TOKEN = "Bearer fakeBearerToken1234567890"


def test_redact_text_redacts_fake_github_token():
    redacted = redact_text(f"token={GITHUB_TOKEN}")

    assert GITHUB_TOKEN not in redacted
    assert "[REDACTED_GITHUB_TOKEN]" in redacted


def test_redact_text_redacts_fake_openai_key():
    redacted = redact_text(f"api_key={OPENAI_KEY}")

    assert OPENAI_KEY not in redacted
    assert "[REDACTED_API_KEY]" in redacted


def test_redact_text_redacts_bearer_token():
    redacted = redact_text(f"Authorization: {BEARER_TOKEN}")

    assert BEARER_TOKEN not in redacted
    assert "[REDACTED_BEARER_TOKEN]" in redacted


def test_memory_write_stores_redacted_value_not_raw_token(tmp_path: Path):
    repository = LocalDemoMemoryRepository(store_dir=tmp_path / "memory")
    service = MemoryService(
        repository=repository,
        audit_service=AuditService(
            repository=AuditEventRepository(path=tmp_path / "audit" / "audit_events.jsonl")
        ),
    )
    user = CurrentUser(email="user@example.com", role="user")

    memory = service.write_memory(
        user,
        MemoryWriteRequest(
            text=f"Remember this GitHub token: {GITHUB_TOKEN}",
            metadata={"api_key": OPENAI_KEY, "nested": {"authorization": BEARER_TOKEN}},
        ),
    )

    raw_memory_file = repository.memories_path.read_text(encoding="utf-8")
    raw_audit_file = repository.audit_log_path.read_text(encoding="utf-8")

    assert GITHUB_TOKEN not in memory.text
    assert GITHUB_TOKEN not in raw_memory_file
    assert OPENAI_KEY not in raw_memory_file
    assert BEARER_TOKEN not in raw_memory_file
    assert OPENAI_KEY not in raw_audit_file
    assert BEARER_TOKEN not in raw_audit_file
    assert "[REDACTED_GITHUB_TOKEN]" in raw_memory_file
    assert "[REDACTED_API_KEY]" in raw_memory_file


class EchoSecretChatTools:
    def classify_issue_tool(self, text, title=None, body=None):
        return ToolResult(
            tool_name="classify_issue",
            status="ok",
            summary=f"classified with {OPENAI_KEY}",
            data=ClassifyResponse(
                label="bug",
                confidence=0.99,
                probabilities={"bug": 0.99},
                model_name="fake",
            ),
        )

    def extract_entities_tool(self, text):
        return ToolResult("extract_entities", "ok", summary=f"entity {GITHUB_TOKEN}")

    def summarize_issue_tool(self, text):
        return ToolResult(
            "summarize_thread",
            "ok",
            summary=f"summary {OPENAI_KEY}",
            data=SummarizeResponse(summary=f"summary includes api_key={OPENAI_KEY}"),
        )

    def rag_search_tool(self, question, top_k=5):
        return ToolResult("rag_query", "skipped", summary=None)


def test_chat_response_does_not_return_raw_fake_api_key():
    service = ChatService(tools=EchoSecretChatTools())
    request = ChatRequest(
        message=f"Please triage this issue. api_key={OPENAI_KEY} " + ("Details. " * 160),
        issue_title="Build fails",
        issue_body=f"Failure logs include api_key={OPENAI_KEY}. " + ("More context. " * 160),
        use_rag=False,
    )

    response = service.chat(request)
    response_json = response.model_dump_json()

    assert OPENAI_KEY not in response_json
    assert "[REDACTED_API_KEY]" in response_json
    assert json.loads(response_json)["trace"]["routing"]["classify_issue"] is True
