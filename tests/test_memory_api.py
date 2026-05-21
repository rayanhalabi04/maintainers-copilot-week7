from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_chat_service
from app.api.deps import get_auth_service
from app.api.memory import get_memory_service
from app.domain.chat import ChatRequest, ChatResponse
from app.main import app
from app.repositories.memory_repository import LocalDemoMemoryRepository
from app.repositories.audit_repository import AuditEventRepository
from app.services.audit_service import AuditService
from app.services.auth_service import AuthService
from app.services.memory_service import MemoryService


class NoWriteChatService:
    def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            answer="Memory writes are explicit only.",
            tool_calls=[],
            trace={"memory_writes": "explicit_only"},
        )


@pytest.fixture()
def client_with_demo_stores(tmp_path: Path):
    auth_service = AuthService(
        user_store_path=tmp_path / "auth" / "demo_users.json",
        jwt_secret="test-secret",
    )
    repository = LocalDemoMemoryRepository(store_dir=tmp_path / "memory")
    audit_repository = AuditEventRepository(path=tmp_path / "audit" / "audit_events.jsonl")
    memory_service = MemoryService(
        repository=repository,
        audit_service=AuditService(repository=audit_repository),
    )

    app.dependency_overrides[get_auth_service] = lambda: auth_service
    app.dependency_overrides[get_memory_service] = lambda: memory_service
    app.dependency_overrides[get_chat_service] = lambda: NoWriteChatService()
    try:
        yield TestClient(app), repository, audit_repository
    finally:
        app.dependency_overrides.clear()


def login(client: TestClient, email: str, password: str) -> str:
    response = client.post(
        "/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def write_demo_memory(client: TestClient, token: str, text: str = "Remember flaky timers."):
    return client.post(
        "/memory/write",
        headers=auth_headers(token),
        json={
            "text": text,
            "memory_type": "episodic",
            "metadata": {"repo": "nodejs/node"},
        },
    )


def test_user_can_login_and_write_memory(client_with_demo_stores):
    client, _repository, _audit_repository = client_with_demo_stores
    token = login(client, "user@example.com", "user123")

    response = write_demo_memory(client, token)

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_email"] == "user@example.com"
    assert payload["memory_type"] == "episodic"
    assert payload["text"] == "Remember flaky timers."


def test_memory_write_creates_audit_log(client_with_demo_stores):
    client, repository, audit_repository = client_with_demo_stores
    token = login(client, "user@example.com", "user123")

    response = write_demo_memory(client, token)

    assert response.status_code == 200
    audit_logs = repository.list_audit()
    assert len(audit_logs) == 1
    assert audit_logs[0].action == "write_memory"
    assert audit_logs[0].target_id == response.json()["memory_id"]
    audit_events = audit_repository.list_events()
    assert len(audit_events) == 1
    assert audit_events[0].action == "write_memory"
    assert audit_events[0].target_id == response.json()["memory_id"]


def test_user_can_list_own_memories(client_with_demo_stores):
    client, _repository, _audit_repository = client_with_demo_stores
    token = login(client, "user@example.com", "user123")
    write_demo_memory(client, token, text="Remember stream regression context.")

    response = client.get("/memory", headers=auth_headers(token))

    assert response.status_code == 200
    memories = response.json()["memories"]
    assert len(memories) == 1
    assert memories[0]["text"] == "Remember stream regression context."


def test_user_can_search_own_memories(client_with_demo_stores):
    client, _repository, _audit_repository = client_with_demo_stores
    token = login(client, "user@example.com", "user123")
    write_demo_memory(client, token, text="Timers are flaky on Windows.")
    write_demo_memory(client, token, text="Docs issue about buffer examples.")

    response = client.post(
        "/memory/search",
        headers=auth_headers(token),
        json={"query": "timers windows", "limit": 5},
    )

    assert response.status_code == 200
    memories = response.json()["memories"]
    assert len(memories) == 1
    assert memories[0]["text"] == "Timers are flaky on Windows."


def test_admin_can_access_memory_audit(client_with_demo_stores):
    client, _repository, _audit_repository = client_with_demo_stores
    user_token = login(client, "user@example.com", "user123")
    admin_token = login(client, "admin@example.com", "admin123")
    write_demo_memory(client, user_token)

    response = client.get("/memory/audit", headers=auth_headers(admin_token))

    assert response.status_code == 200
    assert response.json()[0]["action"] == "write_memory"


def test_normal_user_cannot_access_memory_audit(client_with_demo_stores):
    client, _repository, _audit_repository = client_with_demo_stores
    token = login(client, "user@example.com", "user123")

    response = client.get("/memory/audit", headers=auth_headers(token))

    assert response.status_code == 403


def test_chat_does_not_auto_write_memory(client_with_demo_stores):
    client, repository, _audit_repository = client_with_demo_stores
    token = login(client, "user@example.com", "user123")

    response = client.post(
        "/chat",
        headers=auth_headers(token),
        json={"message": "Remember that timers are flaky on Windows."},
    )

    assert response.status_code == 200
    assert repository.list_memories("user@example.com") == []
    assert repository.list_audit() == []
