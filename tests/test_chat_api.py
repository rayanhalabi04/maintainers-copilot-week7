from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.chat import get_chat_service
from app.api.deps import get_auth_service
from app.domain.chat import ChatRequest
from app.domain.classification import ClassifyResponse
from app.main import app
from app.services.auth_service import AuthService
from app.services.chat_service import ChatService
from app.services.chat_tools import ToolResult


class ApiFakeTools:
    def classify_issue_tool(self, text, title=None, body=None):
        return ToolResult(
            tool_name="classify_issue",
            status="ok",
            summary="bug (0.86)",
            data=ClassifyResponse(
                label="bug",
                confidence=0.86,
                probabilities={"bug": 0.86, "feature": 0.04, "docs": 0.05, "question": 0.05},
                model_name="fake",
            ),
        )

    def extract_entities_tool(self, text):
        return ToolResult("extract_entities", "ok", summary="No code-shaped entities found.")

    def summarize_issue_tool(self, text):
        return ToolResult("summarize_thread", "ok", summary="Summary.")

    def rag_search_tool(self, question, top_k=5):
        return ToolResult("rag_query", "ok", summary="No strong evidence.")


class FailingClassifierTools(ApiFakeTools):
    def classify_issue_tool(self, text, title=None, body=None):
        return ToolResult(
            tool_name="classify_issue",
            status="error",
            error="Classifier tool failed.",
        )


@pytest.fixture()
def client_with_auth_and_chat(tmp_path: Path):
    auth_service = AuthService(
        user_store_path=tmp_path / "demo_users.json",
        jwt_secret="test-secret",
    )
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    app.dependency_overrides[get_chat_service] = lambda: ChatService(tools=ApiFakeTools())
    try:
        yield TestClient(app)
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


def chat_payload() -> dict:
    return {
        "message": "Please triage this issue.",
        "issue_title": "fs test fails",
        "issue_body": "The test fails with an assertion.",
        "use_rag": False,
    }


def test_chat_endpoint_without_token_returns_401(client_with_auth_and_chat: TestClient):
    response = client_with_auth_and_chat.post("/chat", json=chat_payload())

    assert response.status_code == 401


def test_chat_endpoint_with_user_token_returns_answer_and_tool_calls(
    client_with_auth_and_chat: TestClient,
):
    token = login(client_with_auth_and_chat, "user@example.com", "user123")

    response = client_with_auth_and_chat.post(
        "/chat",
        headers=auth_headers(token),
        json=chat_payload(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["tool_calls"]
    assert payload["tool_calls"][0]["tool_name"] == "classify_issue"
    assert payload["trace"]["authenticated_user"] == {
        "email": "user@example.com",
        "role": "user",
    }


def test_chat_endpoint_with_admin_token_returns_200(client_with_auth_and_chat: TestClient):
    token = login(client_with_auth_and_chat, "admin@example.com", "admin123")

    response = client_with_auth_and_chat.post(
        "/chat",
        headers=auth_headers(token),
        json=chat_payload(),
    )

    assert response.status_code == 200
    assert response.json()["trace"]["authenticated_user"]["role"] == "admin"


def test_chat_endpoint_tool_orchestration_still_works_with_auth(
    client_with_auth_and_chat: TestClient,
):
    token = login(client_with_auth_and_chat, "user@example.com", "user123")

    response = client_with_auth_and_chat.post(
        "/chat",
        headers=auth_headers(token),
        json={
            "message": "Find similar resolved issues and maintainer guidance.",
            "issue_title": "test.js fails with ERR_ASSERTION on node v20.1.0",
            "issue_body": "The failing file is test/parallel/test-fs.js.",
            "use_rag": True,
        },
    )

    assert response.status_code == 200
    tool_names = [tool_call["tool_name"] for tool_call in response.json()["tool_calls"]]
    assert "classify_issue" in tool_names
    assert "extract_entities" in tool_names
    assert "rag_query" in tool_names


def test_chat_endpoint_triage_request_with_issue_context_uses_rag(
    client_with_auth_and_chat: TestClient,
):
    token = login(client_with_auth_and_chat, "user@example.com", "user123")

    response = client_with_auth_and_chat.post(
        "/chat",
        headers=auth_headers(token),
        json={
            "message": "Can you triage this issue?",
            "issue_title": "Build fails on macOS with node-gyp error",
            "issue_body": "When I run npm install, node-gyp fails with a Python version error.",
            "use_rag": True,
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace"]["routing"]["rag_query"] is True
    tool_names = [tool_call["tool_name"] for tool_call in payload["tool_calls"]]
    assert "rag_query" in tool_names


def test_chat_endpoint_survives_failed_tool_with_auth(tmp_path: Path):
    app.dependency_overrides[get_chat_service] = lambda: ChatService(tools=ApiFakeTools())
    auth_service = AuthService(
        user_store_path=tmp_path / "demo_users.json",
        jwt_secret="test-secret",
    )
    app.dependency_overrides[get_auth_service] = lambda: auth_service
    client = TestClient(app)
    token = login(client, "user@example.com", "user123")
    app.dependency_overrides[get_chat_service] = lambda: ChatService(tools=FailingClassifierTools())
    try:
        response = client.post(
            "/chat",
            headers=auth_headers(token),
            json=ChatRequest(
                message="Please triage this bug.",
                issue_title="Regression in streams",
                issue_body="Streams throw unexpectedly.",
                use_rag=False,
            ).model_dump(),
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["tool_calls"][0]["status"] == "error"
    assert "stack" not in payload["tool_calls"][0]["error"].lower()
