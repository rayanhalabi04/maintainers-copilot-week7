from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_auth_service
from app.main import app
from app.services.auth_service import AuthService


@pytest.fixture()
def client_with_auth_service(tmp_path: Path):
    auth_service = AuthService(
        user_store_path=tmp_path / "demo_users.json",
        jwt_secret="test-secret",
    )
    app.dependency_overrides[get_auth_service] = lambda: auth_service
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


def test_register_user_succeeds(client_with_auth_service: TestClient):
    response = client_with_auth_service.post(
        "/auth/register",
        json={
            "email": "new-user@example.com",
            "password": "secret123",
            "role": "user",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {"email": "new-user@example.com", "role": "user"}


def test_login_returns_token(client_with_auth_service: TestClient):
    token = login(client_with_auth_service, "user@example.com", "user123")

    assert token.count(".") == 2


def test_auth_me_works_with_token(client_with_auth_service: TestClient):
    token = login(client_with_auth_service, "user@example.com", "user123")

    response = client_with_auth_service.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == {"email": "user@example.com", "role": "user"}


def test_auth_me_fails_without_token(client_with_auth_service: TestClient):
    response = client_with_auth_service.get("/auth/me")

    assert response.status_code == 401


def test_admin_ping_works_for_admin_token(client_with_auth_service: TestClient):
    token = login(client_with_auth_service, "admin@example.com", "admin123")

    response = client_with_auth_service.get(
        "/admin/ping",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "role": "admin"}


def test_admin_ping_fails_for_user_token(client_with_auth_service: TestClient):
    token = login(client_with_auth_service, "user@example.com", "user123")

    response = client_with_auth_service.get(
        "/admin/ping",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 403
