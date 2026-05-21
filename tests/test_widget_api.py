from fastapi.testclient import TestClient

from app.api.widget import get_widget_service
from app.main import app
from app.repositories.audit_repository import AuditEventRepository
from app.repositories.widget_repository import WidgetConfigRepository
from app.services.audit_service import AuditService
from app.services.widget_service import WidgetService


def test_widget_config_returns_demo_config():
    response = TestClient(app).get("/widget/config/demo-widget")

    assert response.status_code == 200
    payload = response.json()
    assert payload["widget_id"] == "demo-widget"
    assert payload["theme"]["primaryColor"] == "#2563eb"


def test_widget_loader_returns_javascript():
    response = TestClient(app).get("/widget.js")

    assert response.status_code == 200
    assert "application/javascript" in response.headers["content-type"]
    assert "maintainers-copilot:resize" in response.text
    assert "data-widget-id" in response.text


def test_widget_service_can_update_config_and_record_audit(tmp_path):
    repository = WidgetConfigRepository(path=tmp_path / "widget_configs.json")
    audit_repository = AuditEventRepository(path=tmp_path / "audit_events.jsonl")
    service = WidgetService(
        repository=repository,
        audit_service=AuditService(repository=audit_repository),
    )

    updated = service.update_config(
        "demo-widget",
        {
            "theme": {"primaryColor": "#111827"},
            "greeting": "Hello.",
            "enabled_tools": ["rag"],
        },
        actor="admin@example.com",
    )

    assert updated.widget_id == "demo-widget"
    assert service.get_config("demo-widget").theme["primaryColor"] == "#111827"
    events = audit_repository.list_events()
    assert events[0].action == "update_widget_config"
    assert events[0].target_id == "demo-widget"


def test_widget_api_accepts_service_override(tmp_path):
    repository = WidgetConfigRepository(path=tmp_path / "missing.json")
    service = WidgetService(repository=repository)
    app.dependency_overrides[get_widget_service] = lambda: service
    try:
        response = TestClient(app).get("/widget/config/demo-widget")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["widget_id"] == "demo-widget"
