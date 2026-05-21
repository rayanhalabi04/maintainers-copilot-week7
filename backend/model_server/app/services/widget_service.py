import os
from typing import Any

from app.domain.widget import WidgetConfig
from app.services.audit_service import AuditService
from app.repositories.widget_repository import WidgetConfigRepository


class WidgetConfigNotFoundError(Exception):
    pass


class WidgetService:
    def __init__(
        self,
        repository: WidgetConfigRepository | None = None,
        audit_service: AuditService | None = None,
    ) -> None:
        self.repository = repository or WidgetConfigRepository()
        self.audit_service = audit_service

    def get_config(self, widget_id: str) -> WidgetConfig:
        config = self.repository.load_configs().get(widget_id)
        if config is None:
            raise WidgetConfigNotFoundError("Widget config not found.")
        return WidgetConfig(**config)

    def widget_app_url(self) -> str:
        return os.getenv("WIDGET_APP_URL", "http://localhost:5173").rstrip("/")

    def build_loader_script(self) -> str:
        return f"""
(function () {{
  var currentScript = document.currentScript;
  var widgetId = currentScript && currentScript.getAttribute("data-widget-id") || "demo-widget";
  var widgetUrl = currentScript && currentScript.getAttribute("data-widget-url") || "{self.widget_app_url()}";
  var modelServerUrl = new URL(currentScript.src).origin;
  var iframe = document.createElement("iframe");
  iframe.title = "Maintainer's Copilot";
  iframe.src = widgetUrl + "?widget_id=" + encodeURIComponent(widgetId) +
    "&model_server_url=" + encodeURIComponent(modelServerUrl);
  iframe.style.position = "fixed";
  iframe.style.right = "20px";
  iframe.style.bottom = "20px";
  iframe.style.width = "88px";
  iframe.style.height = "88px";
  iframe.style.border = "0";
  iframe.style.zIndex = "2147483647";
  iframe.style.background = "transparent";
  iframe.style.colorScheme = "normal";
  iframe.allow = "clipboard-write";
  document.body.appendChild(iframe);

  window.addEventListener("message", function (event) {{
    if (!event.data || event.data.type !== "maintainers-copilot:resize") {{
      return;
    }}
    iframe.style.width = Number(event.data.width || 88) + "px";
    iframe.style.height = Number(event.data.height || 88) + "px";
  }});
}})();
""".strip()

    def update_config(
        self,
        widget_id: str,
        config: dict[str, Any],
        actor: str = "system",
    ) -> WidgetConfig:
        updated = dict(config)
        updated["widget_id"] = widget_id
        self.repository.save_config(widget_id, updated)
        audit_service = self.audit_service or AuditService()
        audit_service.record_event(
            actor=actor,
            action="update_widget_config",
            target_type="widget_config",
            target_id=widget_id,
            metadata={"widget_id": widget_id},
        )
        return WidgetConfig(**updated)
