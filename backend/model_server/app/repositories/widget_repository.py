import json
import os
from pathlib import Path
from typing import Any


DEFAULT_WIDGET_CONFIG = {
    "widget_id": "demo-widget",
    "theme": {
        "primaryColor": "#2563eb",
        "position": "bottom-right",
    },
    "greeting": "Hi, I'm the Maintainer's Copilot.",
    "enabled_tools": ["classify", "ner", "summarize", "rag"],
}


class WidgetConfigRepository:
    def __init__(self, path: Path | None = None) -> None:
        configured_path = os.getenv("WIDGET_CONFIG_PATH")
        self.path = path or Path(configured_path or "data/widget/widget_configs.json")

    def load_configs(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {DEFAULT_WIDGET_CONFIG["widget_id"]: DEFAULT_WIDGET_CONFIG}
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def save_config(self, widget_id: str, config: dict[str, Any]) -> None:
        configs = self.load_configs()
        configs[widget_id] = config
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(configs, file, indent=2, sort_keys=True)
