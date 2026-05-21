import json
import os
from typing import TypedDict
from urllib.error import URLError
from urllib.request import Request, urlopen


class VaultConfig(TypedDict):
    addr: str
    token: str


def get_vault_config() -> VaultConfig:
    return {
        "addr": os.getenv("VAULT_ADDR", "http://localhost:8200").rstrip("/"),
        "token": os.getenv("VAULT_DEV_ROOT_TOKEN_ID") or os.getenv("VAULT_TOKEN", "root"),
    }


def vault_available(timeout_seconds: float = 1.0) -> bool:
    config = get_vault_config()
    try:
        with urlopen(f"{config['addr']}/v1/sys/health", timeout=timeout_seconds) as response:
            return response.status in (200, 429, 472, 473, 501, 503)
    except (OSError, URLError):
        return False


def get_secret(path: str, key: str) -> str | None:
    config = get_vault_config()
    request = Request(
        f"{config['addr']}/v1/{path.lstrip('/')}",
        headers={"X-Vault-Token": config["token"]},
    )
    try:
        with urlopen(request, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    data = payload.get("data", {})
    nested_data = data.get("data", {})
    return nested_data.get(key) or data.get(key)
