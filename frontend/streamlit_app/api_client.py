import os
from typing import Any

import requests


MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8001").rstrip("/")
TIMEOUT_SECONDS = 20


class ApiClientError(Exception):
    pass


def api_get(path: str, token: str | None = None) -> Any:
    return _request("GET", path, token=token)


def api_post(path: str, token: str | None = None, json: dict[str, Any] | None = None) -> Any:
    return _request("POST", path, token=token, json=json)


def _request(
    method: str,
    path: str,
    token: str | None = None,
    json: dict[str, Any] | None = None,
) -> Any:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{MODEL_SERVER_URL}{path}"
    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            json=json,
            timeout=TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ApiClientError(f"Could not reach model server at {MODEL_SERVER_URL}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise ApiClientError(
            f"{method} {path} returned non-JSON response with status {response.status_code}."
        ) from exc

    if response.status_code >= 400:
        detail = payload.get("detail", payload)
        raise ApiClientError(f"{method} {path} failed with {response.status_code}: {detail}")

    return payload
