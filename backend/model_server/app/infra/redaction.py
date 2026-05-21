import re
from typing import Any


_GITHUB_TOKEN_RE = re.compile(
    r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{8,}\b|\bgithub_pat_[A-Za-z0-9_]{8,}\b"
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_BEARER_TOKEN_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b", re.IGNORECASE)
_PASSWORD_RE = re.compile(
    r"(?i)(\"password\"\s*:\s*\")([^\"]+)(\")|(\bpassword\s*[:=]\s*)([^\s,;]+)"
)
_API_KEY_RE = re.compile(
    r"(?i)(\"api_key\"\s*:\s*\")([^\"]+)(\")|(\bapi_key\s*[:=]\s*)([^\s,;]+)"
)


def redact_text(text: str) -> str:
    redacted = _GITHUB_TOKEN_RE.sub("[REDACTED_GITHUB_TOKEN]", text)
    redacted = _OPENAI_KEY_RE.sub("[REDACTED_API_KEY]", redacted)
    redacted = _BEARER_TOKEN_RE.sub("[REDACTED_BEARER_TOKEN]", redacted)
    redacted = _PASSWORD_RE.sub(_replace_password, redacted)
    redacted = _API_KEY_RE.sub(_replace_api_key, redacted)
    return redacted


def redact_obj(obj: Any) -> Any:
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, list):
        return [redact_obj(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(redact_obj(item) for item in obj)
    if isinstance(obj, dict):
        return {key: redact_obj(value) for key, value in obj.items()}
    if hasattr(obj, "model_dump"):
        return redact_obj(obj.model_dump(mode="json"))
    return obj


def _replace_password(match: re.Match[str]) -> str:
    if match.group(1):
        return f'{match.group(1)}[REDACTED_PASSWORD]{match.group(3)}'
    return f"{match.group(4)}[REDACTED_PASSWORD]"


def _replace_api_key(match: re.Match[str]) -> str:
    if match.group(1):
        return f'{match.group(1)}[REDACTED_API_KEY]{match.group(3)}'
    return f"{match.group(4)}[REDACTED_API_KEY]"


# Authenticated actor emails intentionally remain visible in audit logs so maintainers
# can identify who performed an explicit memory write in this local/demo layer.
