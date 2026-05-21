import os
import socket
from urllib.parse import urlparse


def get_redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0")


def redis_available(timeout_seconds: float = 1.0) -> bool:
    parsed = urlparse(get_redis_url())
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False
