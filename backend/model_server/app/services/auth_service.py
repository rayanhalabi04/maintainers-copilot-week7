import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path
from typing import Any

from app.domain.auth import CurrentUser, Role


ALLOWED_ROLES: set[str] = {"user", "admin"}
DEFAULT_DEMO_USERS = (
    ("admin@example.com", "admin123", "admin"),
    ("user@example.com", "user123", "user"),
)


class AuthError(Exception):
    pass


class DuplicateUserError(AuthError):
    pass


class InvalidCredentialsError(AuthError):
    pass


class InvalidTokenError(AuthError):
    pass


class AuthService:
    """Minimal local/demo auth service; replace with persistent production auth later."""

    def __init__(
        self,
        user_store_path: Path | None = None,
        jwt_secret: str | None = None,
        token_ttl_seconds: int = 3600,
    ) -> None:
        self.user_store_path = user_store_path or Path("data/auth/demo_users.json")
        # Local-dev fallback only. Production should supply this from Vault or env.
        self.jwt_secret = jwt_secret or os.getenv(
            "JWT_SECRET", "local-demo-secret-change-me"
        )
        self.token_ttl_seconds = token_ttl_seconds
        self._ensure_store()

    def register_user(
        self,
        email: str,
        password: str,
        role: str = "user",
    ) -> CurrentUser:
        normalized_email = self._normalize_email(email)
        normalized_role = self._validate_role(role)
        users = self._load_users()
        if normalized_email in users:
            raise DuplicateUserError("User already exists.")

        users[normalized_email] = {
            "email": normalized_email,
            "role": normalized_role,
            "password_hash": self._hash_password(password),
        }
        self._save_users(users)
        return CurrentUser(email=normalized_email, role=normalized_role)  # type: ignore[arg-type]

    def authenticate_user(self, email: str, password: str) -> CurrentUser:
        normalized_email = self._normalize_email(email)
        user_record = self._load_users().get(normalized_email)
        if not user_record or not self._verify_password(
            password, user_record["password_hash"]
        ):
            raise InvalidCredentialsError("Invalid email or password.")
        return CurrentUser(email=user_record["email"], role=user_record["role"])

    def create_access_token(self, user: CurrentUser) -> str:
        now = int(time.time())
        payload = {
            "sub": user.email,
            "email": user.email,
            "role": user.role,
            "exp": now + self.token_ttl_seconds,
            "iat": now,
        }
        return self._encode_jwt(payload)

    def decode_access_token(self, token: str) -> dict[str, Any]:
        try:
            header_segment, payload_segment, signature_segment = token.split(".")
        except ValueError as exc:
            raise InvalidTokenError("Invalid token.") from exc

        signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
        expected_signature = self._sign(signing_input)
        actual_signature = self._base64url_decode(signature_segment)
        if not hmac.compare_digest(expected_signature, actual_signature):
            raise InvalidTokenError("Invalid token signature.")

        try:
            header = json.loads(self._base64url_decode(header_segment))
            payload = json.loads(self._base64url_decode(payload_segment))
        except (json.JSONDecodeError, ValueError) as exc:
            raise InvalidTokenError("Invalid token payload.") from exc

        if header.get("alg") != "HS256" or header.get("typ") != "JWT":
            raise InvalidTokenError("Unsupported token header.")
        if int(payload.get("exp", 0)) < int(time.time()):
            raise InvalidTokenError("Token expired.")
        if payload.get("role") not in ALLOWED_ROLES or not payload.get("sub"):
            raise InvalidTokenError("Invalid token claims.")
        return payload

    def get_current_user_from_token(self, token: str) -> CurrentUser:
        payload = self.decode_access_token(token)
        email = self._normalize_email(str(payload["sub"]))
        user_record = self._load_users().get(email)
        if not user_record:
            raise InvalidTokenError("Token user does not exist.")
        return CurrentUser(email=email, role=user_record["role"])

    def _ensure_store(self) -> None:
        if self.user_store_path.exists():
            return
        self.user_store_path.parent.mkdir(parents=True, exist_ok=True)
        users = {}
        for email, password, role in DEFAULT_DEMO_USERS:
            users[email] = {
                "email": email,
                "role": role,
                "password_hash": self._hash_password(password),
            }
        self._save_users(users)

    def _load_users(self) -> dict[str, dict[str, str]]:
        with self.user_store_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _save_users(self, users: dict[str, dict[str, str]]) -> None:
        with self.user_store_path.open("w", encoding="utf-8") as file:
            json.dump(users, file, indent=2, sort_keys=True)

    def _hash_password(self, password: str) -> str:
        salt = secrets.token_bytes(16)
        derived = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, 200_000
        )
        return f"pbkdf2_sha256$200000${self._base64url_encode(salt)}${self._base64url_encode(derived)}"

    def _verify_password(self, password: str, password_hash: str) -> bool:
        try:
            algorithm, iterations, salt, expected = password_hash.split("$")
        except ValueError:
            return False
        if algorithm != "pbkdf2_sha256":
            return False
        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            self._base64url_decode(salt),
            int(iterations),
        )
        return hmac.compare_digest(self._base64url_encode(derived), expected)

    def _encode_jwt(self, payload: dict[str, Any]) -> str:
        header = {"alg": "HS256", "typ": "JWT"}
        header_segment = self._base64url_encode_json(header)
        payload_segment = self._base64url_encode_json(payload)
        signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
        signature_segment = self._base64url_encode(self._sign(signing_input))
        return f"{header_segment}.{payload_segment}.{signature_segment}"

    def _sign(self, signing_input: bytes) -> bytes:
        return hmac.new(
            self.jwt_secret.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()

    def _base64url_encode_json(self, value: dict[str, Any]) -> str:
        raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return self._base64url_encode(raw)

    def _base64url_encode(self, value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    def _base64url_decode(self, value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(f"{value}{padding}")

    def _normalize_email(self, email: str) -> str:
        return email.strip().lower()

    def _validate_role(self, role: str) -> str:
        if role not in ALLOWED_ROLES:
            raise ValueError("Role must be user or admin.")
        return role
