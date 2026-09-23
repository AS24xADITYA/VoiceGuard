"""Security utilities: password hashing and JWT token management."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.config import Settings

# Argon2id password hashing per spec: time_cost=3, memory_cost=65536, parallelism=4
_pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__time_cost=3,
    argon2__memory_cost=65536,
    argon2__parallelism=4,
    argon2__type="ID",
)


def hash_password(password: str) -> str:
    """Hash a password using Argon2id."""
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its Argon2id hash. Constant-time."""
    return _pwd_context.verify(plain_password, hashed_password)


def hash_token(token: str) -> str:
    """SHA-256 hash of a refresh token. Raw tokens are never stored."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_refresh_token() -> str:
    """Generate a cryptographically random refresh token."""
    return secrets.token_urlsafe(48)


def create_refresh_token(ttl_days: int = 14) -> tuple[str, str, datetime]:
    """Generate a refresh token, its SHA-256 hash, and expiration datetime."""
    raw = generate_refresh_token()
    token_hash = hash_token(raw)
    expires_at = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=ttl_days)
    return raw, token_hash, expires_at


def create_access_token(
    subject_or_claims: str | dict[str, Any],
    settings: Settings | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    """Create a short-lived JWT access token.

    Accepts either a string subject or a claims dict.
    Claims: sub, exp, iat, jti.
    """
    if settings is None:
        from app.config import get_settings
        settings = get_settings()

    now = datetime.now(UTC)
    expire = now + timedelta(minutes=settings.jwt_access_ttl_minutes)

    base_claims: dict[str, Any] = {
        "type": "access",
        "iat": now,
        "exp": expire,
        "jti": secrets.token_urlsafe(16),
    }

    if isinstance(subject_or_claims, dict):
        claims = {**base_claims, **subject_or_claims}
    else:
        claims = {**base_claims, "sub": str(subject_or_claims)}
    if extra_claims:
        claims.update(extra_claims)
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str, settings: Settings | None = None) -> dict[str, Any]:
    """Decode and validate a JWT access token.

    Raises JWTError on invalid or expired tokens.
    """
    if settings is None:
        from app.config import get_settings
        settings = get_settings()

    return jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )


decode_token = decode_access_token


def create_guest_session_token(
    session_id: str | None = None,
    settings: Settings | None = None,
    ttl_hours: int = 24,
) -> tuple[str, str]:
    """Create a cryptographically signed guest session token (JWT) with 24-hour TTL per 14 §4.

    Returns (raw_signed_token, session_id).
    """
    if settings is None:
        from app.config import get_settings
        settings = get_settings()

    if session_id is None:
        session_id = secrets.token_urlsafe(32)

    now = datetime.now(UTC)
    expire = now + timedelta(hours=ttl_hours)

    claims: dict[str, Any] = {
        "type": "guest_session",
        "sub": "guest",
        "session_id": session_id,
        "iat": now,
        "exp": expire,
        "jti": secrets.token_urlsafe(16),
    }
    token = jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, session_id


def verify_guest_session_token(
    token: str | None,
    settings: Settings | None = None,
) -> str | None:
    """Verify a signed guest session token and return the validated session_id.

    Returns None if token is invalid, expired, or malformed.
    """
    if not token:
        return None
    if settings is None:
        from app.config import get_settings
        settings = get_settings()

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
        if payload.get("type") == "guest_session" and payload.get("session_id"):
            return str(payload["session_id"])
        if payload.get("session_id"):
            return str(payload["session_id"])
        return None
    except Exception:
        return None


# Common password denylist — a minimal set for the prototype.
# In production this would be loaded from a file.
COMMON_PASSWORDS: set[str] = {
    "password123",
    "12345678910",
    "qwertyuiop",
    "1234567890",
    "letmein1234",
    "password1234",
    "123456789012",
    "iloveyou123",
    "admin12345",
    "welcome1234",
    "monkey12345",
    "master12345",
    "dragon12345",
    "login12345",
    "princess123",
    "abcdefghij",
    "trustno1234",
    "sunshine123",
    "password12345",
    "changeme1234",
}


def validate_password_strength(password: str) -> str | None:
    """Validate password meets policy. Returns error message or None."""
    if len(password) < 10:
        return "Password must be at least 10 characters."
    if password.lower() in COMMON_PASSWORDS:
        return "This password is too common. Choose a different one."
    return None
