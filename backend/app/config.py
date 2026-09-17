"""VoiceGuard application configuration.

Environment-driven via pydantic-settings. Twelve-factor compliant.
Startup fails loudly if JWT_SECRET is unset in production or any
required model artefact is unreachable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"

    # ── Database ─────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/voiceguard.db"

    # ── Storage ──────────────────────────────────────────────────
    storage_backend: Literal["local", "s3"] = "local"
    storage_local_root: str = "./data/artifacts"
    s3_endpoint: str = ""
    s3_bucket: str = ""
    s3_access_key: str = ""
    s3_secret_key: str = ""

    # ── Authentication ───────────────────────────────────────────
    jwt_secret: str = ""
    jwt_access_ttl_minutes: int = 15
    jwt_refresh_ttl_days: int = 14
    jwt_algorithm: str = "HS256"

    # ── AI Models ────────────────────────────────────────────────
    acoustic_model_id: str = ""
    acoustic_borderline_low: float = 0.35
    acoustic_borderline_high: float = 0.65

    whisper_model_size: Literal["base", "small", "medium"] = "small"
    whisper_compute_type: Literal["int8", "float16", "float32"] = "int8"
    whisper_languages: str = "en,hi,mr,bn,ta"

    scam_model_id: str = ""
    fusion_model_path: str = ""

    # ── Verdict thresholds ───────────────────────────────────────
    verdict_threshold_moderate: float = 0.30
    verdict_threshold_high: float = 0.65

    # ── Upload limits ────────────────────────────────────────────
    max_upload_mb: int = 25
    max_duration_seconds: float = 300.0
    min_duration_seconds: float = 1.5

    # ── Concurrency ──────────────────────────────────────────────
    max_concurrent_inference: int = 2

    # ── Device ───────────────────────────────────────────────────
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # ── Retention ────────────────────────────────────────────────
    retention_days: int = 30

    # ── CORS ─────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:5173"

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived ──────────────────────────────────────────────────
    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def whisper_language_list(self) -> list[str]:
        """Parse comma-separated supported Whisper languages."""
        return [l.strip() for l in self.whisper_languages.split(",") if l.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def storage_local_path(self) -> Path:
        return Path(self.storage_local_root)

    @field_validator("jwt_secret")
    @classmethod
    def _check_jwt_secret(cls, v: str, info: object) -> str:
        """JWT_SECRET must be set. In production, fail loudly."""
        # We cannot access other fields easily in a validator, so
        # the production check is enforced at startup in main.py.
        return v

    def validate_production(self) -> None:
        """Call at startup to enforce production-mode constraints."""
        if self.app_env == "production":
            if not self.jwt_secret or self.jwt_secret == "CHANGE_ME_GENERATE_A_REAL_SECRET":
                raise ValueError(
                    "JWT_SECRET must be set to a real secret in production. "
                    "Generate one: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
                )
            if self.log_level == "DEBUG":
                import warnings

                warnings.warn(
                    "LOG_LEVEL=DEBUG in production may log sensitive transcript content. "
                    "Set LOG_LEVEL=INFO or higher.",
                    UserWarning,
                    stacklevel=2,
                )

    def resolve_device(self) -> str:
        """Resolve 'auto' to 'cuda' or 'cpu'."""
        if self.device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


def get_settings() -> Settings:
    """Singleton-style settings loader."""
    return Settings()
