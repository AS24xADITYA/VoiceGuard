"""SQLAlchemy ORM models per 07-DATA-MODELS-AND-SCHEMA.md.

UUIDv7 primary keys stored as 36-character strings.
Branch results live in JSON columns (write-once, read-whole).
Timestamps are UTC, timezone-aware.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

try:
    import uuid7

    def _new_id() -> str:
        return str(uuid7.create())
except ImportError:
    def _new_id() -> str:
        return str(uuid.uuid4())


def _utcnow() -> datetime:
    # Return a naive datetime so SQLite (which strips tz anyway) is consistent.
    # datetime.now(UTC) produces tz-aware; SQLite reads it back as naive —
    # mixing the two causes "can't subtract offset-naive and offset-aware" errors.
    return datetime.now(UTC).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    analysis_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    preferences: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Relationships
    analyses: Mapped[list[Analysis]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    refresh_tokens: Mapped[list[RefreshToken]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    ip_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    user: Mapped[User] = relationship(back_populates="refresh_tokens")


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    guest_session_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True, default="QUEUED")
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    progress_pct: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    correlation_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True, default=_new_id
    )

    # Source audio
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_type: Mapped[str] = mapped_column(String(16), nullable=False, default="UPLOAD")
    audio_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False, default=16000)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)

    # Quality
    quality: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Branch results (JSON blobs — write-once, read-whole)
    acoustic_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    transcript_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    scam_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    fusion_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    explanation_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Denormalised for listing
    risk_probability: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    verdict: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    acoustic_spoof_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    scam_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    detected_language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    is_borderline: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Versioning
    model_versions: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False, default="0.1.0")

    # Error
    error_code: Mapped[str | None] = mapped_column(String(48), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    degraded_branches: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    # Timing
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stage_timings_ms: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    # Label (user-supplied)
    label: Mapped[str | None] = mapped_column(String(80), nullable=True)

    # Relationships
    user: Mapped[User | None] = relationship(back_populates="analyses")
    challenges: Mapped[list[Challenge]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_analyses_user_created", "user_id", "created_at"),
        Index("ix_analyses_dedup", "audio_sha256", "user_id"),
        Index("ix_analyses_status_created", "status", "created_at"),
    )


class Challenge(Base):
    __tablename__ = "challenges"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    challenge_type: Mapped[str] = mapped_column(String(32), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    expected_phrase: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_duration_s: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="ISSUED")
    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    responded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    response_audio_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    response_duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    consistency_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    feature_deltas: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    expected_deltas: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    notes: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    analysis: Mapped[Analysis] = relationship(back_populates="challenges")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    analysis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    analysis: Mapped[Analysis] = relationship(back_populates="artifacts")


class SystemEvent(Base):
    """Lightweight audit trail for debugging."""

    __tablename__ = "system_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    analysis_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )
