"""Initial VoiceGuard schema migration.

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-09-17 12:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── 1. users table ──
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=80), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("analysis_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("preferences", sa.JSON(), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # ── 2. refresh_tokens table ──
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.String(length=255), nullable=False, unique=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column("ip_hash", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"])
    op.create_index("ix_refresh_tokens_expires_at", "refresh_tokens", ["expires_at"])

    # ── 3. analyses table ──
    op.create_table(
        "analyses",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="QUEUED"),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("progress_pct", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("correlation_id", sa.String(length=36), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("source_type", sa.String(length=16), nullable=False, server_default="UPLOAD"),
        sa.Column("audio_sha256", sa.String(length=64), nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("sample_rate", sa.Integer(), nullable=False, server_default="16000"),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False),
        sa.Column("quality", sa.JSON(), nullable=False),
        sa.Column("acoustic_result", sa.JSON(), nullable=True),
        sa.Column("transcript_result", sa.JSON(), nullable=True),
        sa.Column("scam_result", sa.JSON(), nullable=True),
        sa.Column("fusion_result", sa.JSON(), nullable=True),
        sa.Column("explanation_result", sa.JSON(), nullable=True),
        sa.Column("risk_probability", sa.Float(), nullable=True),
        sa.Column("verdict", sa.String(length=16), nullable=True),
        sa.Column("acoustic_spoof_prob", sa.Float(), nullable=True),
        sa.Column("scam_probability", sa.Float(), nullable=True),
        sa.Column("detected_language", sa.String(length=8), nullable=True),
        sa.Column("is_borderline", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("model_versions", sa.JSON(), nullable=False),
        sa.Column("pipeline_version", sa.String(length=32), nullable=False, server_default="0.1.0"),
        sa.Column("error_code", sa.String(length=48), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("degraded_branches", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("total_duration_ms", sa.Integer(), nullable=True),
        sa.Column("stage_timings_ms", sa.JSON(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("label", sa.String(length=80), nullable=True),
    )
    op.create_index("ix_analyses_user_id", "analyses", ["user_id"])
    op.create_index("ix_analyses_audio_sha256", "analyses", ["audio_sha256"])
    op.create_index("ix_analyses_status", "analyses", ["status"])
    op.create_index("ix_analyses_created_at", "analyses", ["created_at"])
    op.create_index("ix_analyses_risk_probability", "analyses", ["risk_probability"])
    op.create_index("ix_analyses_verdict", "analyses", ["verdict"])
    op.create_index("ix_analyses_user_created", "analyses", ["user_id", "created_at"])
    op.create_index("ix_analyses_dedup", "analyses", ["audio_sha256", "user_id"])
    op.create_index("ix_analyses_status_created", "analyses", ["status", "created_at"])

    # ── 4. challenges table ──
    op.create_table(
        "challenges",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("analysis_id", sa.String(length=36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("challenge_type", sa.String(length=32), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("expected_phrase", sa.Text(), nullable=True),
        sa.Column("expected_duration_s", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False, server_default="ISSUED"),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("response_audio_sha256", sa.String(length=64), nullable=True),
        sa.Column("response_duration_s", sa.Float(), nullable=True),
        sa.Column("consistency_score", sa.Float(), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("feature_deltas", sa.JSON(), nullable=True),
        sa.Column("expected_deltas", sa.JSON(), nullable=True),
        sa.Column("notes", sa.JSON(), nullable=False),
    )
    op.create_index("ix_challenges_analysis_id", "challenges", ["analysis_id"])

    # ── 5. artifacts table ──
    op.create_table(
        "artifacts",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("analysis_id", sa.String(length=36), sa.ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_artifacts_analysis_id", "artifacts", ["analysis_id"])

    # ── 6. system_events table ──
    op.create_table(
        "system_events",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("analysis_id", sa.String(length=36), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_system_events_event_type", "system_events", ["event_type"])
    op.create_index("ix_system_events_created_at", "system_events", ["created_at"])


def downgrade() -> None:
    op.drop_table("system_events")
    op.drop_table("artifacts")
    op.drop_table("challenges")
    op.drop_table("analyses")
    op.drop_table("refresh_tokens")
    op.drop_table("users")
