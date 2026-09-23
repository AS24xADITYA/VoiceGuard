"""Pydantic schemas for analysis requests, progress polling, and full results per 08 §3."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalysisCreateResponse(BaseModel):
    """Returned immediately on upload (202 Accepted)."""

    id: str
    status: str = "QUEUED"
    stage: str | None = None
    progress_pct: int = 0
    created_at: datetime
    poll_url: str
    deduplicated: bool = False
    guest_session_token: str | None = None


class AnalysisPollResponse(BaseModel):
    """Returned while analysis is QUEUED or RUNNING."""

    id: str
    status: str
    stage: str | None = None
    progress_pct: int = 0
    stage_label: str | None = None
    elapsed_ms: int | None = None
    estimated_remaining_ms: int | None = None


class AnalysisSourceMeta(BaseModel):
    filename: str | None = None
    source_type: str = "UPLOAD"
    duration_seconds: float
    sample_rate: int = 16000
    file_size_bytes: int


class AnalysisArtifactItem(BaseModel):
    id: str
    kind: str
    download_url: str
    content_type: str
    size_bytes: int
    sha256: str


class AnalysisResponse(BaseModel):
    """Full analysis result returned when status is COMPLETE."""

    id: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    total_duration_ms: int | None = None
    source: AnalysisSourceMeta
    quality: dict[str, Any] = Field(default_factory=dict)
    acoustic: dict[str, Any] | None = None
    transcript: dict[str, Any] | None = None
    scam: dict[str, Any] | None = None
    fusion: dict[str, Any] | None = None
    explanation: dict[str, Any] | None = None
    challenges: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[AnalysisArtifactItem] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    degraded_branches: list[str] = Field(default_factory=list)
    deduplicated: bool = False


class AnalysisSummaryItem(BaseModel):
    id: str
    status: str
    created_at: datetime
    original_filename: str | None = None
    duration_seconds: float
    verdict: str | None = None
    risk_probability: float | None = None
    detected_language: str | None = None
    is_borderline: bool = False


class AnalysisListResponse(BaseModel):
    items: list[AnalysisSummaryItem]
    total: int
    page: int
    page_size: int
