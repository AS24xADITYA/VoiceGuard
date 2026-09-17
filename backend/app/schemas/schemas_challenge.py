"""Pydantic schemas for challenge issuance and response verification per 08 §4."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ChallengeIssueResponse(BaseModel):
    id: str
    analysis_id: str
    challenge_type: str
    prompt_text: str
    expected_phrase: str | None = None
    expected_duration_s: float
    instructions: list[str] = Field(default_factory=list)
    issued_at: datetime
    expires_at: datetime


class ChallengeVerifyResponse(BaseModel):
    challenge_id: str
    analysis_id: str
    status: str
    consistency_score: float
    passed: bool
    compliance: dict[str, Any] = Field(default_factory=dict)
    feature_deltas: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    re_fused: bool = False
    new_verdict: str | None = None
    new_risk_probability: float | None = None
