"""Pydantic schemas for authentication and user management per 08 §2."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=10, description="Password must be at least 10 characters")
    display_name: str | None = Field(None, max_length=80)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None = None
    created_at: datetime
    last_login_at: datetime | None = None
    analysis_count: int = 0
    preferences: dict[str, Any] = Field(default_factory=dict)


class TokenResponse(BaseModel):
    user: UserResponse
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 900  # 15 minutes in seconds


class UpdateUserRequest(BaseModel):
    display_name: str | None = Field(None, max_length=80)
    preferences: dict[str, Any] | None = None


class DeleteAccountRequest(BaseModel):
    password: str
    confirm: str = Field(..., description="Must equal 'DELETE'")
