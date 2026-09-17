"""Unit tests verifying OpenAPI schema generation and endpoint completeness per 08.

Verifies Phase 8 exit criteria:
  - OpenAPI schema generates without warnings or errors
  - Every documented endpoint path exists in the OpenAPI schema
"""

from __future__ import annotations

import pytest
from app.main import create_app


def test_openapi_schema_generation():
    """Verify FastAPI application successfully builds OpenAPI specification."""
    app = create_app()
    schema = app.openapi()

    assert schema is not None
    assert schema["info"]["title"] == "VoiceGuard API"
    assert "paths" in schema

    paths = schema["paths"]

    # Verify all documented endpoints are present
    expected_endpoints = [
        "/api/v1/system/health",
        "/api/v1/system/config",
        "/api/v1/system/metrics",
        "/api/v1/auth/register",
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v1/auth/logout",
        "/api/v1/auth/me",
        "/api/v1/analyses",
        "/api/v1/analyses/{analysis_id}",
        "/api/v1/analyses/{analysis_id}/challenges",
        "/api/v1/challenges/{challenge_id}/respond",
        "/api/v1/artifacts/{artifact_key}",
    ]

    for ep in expected_endpoints:
        assert ep in paths, f"Endpoint {ep} missing from OpenAPI schema!"
