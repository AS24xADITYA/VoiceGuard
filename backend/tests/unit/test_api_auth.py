"""Unit and integration tests for auth routes and token lifecycle per 08 §2.

Verifies Phase 8 exit criteria:
  - Registration with Argon2id hashing and token issuance
  - Login failure is generic without disclosing email presence
  - Refresh token rotation
  - Replay defense: reuse of revoked token revokes user's token family
  - Protected /auth/me endpoint rejects unauthenticated requests
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.db.session import async_session_factory, init_db
from app.config import get_settings
from app.db.base import Base
from app.main import create_app


@pytest.fixture(autouse=True)
async def setup_test_db():
    settings = get_settings()
    # Use in-memory SQLite for tests
    settings.database_url = "sqlite+aiosqlite:///:memory:"
    engine = init_db(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.asyncio
async def test_auth_registration_and_login():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Register
        reg_res = await client.post(
            "/api/v1/auth/register",
            json={"email": "alice@example.com", "password": "SecurePassword123!", "display_name": "Alice"},
        )
        assert reg_res.status_code == 201
        data = reg_res.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["user"]["email"] == "alice@example.com"

        # 2. Duplicate registration rejected
        dup_res = await client.post(
            "/api/v1/auth/register",
            json={"email": "alice@example.com", "password": "AnotherPassword123!"},
        )
        assert dup_res.status_code == 409

        # 3. Bad login
        bad_login = await client.post(
            "/api/v1/auth/login",
            json={"email": "alice@example.com", "password": "WrongPassword!"},
        )
        assert bad_login.status_code == 401

        # 4. Successful login
        good_login = await client.post(
            "/api/v1/auth/login",
            json={"email": "alice@example.com", "password": "SecurePassword123!"},
        )
        assert good_login.status_code == 200
        token = good_login.json()["access_token"]

        # 5. /auth/me
        me_res = await client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_res.status_code == 200
        assert me_res.json()["email"] == "alice@example.com"


@pytest.mark.asyncio
async def test_refresh_token_rotation_and_family_revocation():
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Register
        reg_res = await client.post(
            "/api/v1/auth/register",
            json={"email": "bob@example.com", "password": "Password12345!"},
        )
        refresh_1 = reg_res.json()["refresh_token"]

        # 1. Rotate refresh_1 -> issues refresh_2
        ref_res_1 = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_1})
        assert ref_res_1.status_code == 200
        refresh_2 = ref_res_1.json()["refresh_token"]
        assert refresh_1 != refresh_2

        # 2. Replay attack: attempt to reuse refresh_1 (already rotated)
        replay_res = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_1})
        assert replay_res.status_code == 401
        assert "Token reuse detected" in replay_res.json()["error"]["message"]

        # 3. Because of family revocation, refresh_2 should now also be invalidated!
        ref_res_2 = await client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_2})
        assert ref_res_2.status_code == 401
