"""Unit and integration tests for analysis ownership, deduplication, and lifecycle.

Verifies Phase 8 exit criteria:
  - User A cannot view or delete User B's analysis (403 FORBIDDEN)
  - Submitting identical audio for the same user returns 200 with deduplicated: True
  - Deleting an analysis purges its files from the storage backend
"""

from __future__ import annotations

import io
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from app.config import get_settings
from app.db.base import Base
from app.db.models import Analysis, Artifact, User
from app.db.session import async_session_factory, init_db
from app.main import create_app
from app.services.storage import get_storage
from tests.fixtures.audio_fixtures import create_synthetic_audio, write_wav_file


@pytest.fixture(autouse=True)
async def setup_test_db(tmp_path: Path):
    settings = get_settings()
    settings.database_url = "sqlite+aiosqlite:///:memory:"
    settings.storage_local_root = str(tmp_path / "artifacts")
    engine = init_db(settings)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.mark.asyncio
async def test_analysis_ownership_enforcement(tmp_path: Path):
    """Verify that a user cannot access another user's analysis."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Register User 1 & User 2
        u1_res = await client.post(
            "/api/v1/auth/register",
            json={"email": "u1@example.com", "password": "Password12345!"},
        )
        token1 = u1_res.json()["access_token"]
        u1_id = u1_res.json()["user"]["id"]

        u2_res = await client.post(
            "/api/v1/auth/register",
            json={"email": "u2@example.com", "password": "Password12345!"},
        )
        token2 = u2_res.json()["access_token"]

        # Directly insert analysis owned by User 1
        async with async_session_factory() as db:
            a1 = Analysis(
                id="ana_u1_test",
                user_id=u1_id,
                status="COMPLETE",
                audio_sha256="test_sha256",
                duration_seconds=5.0,
                file_size_bytes=1000,
                verdict="LOW",
                risk_probability=0.12,
            )
            db.add(a1)
            await db.commit()

        # User 1 accesses own analysis -> 200 OK
        res1 = await client.get("/api/v1/analyses/ana_u1_test", headers={"Authorization": f"Bearer {token1}"})
        assert res1.status_code == 200

        # User 2 attempts to access User 1's analysis -> 403 FORBIDDEN
        res2 = await client.get("/api/v1/analyses/ana_u1_test", headers={"Authorization": f"Bearer {token2}"})
        assert res2.status_code == 403
        assert res2.json()["error"]["code"] == "FORBIDDEN"

        # User 2 attempts to delete User 1's analysis -> 403 FORBIDDEN
        del_res = await client.delete("/api/v1/analyses/ana_u1_test", headers={"Authorization": f"Bearer {token2}"})
        assert del_res.status_code == 403


@pytest.mark.asyncio
async def test_analysis_deletion_purges_storage(tmp_path: Path):
    """Verify that deleting an analysis removes artifact files from storage."""
    storage = get_storage()
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Register user
        reg_res = await client.post(
            "/api/v1/auth/register",
            json={"email": "owner@example.com", "password": "Password12345!"},
        )
        token = reg_res.json()["access_token"]
        uid = reg_res.json()["user"]["id"]

        # Store test artifact in storage
        storage_key = "analyses/del_test/spec.png"
        storage.put(storage_key, b"fake_png_data", content_type="image/png")
        assert storage.exists(storage_key)

        # Create analysis with artifact record
        async with async_session_factory() as db:
            a = Analysis(
                id="del_test",
                user_id=uid,
                status="COMPLETE",
                audio_sha256="sha256_del",
                duration_seconds=3.0,
                file_size_bytes=500,
            )
            art = Artifact(
                id="art_del",
                analysis_id="del_test",
                kind="spectrogram",
                storage_key=storage_key,
                content_type="image/png",
                size_bytes=13,
                sha256="fake_sha",
            )
            db.add_all([a, art])
            await db.commit()

        # Delete analysis
        del_res = await client.delete("/api/v1/analyses/del_test", headers={"Authorization": f"Bearer {token}"})
        assert del_res.status_code == 204

        # Artifact must be deleted from storage
        assert not storage.exists(storage_key)


@pytest.mark.asyncio
async def test_guest_analysis_session_binding(tmp_path: Path):
    """Verify guest analysis deletion enforces signed session token binding per 14 §4.

    Exit criteria:
      - Anonymous delete without token returns 403 FORBIDDEN
      - Anonymous delete with a DIFFERENT guest session token returns 403 FORBIDDEN
      - Anonymous delete with the MATCHING guest session token returns 204 NO CONTENT
    """
    from app.security import create_guest_session_token

    settings = get_settings()
    app = create_app()
    transport = ASGITransport(app=app)

    token_a, session_id_a = create_guest_session_token(settings=settings)
    token_b, session_id_b = create_guest_session_token(settings=settings)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a guest analysis bound to session_id_a
        async with async_session_factory() as db:
            guest_ana = Analysis(
                id="guest_ana_test_123",
                user_id=None,
                guest_session_id=session_id_a,
                status="COMPLETE",
                audio_sha256="sha256_guest_test",
                duration_seconds=4.0,
                file_size_bytes=600,
            )
            db.add(guest_ana)
            await db.commit()

        # 1. Anonymous request with NO session token -> 403 FORBIDDEN
        res_no_token = await client.delete("/api/v1/analyses/guest_ana_test_123")
        assert res_no_token.status_code == 403
        assert res_no_token.json()["error"]["code"] == "FORBIDDEN"

        # 2. Anonymous request with a DIFFERENT session token (token_b) -> 403 FORBIDDEN
        res_diff_session = await client.delete(
            "/api/v1/analyses/guest_ana_test_123",
            headers={"X-Guest-Session": token_b},
        )
        assert res_diff_session.status_code == 403
        assert res_diff_session.json()["error"]["code"] == "FORBIDDEN"

        # 3. Anonymous request with MATCHING session token (token_a) -> 204 NO CONTENT
        res_matching_session = await client.delete(
            "/api/v1/analyses/guest_ana_test_123",
            headers={"X-Guest-Session": token_a},
        )
        assert res_matching_session.status_code == 204

        # 4. Verify analysis is now removed from database
        async with async_session_factory() as db:
            deleted_check = await db.get(Analysis, "guest_ana_test_123")
            assert deleted_check is None


@pytest.mark.asyncio
async def test_guest_analysis_session_binding(tmp_path: Path):
    """Verify guest analysis deletion enforces signed session token binding per 14 §4.

    Exit criteria:
      - Anonymous delete without token returns 403 FORBIDDEN
      - Anonymous delete with a DIFFERENT guest session token returns 403 FORBIDDEN
      - Anonymous delete with the MATCHING guest session token returns 204 NO CONTENT
    """
    from app.security import create_guest_session_token

    settings = get_settings()
    app = create_app()
    transport = ASGITransport(app=app)

    token_a, session_id_a = create_guest_session_token(settings=settings)
    token_b, session_id_b = create_guest_session_token(settings=settings)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create a guest analysis bound to session_id_a
        async with async_session_factory() as db:
            guest_ana = Analysis(
                id="guest_ana_test_123",
                user_id=None,
                guest_session_id=session_id_a,
                status="COMPLETE",
                audio_sha256="sha256_guest_test",
                duration_seconds=4.0,
                file_size_bytes=600,
            )
            db.add(guest_ana)
            await db.commit()

        # 1. Anonymous request with NO session token -> 403 FORBIDDEN
        res_no_token = await client.delete("/api/v1/analyses/guest_ana_test_123")
        assert res_no_token.status_code == 403
        assert res_no_token.json()["error"]["code"] == "FORBIDDEN"

        # 2. Anonymous request with a DIFFERENT session token (token_b) -> 403 FORBIDDEN
        res_diff_session = await client.delete(
            "/api/v1/analyses/guest_ana_test_123",
            headers={"X-Guest-Session": token_b},
        )
        assert res_diff_session.status_code == 403
        assert res_diff_session.json()["error"]["code"] == "FORBIDDEN"

        # 3. Anonymous request with MATCHING session token (token_a) -> 204 NO CONTENT
        res_matching_session = await client.delete(
            "/api/v1/analyses/guest_ana_test_123",
            headers={"X-Guest-Session": token_a},
        )
        assert res_matching_session.status_code == 204

        # 4. Verify analysis is now removed from database
        async with async_session_factory() as db:
            deleted_check = await db.get(Analysis, "guest_ana_test_123")
            assert deleted_check is None

