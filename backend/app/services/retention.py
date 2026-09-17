"""Data retention and artifact purge background service.

Per 14 §3 and 08 §1:
  - Purges analyses past their expires_at retention window
  - Deletes all associated files from storage backend
  - Purges expired and revoked refresh tokens older than retention limit
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.db.models import Analysis, Artifact, RefreshToken
from app.db.session import async_session_factory
from app.services.storage import get_storage

log = structlog.get_logger()


async def run_retention_purge() -> int:
    """Run data retention purge. Returns count of purged analyses."""
    settings = get_settings()
    storage = get_storage(settings)
    now = datetime.now(UTC)

    purged_count = 0

    async with async_session_factory() as db:
        # Find expired analyses
        stmt = (
            select(Analysis)
            .options(selectinload(Analysis.artifacts))
            .where(Analysis.expires_at != None, Analysis.expires_at < now)
        )
        expired_analyses = (await db.execute(stmt)).scalars().all()

        for a in expired_analyses:
            for art in a.artifacts:
                try:
                    storage.delete(art.storage_key)
                except Exception as e:
                    log.warning("retention_artifact_delete_failed", key=art.storage_key, error=str(e))

            await db.delete(a)
            purged_count += 1

        # Purge stale refresh tokens older than retention days
        cutoff = now - timedelta(days=settings.retention_days)
        token_stmt = delete(RefreshToken).where(
            (RefreshToken.revoked_at != None) & (RefreshToken.revoked_at < cutoff)
            | (RefreshToken.expires_at < cutoff)
        )
        await db.execute(token_stmt)
        await db.commit()

    log.info("retention_purge_completed", purged_analyses=purged_count)
    return purged_count
