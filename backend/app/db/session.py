"""Async database session management."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_db(settings: Settings):
    """Initialize the async engine and session factory."""
    global _engine, _session_factory

    connect_args = {}
    pool_kwargs: dict = {}

    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        path_part = settings.database_url.split(":///")[-1]
        if path_part and not path_part.startswith(":memory:"):
            from pathlib import Path
            Path(path_part).parent.mkdir(parents=True, exist_ok=True)
        elif path_part.startswith(":memory:"):
            from sqlalchemy.pool import StaticPool
            pool_kwargs["poolclass"] = StaticPool
    else:
        # PostgreSQL — small pool for free-tier connection limits
        pool_kwargs.update(
            pool_size=3,
            max_overflow=2,
            pool_pre_ping=True,
            pool_recycle=300,
        )

    _engine = create_async_engine(
        settings.database_url,
        echo=False,
        connect_args=connect_args,
        **pool_kwargs,
    )
    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    return _engine


async def close_db() -> None:
    """Dispose the engine on shutdown."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency: yields an async session, rolls back on error."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

def async_session_factory() -> AsyncSession:
    """Return a new async session from the configured session maker."""
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _session_factory()


get_session = get_db_session
