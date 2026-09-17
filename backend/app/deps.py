"""FastAPI dependency injection utilities.

Per 04 §5 and 08 §1:
  - get_db: async database session dependency
  - get_current_user: authenticated user dependency (raises 401 AUTH_ERROR on invalid/missing token)
  - get_current_user_optional: allows guest / optional auth on endpoints like POST /analyses
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User
from app.db.session import get_session
from app.security import decode_token

security = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency yielding an async SQLAlchemy session."""
    async for session in get_session():
        yield session


async def get_current_user_optional(
    auth_header: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User | None:
    """Extract and validate the current user from the Authorization header if present."""
    if not auth_header or not auth_header.credentials:
        return None

    token = auth_header.credentials
    payload = decode_token(token)
    if payload is None or payload.get("type") != "access":
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    stmt = select(User).where(User.id == user_id, User.is_active == True)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    return user


async def get_current_user(
    user: Annotated[User | None, Depends(get_current_user_optional)],
) -> User:
    """Require an authenticated user. Raises HTTP 401 if unauthenticated."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_ERROR", "message": "Authentication required. Please provide a valid token."},
        )
    return user
