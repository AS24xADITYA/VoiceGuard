"""Authentication routes implementing registration, login, rotating refresh, and user lifecycle.

Per 08 §2:
  - Argon2id password hashing
  - JWT tokens (15m access, 14d refresh)
  - Rotating refresh tokens: old token revoked on refresh; reuse of revoked token revokes family
  - Generic 401 on login failure (never discloses email presence)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RefreshToken, User
from app.deps import get_current_user, get_db
from app.schemas.schemas_auth import (
    DeleteAccountRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UpdateUserRequest,
    UserResponse,
)
from app.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    hash_token,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new account",
)
async def register(
    body: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    # Check if email is already registered
    stmt = select(User).where(User.email == body.email.lower())
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "EMAIL_EXISTS", "message": "An account with this email address already exists."},
        )

    # Hash password with Argon2id
    hashed = hash_password(body.password)
    user = User(
        email=body.email.lower(),
        password_hash=hashed,
        display_name=body.display_name,
    )
    db.add(user)
    await db.flush()

    # Issue tokens
    access_token = create_access_token({"sub": user.id, "email": user.email})
    refresh_token, token_hash, expires_at = create_refresh_token()

    rt_record = RefreshToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
    )
    db.add(rt_record)
    await db.commit()
    await db.refresh(user)

    user_resp = UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
        analysis_count=user.analysis_count,
        preferences=user.preferences or {},
    )

    return TokenResponse(
        user=user_resp,
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=900,
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate user and issue tokens",
)
async def login(
    body: LoginRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    stmt = select(User).where(User.email == body.email.lower(), User.is_active == True)
    user = (await db.execute(stmt)).scalar_one_or_none()

    # Generic failure to avoid user enumeration per 08 §2
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_ERROR", "message": "Invalid email or password."},
        )

    # Update last login
    user.last_login_at = datetime.now(UTC)

    access_token = create_access_token({"sub": user.id, "email": user.email})
    refresh_token, token_hash, expires_at = create_refresh_token()

    rt_record = RefreshToken(
        user_id=user.id,
        token_hash=token_hash,
        expires_at=expires_at,
        user_agent=request.headers.get("user-agent", "")[:255],
    )
    db.add(rt_record)
    await db.commit()
    await db.refresh(user)

    user_resp = UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
        analysis_count=user.analysis_count,
        preferences=user.preferences or {},
    )

    return TokenResponse(
        user=user_resp,
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=900,
    )


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Rotate refresh token and issue new token pair",
)
async def refresh(
    body: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TokenResponse:
    t_hash = hash_token(body.refresh_token)

    stmt = select(RefreshToken).where(RefreshToken.token_hash == t_hash)
    rt = (await db.execute(stmt)).scalar_one_or_none()

    now = datetime.now(UTC)

    # Replay detection: If token was already revoked, revoke the entire family for this user!
    if rt is not None and rt.revoked_at is not None:
        log_user_id = rt.user_id
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == log_user_id)
            .values(revoked_at=now)
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "AUTH_ERROR",
                "message": "Token reuse detected. All sessions for this user have been revoked.",
            },
        )

    if rt is None or rt.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_ERROR", "message": "Invalid or expired refresh token."},
        )

    # Revoke current token
    rt.revoked_at = now

    # Fetch user
    user = (await db.execute(select(User).where(User.id == rt.user_id))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "AUTH_ERROR", "message": "User account inactive."},
        )

    # Issue new pair
    access_token = create_access_token({"sub": user.id, "email": user.email})
    new_refresh, new_hash, expires_at = create_refresh_token()

    new_rt = RefreshToken(
        user_id=user.id,
        token_hash=new_hash,
        expires_at=expires_at,
    )
    db.add(new_rt)
    await db.commit()

    user_resp = UserResponse(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
        analysis_count=user.analysis_count,
        preferences=user.preferences or {},
    )

    return TokenResponse(
        user=user_resp,
        access_token=access_token,
        refresh_token=new_refresh,
        token_type="bearer",
        expires_in=900,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Revoke refresh token")
async def logout(
    body: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    t_hash = hash_token(body.refresh_token)
    stmt = update(RefreshToken).where(RefreshToken.token_hash == t_hash).values(revoked_at=datetime.now(UTC))
    await db.execute(stmt)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserResponse, summary="Get current user details")
async def get_me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserResponse:
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        created_at=current_user.created_at,
        last_login_at=current_user.last_login_at,
        analysis_count=current_user.analysis_count,
        preferences=current_user.preferences or {},
    )


@router.patch("/me", response_model=UserResponse, summary="Update profile preferences")
async def update_me(
    body: UpdateUserRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserResponse:
    if body.display_name is not None:
        current_user.display_name = body.display_name
    if body.preferences is not None:
        merged = dict(current_user.preferences or {})
        merged.update(body.preferences)
        current_user.preferences = merged

    await db.commit()
    await db.refresh(current_user)

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        created_at=current_user.created_at,
        last_login_at=current_user.last_login_at,
        analysis_count=current_user.analysis_count,
        preferences=current_user.preferences or {},
    )


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT, summary="Irreversibly delete account")
async def delete_me(
    body: DeleteAccountRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    if body.confirm != "DELETE" or not verify_password(body.password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "VALIDATION_ERROR", "message": "Confirmation text must be 'DELETE' and password correct."},
        )

    await db.delete(current_user)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
