"""Artifact streaming and signed download routes per 08 §5."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Artifact, User
from app.deps import get_current_user_optional, get_db
from app.services.storage import get_storage

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get(
    "/{artifact_key:path}",
    summary="Download or stream an analysis artifact",
)
async def get_artifact(
    artifact_key: str,
    current_user: Annotated[User | None, Depends(get_current_user_optional)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    storage = get_storage()

    # Look up artifact record by storage_key or artifact id to verify ownership
    stmt = (
        select(Artifact)
        .options(selectinload(Artifact.analysis))
        .where((Artifact.storage_key == artifact_key) | (Artifact.id == artifact_key))
    )
    artifact = (await db.execute(stmt)).scalar_one_or_none()

    if artifact is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Artifact not found."},
        )

    # Ownership enforcement: if analysis belongs to an authenticated user, require ownership
    if artifact.analysis is not None:
        analysis = artifact.analysis
        if analysis.user_id is not None and (current_user is None or current_user.id != analysis.user_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "FORBIDDEN", "message": "You do not have permission to access this artifact."},
            )

    try:
        data = storage.get(artifact.storage_key)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Artifact file not found in storage."},
        )

    content_type = artifact.content_type or "application/octet-stream"
    if not content_type or content_type == "application/octet-stream":
        if artifact.storage_key.endswith(".png"):
            content_type = "image/png"
        elif artifact.storage_key.endswith(".wav"):
            content_type = "audio/wav"
        elif artifact.storage_key.endswith(".json"):
            content_type = "application/json"

    return Response(
        content=data,
        media_type=content_type,
        headers={"Cache-Control": "private, max-age=3600"},
    )
