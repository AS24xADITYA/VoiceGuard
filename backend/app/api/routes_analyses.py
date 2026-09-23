"""Analysis routes for audio upload, polling, result retrieval, and history management.

Per 08 §3:
  - POST /analyses: upload audio (multipart/form-data), background execution, deduplication
  - GET /analyses/{id}: ownership verification, poll envelope when running, full response when complete
  - GET /analyses: user history with pagination
  - DELETE /analyses/{id}: ownership check, cascades artifact deletion from storage
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ai.audio.io import (
    AudioValidationError,
    compute_sha256,
    probe_duration,
    sanitize_filename,
    validate_duration,
    validate_file_size,
    validate_magic_bytes,
)
from app.config import get_settings
from app.db.models import Analysis, Artifact, Challenge, User
from app.deps import get_current_user, get_current_user_optional, get_db
from app.security import create_guest_session_token, verify_guest_session_token
from app.schemas.schemas_analysis import (
    AnalysisArtifactItem,
    AnalysisCreateResponse,
    AnalysisListResponse,
    AnalysisPollResponse,
    AnalysisResponse,
    AnalysisSourceMeta,
    AnalysisSummaryItem,
)
from app.services.analysis_service import process_analysis_background
from app.services.storage import get_storage

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post(
    "",
    response_model=AnalysisCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit audio for deepfake and scam analysis",
)
async def create_analysis(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    source_type: str = Form("UPLOAD"),
    auto_challenge: bool = Form(True),
    label: str | None = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Annotated[User | None, Depends(get_current_user_optional)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
) -> Any:
    settings = get_settings()

    # Save uploaded file to temp path for validation & hashing
    clean_filename = sanitize_filename(file.filename or "upload.wav")
    suffix = Path(clean_filename).suffix or ".wav"

    temp_dir = Path(tempfile.mkdtemp(prefix="vg_upload_"))
    temp_file = temp_dir / clean_filename

    with open(temp_file, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    # 1. Validation: magic bytes
    try:
        detected_fmt = validate_magic_bytes(temp_file)
    except AudioValidationError as e:
        temp_file.unlink(missing_ok=True)
        temp_dir.rmdir()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": e.code, "message": e.message},
        )

    # 2. Validation: file size
    try:
        validate_file_size(temp_file, max_mb=settings.max_upload_mb)
    except AudioValidationError as e:
        temp_file.unlink(missing_ok=True)
        temp_dir.rmdir()
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"code": e.code, "message": e.message},
        )

    # 3. Validation: duration
    try:
        duration_s = probe_duration(temp_file)
        validate_duration(
            duration_s,
            min_seconds=settings.min_duration_seconds,
            max_seconds=settings.max_duration_seconds,
        )
    except AudioValidationError as e:
        temp_file.unlink(missing_ok=True)
        temp_dir.rmdir()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": e.code, "message": e.message},
        )

    # 4. Compute SHA-256 for deduplication
    sha256 = compute_sha256(temp_file)
    user_id = current_user.id if current_user else None

    # Guest session handling per 08 §3 & 14 §4
    guest_session_token = None
    guest_session_id = None
    if user_id is None:
        presented_guest_token = (
            request.headers.get("X-Guest-Session")
            or request.headers.get("X-Session-ID")
            or request.cookies.get("vg_guest_session")
        )
        validated_id = verify_guest_session_token(presented_guest_token, settings) if presented_guest_token else None
        if validated_id:
            guest_session_id = validated_id
            guest_session_token = presented_guest_token
        else:
            guest_session_token, guest_session_id = create_guest_session_token(settings=settings)

        response.headers["X-Guest-Session"] = guest_session_token

    # 5. Deduplication check per 08 §3.1
    if user_id:
        stmt = (
            select(Analysis)
            .where(
                Analysis.audio_sha256 == sha256,
                Analysis.user_id == user_id,
                Analysis.status == "COMPLETE",
            )
            .order_by(desc(Analysis.created_at))
        )
        existing = (await db.execute(stmt)).scalars().first()
        if existing:
            temp_file.unlink(missing_ok=True)
            temp_dir.rmdir()
            return AnalysisCreateResponse(
                id=existing.id,
                status=existing.status,
                stage=existing.stage,
                progress_pct=100,
                created_at=existing.created_at,
                poll_url=f"/api/v1/analyses/{existing.id}",
                deduplicated=True,
                guest_session_token=guest_session_token,
            )

    # 6. Create initial database row
    analysis = Analysis(
        user_id=user_id,
        guest_session_id=guest_session_id,
        status="QUEUED",
        stage="INGESTING",
        progress_pct=0,
        original_filename=clean_filename,
        source_type=source_type,
        audio_sha256=sha256,
        duration_seconds=duration_s,
        sample_rate=16000,
        file_size_bytes=temp_file.stat().st_size,
        label=label[:80] if label else None,
    )
    db.add(analysis)

    if current_user:
        current_user.analysis_count += 1

    await db.commit()
    await db.refresh(analysis)

    # 7. Dispatch background execution
    background_tasks.add_task(
        process_analysis_background,
        analysis_id=analysis.id,
        temp_audio_path=temp_file,
        auto_challenge=auto_challenge,
    )

    return AnalysisCreateResponse(
        id=analysis.id,
        status="QUEUED",
        stage="INGESTING",
        progress_pct=0,
        created_at=analysis.created_at,
        poll_url=f"/api/v1/analyses/{analysis.id}",
        deduplicated=False,
        guest_session_token=guest_session_token,
    )


@router.get(
    "/{analysis_id}",
    summary="Get analysis result or polling status",
)
async def get_analysis(
    analysis_id: str,
    current_user: Annotated[User | None, Depends(get_current_user_optional)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Any:
    stmt = (
        select(Analysis)
        .options(selectinload(Analysis.artifacts), selectinload(Analysis.challenges))
        .where(Analysis.id == analysis_id)
    )
    analysis = (await db.execute(stmt)).scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Analysis not found."},
        )

    # Ownership enforcement: If analysis is owned by an authenticated user, require ownership
    if analysis.user_id is not None:
        if current_user is None or current_user.id != analysis.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "FORBIDDEN", "message": "You do not have permission to view this analysis."},
            )

    # If running or queued, return polling envelope per 08 §3.2
    if analysis.status in ("QUEUED", "RUNNING"):
        return AnalysisPollResponse(
            id=analysis.id,
            status=analysis.status,
            stage=analysis.stage,
            progress_pct=analysis.progress_pct,
            stage_label=f"Stage: {analysis.stage or 'Queued'}",
            elapsed_ms=analysis.total_duration_ms,
        )

    # If failed, return a poll-like envelope with the error details
    if analysis.status == "FAILED":
        return {
            "id": analysis.id,
            "status": "FAILED",
            "stage": analysis.stage,
            "progress_pct": analysis.progress_pct,
            "error_code": analysis.error_code,
            "error_message": analysis.error_message,
        }

    # Full complete analysis response
    storage = get_storage()
    artifact_items = [
        AnalysisArtifactItem(
            id=a.id,
            kind=a.kind,
            download_url=storage.get_url(a.storage_key),
            content_type=a.content_type,
            size_bytes=a.size_bytes,
            sha256=a.sha256,
        )
        for a in analysis.artifacts
    ]

    challenges_meta = [
        {
            "id": c.id,
            "type": c.challenge_type,
            "prompt": c.prompt_text,
            "expected_phrase": c.expected_phrase,
            "status": c.status,
            "consistency_score": c.consistency_score,
            "passed": c.passed,
            "notes": c.notes,
        }
        for c in analysis.challenges
    ]

    return AnalysisResponse(
        id=analysis.id,
        status=analysis.status,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at,
        total_duration_ms=analysis.total_duration_ms,
        source=AnalysisSourceMeta(
            filename=analysis.original_filename,
            source_type=analysis.source_type,
            duration_seconds=analysis.duration_seconds,
            sample_rate=analysis.sample_rate,
            file_size_bytes=analysis.file_size_bytes,
        ),
        quality=analysis.quality or {},
        acoustic=analysis.acoustic_result,
        transcript=analysis.transcript_result,
        scam=analysis.scam_result,
        fusion=analysis.fusion_result,
        explanation=analysis.explanation_result,
        challenges=challenges_meta,
        artifacts=artifact_items,
        model_versions=analysis.model_versions or {},
        degraded_branches=analysis.degraded_branches or [],
    )


@router.get(
    "",
    response_model=AnalysisListResponse,
    summary="List historical analyses for the current user",
)
async def list_analyses(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> AnalysisListResponse:
    # Total count query
    count_stmt = select(func.count(Analysis.id)).where(Analysis.user_id == current_user.id)
    total = (await db.execute(count_stmt)).scalar() or 0

    # Paged query
    stmt = (
        select(Analysis)
        .where(Analysis.user_id == current_user.id)
        .order_by(desc(Analysis.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    items = (await db.execute(stmt)).scalars().all()

    summary_items = [
        AnalysisSummaryItem(
            id=a.id,
            status=a.status,
            created_at=a.created_at,
            original_filename=a.original_filename,
            duration_seconds=a.duration_seconds,
            verdict=a.verdict,
            risk_probability=a.risk_probability,
            detected_language=a.detected_language,
            is_borderline=a.is_borderline,
        )
        for a in items
    ]

    return AnalysisListResponse(
        items=summary_items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.delete(
    "/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an analysis and purge stored artifacts",
)
async def delete_analysis(
    analysis_id: str,
    request: Request,
    current_user: Annotated[User | None, Depends(get_current_user_optional)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
) -> Response:
    settings = get_settings()
    stmt = (
        select(Analysis)
        .options(selectinload(Analysis.artifacts))
        .where(Analysis.id == analysis_id)
    )
    analysis = (await db.execute(stmt)).scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Analysis not found."},
        )

    # Ownership check: registered analyses require user match; guest analyses require signed session binding per 14 §4
    if analysis.user_id is not None:
        if current_user is None or analysis.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "FORBIDDEN", "message": "You do not own this analysis."},
            )
    else:
        # Anonymous / Guest analysis: must present valid signed guest token matching the bound session
        presented_token = (
            request.headers.get("X-Guest-Session")
            or request.headers.get("X-Session-ID")
            or request.cookies.get("vg_guest_session")
        )
        if not presented_token:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "FORBIDDEN", "message": "Missing guest session token for anonymous analysis."},
            )
        verified_session_id = verify_guest_session_token(presented_token, settings)
        if not verified_session_id or (analysis.guest_session_id and verified_session_id != analysis.guest_session_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"code": "FORBIDDEN", "message": "Invalid or mismatched guest session token."},
            )

    # Remove artifacts from storage backend
    storage = get_storage()
    for art in analysis.artifacts:
        try:
            storage.delete(art.storage_key)
        except Exception:
            pass

    # Clean up local analysis directory if local storage backend
    settings = get_settings()
    if settings.storage_backend == "local":
        import shutil
        analysis_dir = Path(settings.storage_local_root) / "analyses" / str(analysis.id)
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir, ignore_errors=True)

    await db.delete(analysis)
    await db.commit()

    return Response(status_code=status.HTTP_204_NO_CONTENT)
