"""Challenge routes for issuance, response verification, and re-fusion per 08 §4."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ai.audio.io import canonicalize, compute_sha256, sanitize_filename
from ai.challenge.catalog import issue_challenge
from ai.challenge.verifier import ChallengeVerifier
from ai.fusion.features import assemble_features
from ai.fusion.fuser import FusionEngine
from ai.linguistic.transcriber import Transcriber
from app.db.models import Analysis, Challenge, User
from app.deps import get_current_user_optional, get_db
from app.schemas.schemas_challenge import (
    ChallengeIssueResponse,
    ChallengeVerifyResponse,
)
from app.services.storage import get_storage

router = APIRouter(tags=["challenges"])


@router.post(
    "/analyses/{analysis_id}/challenges",
    response_model=ChallengeIssueResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Issue an interactive challenge for an analysis",
)
async def issue_new_challenge(
    analysis_id: str,
    challenge_type: str | None = Form(None),
    current_user: Annotated[User | None, Depends(get_current_user_optional)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
) -> ChallengeIssueResponse:
    stmt = select(Analysis).where(Analysis.id == analysis_id)
    analysis = (await db.execute(stmt)).scalar_one_or_none()

    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Analysis not found."},
        )

    # Ownership check
    if analysis.user_id is not None and (current_user is None or current_user.id != analysis.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "You do not have permission to challenge this analysis."},
        )

    # Issue challenge definition
    chal_obj = issue_challenge(challenge_type)
    now = datetime.now(UTC).replace(tzinfo=None)
    expires_at = now + timedelta(seconds=60)  # 60s expiry per 05 §4.5

    record = Challenge(
        analysis_id=analysis_id,
        challenge_type=chal_obj.challenge_type,
        prompt_text=chal_obj.prompt_text,
        expected_phrase=chal_obj.expected_phrase,
        expected_duration_s=chal_obj.expected_duration_s,
        status="ISSUED",
        issued_at=now,
        expires_at=expires_at,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    return ChallengeIssueResponse(
        id=record.id,
        analysis_id=record.analysis_id,
        challenge_type=record.challenge_type,
        prompt_text=record.prompt_text,
        expected_phrase=record.expected_phrase,
        expected_duration_s=record.expected_duration_s,
        instructions=chal_obj.instructions,
        issued_at=record.issued_at,
        expires_at=record.expires_at,
    )


@router.post(
    "/challenges/{challenge_id}/respond",
    response_model=ChallengeVerifyResponse,
    summary="Submit challenge audio, verify acoustic deltas, and re-fuse verdict",
)
async def respond_to_challenge(
    challenge_id: str,
    file: UploadFile = File(...),
    current_user: Annotated[User | None, Depends(get_current_user_optional)] = None,
    db: Annotated[AsyncSession, Depends(get_db)] = None,
) -> ChallengeVerifyResponse:
    stmt = (
        select(Challenge)
        .options(selectinload(Challenge.analysis))
        .where(Challenge.id == challenge_id)
    )
    challenge = (await db.execute(stmt)).scalar_one_or_none()

    if challenge is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Challenge not found."},
        )

    now = datetime.now(UTC).replace(tzinfo=None)

    # Anti-replay: single-use and expiration check per 05 §4.5
    if challenge.status != "ISSUED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "CHALLENGE_ALREADY_USED", "message": "Challenge has already been completed or expired."},
        )

    # Normalize expires_at to naive for comparison (SQLite strips tz)
    exp_at = challenge.expires_at
    if exp_at is not None and getattr(exp_at, 'tzinfo', None) is not None:
        exp_at = exp_at.replace(tzinfo=None)
    if exp_at is not None and now > exp_at:
        challenge.status = "EXPIRED"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "CHALLENGE_EXPIRED", "message": "Challenge has expired (60s limit)."},
        )

    # Ownership check
    analysis = challenge.analysis
    if analysis.user_id is not None and (current_user is None or current_user.id != analysis.user_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "FORBIDDEN", "message": "You do not have permission to respond to this challenge."},
        )

    # Retrieve baseline audio from storage
    storage = get_storage()
    baseline_bytes = storage.get(f"analyses/{analysis.id}/canonical.wav")

    temp_dir = Path(tempfile.mkdtemp(prefix="vg_chal_"))
    base_wav = temp_dir / "baseline.wav"
    base_wav.write_bytes(baseline_bytes)

    # Save response audio
    clean_fn = sanitize_filename(file.filename or "response.wav")
    raw_resp_path = temp_dir / f"raw_{clean_fn}"
    with open(raw_resp_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    canonical_resp_path = temp_dir / "response_canonical.wav"
    canonicalize(raw_resp_path, canonical_resp_path)

    # Transcribe response to check phrase compliance if required
    response_transcript_text = None
    if challenge.expected_phrase:
        transcriber = Transcriber()
        t_res = transcriber.transcribe(canonical_resp_path)
        response_transcript_text = t_res.text

    # Reconstruct Challenge domain object
    from ai.base import Challenge as DomainChallenge

    domain_chal = DomainChallenge(
        id=challenge.id,
        challenge_type=challenge.challenge_type,
        prompt_text=challenge.prompt_text,
        expected_phrase=challenge.expected_phrase,
        expected_duration_s=challenge.expected_duration_s,
        instructions=[],
    )

    verifier = ChallengeVerifier()
    c_res = verifier.verify(
        base_wav,
        canonical_resp_path,
        domain_chal,
        response_transcript=response_transcript_text,
    )

    # Update challenge record
    challenge.status = "COMPLETED" if c_res.passed else "FAILED"
    challenge.responded_at = now
    challenge.response_audio_sha256 = compute_sha256(canonical_resp_path)
    challenge.consistency_score = c_res.consistency_score
    challenge.passed = c_res.passed
    challenge.feature_deltas = c_res.feature_deltas
    challenge.notes = c_res.notes

    # ── Re-fusion: Update analysis verdict with challenge signal ──
    from ai.registry import get_registry
    from ai.base import FusionFeatures

    fuser = get_registry().get_fuser()
    stored_fv = (analysis.fusion_result or {}).get("feature_vector", {})
    if stored_fv and isinstance(stored_fv, dict):
        features = FusionFeatures(
            acoustic_available=float(stored_fv.get("acoustic_available", 0.0)),
            acoustic_spoof_prob=float(stored_fv.get("acoustic_spoof_prob", 0.5)),
            acoustic_uncertainty=float(stored_fv.get("acoustic_uncertainty", 1.0)),
            acoustic_window_std=float(stored_fv.get("acoustic_window_std", 0.0)),
            linguistic_available=float(stored_fv.get("linguistic_available", 0.0)),
            scam_prob=float(stored_fv.get("scam_prob", 0.5)),
            scam_max_category=float(stored_fv.get("scam_max_category", 0.0)),
            scam_n_categories=float(stored_fv.get("scam_n_categories", 0.0)),
            transcript_reliable=float(stored_fv.get("transcript_reliable", 0.0)),
            language_supported=float(stored_fv.get("language_supported", 0.0)),
            transcript_length_norm=float(stored_fv.get("transcript_length_norm", 0.0)),
            challenge_available=1.0,
            challenge_consistency=float(c_res.consistency_score),
            audio_quality_score=float(stored_fv.get("audio_quality_score", 0.8)),
        )
    else:
        features = assemble_features(challenge=c_res)
    # Perform re-fusion
    fusion_result = fuser.fuse(features)

    analysis.risk_probability = fusion_result.risk_probability
    analysis.verdict = fusion_result.verdict.value
    if analysis.fusion_result:
        updated_fusion = dict(analysis.fusion_result)
        updated_fusion["re_fused_with_challenge"] = True
        updated_fusion["challenge_consistency"] = c_res.consistency_score
        updated_fusion["risk_probability"] = fusion_result.risk_probability
        updated_fusion["verdict"] = fusion_result.verdict.value
        analysis.fusion_result = updated_fusion

    await db.commit()

    return ChallengeVerifyResponse(
        challenge_id=challenge.id,
        analysis_id=analysis.id,
        status=challenge.status,
        consistency_score=c_res.consistency_score,
        passed=c_res.passed,
        compliance=c_res.compliance,
        feature_deltas=c_res.feature_deltas,
        notes=c_res.notes,
        re_fused=True,
        new_verdict=analysis.verdict,
        new_risk_probability=analysis.risk_probability,
    )
