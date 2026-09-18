"""Analysis service managing ingestion, deduplication, background execution, and persistence.

Per 08 §3:
  - Deduplication: returns existing analysis if same audio SHA-256 for user exists
  - Background async execution with stage progress updates
  - Automatic challenge issuance when borderline and auto_challenge=True
  - Artifact storage via StorageBackend (spectrograms, heatmaps, canonical audio)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.audio.io import compute_sha256, sanitize_filename
from ai.base import PipelineResult, QualityReport
from ai.challenge.catalog import issue_challenge
from ai.pipeline import PipelineContext, run_pipeline
from app.config import get_settings
from app.db.models import Analysis, Artifact, Challenge
from app.db.session import async_session_factory
from app.services.storage import get_storage

log = structlog.get_logger()

STAGE_PROGRESS_MAP: dict[str, int] = {
    "INGESTING": 15,
    "ANALYZING_ACOUSTIC": 45,
    "SCORING_INTENT": 65,
    "FUSING": 80,
    "EXPLAINING": 90,
    "COMPLETE": 100,
}


async def process_analysis_background(
    analysis_id: str,
    temp_audio_path: Path,
    auto_challenge: bool = True,
) -> None:
    """Background task executing the complete AI pipeline and persisting results."""
    settings = get_settings()
    storage = get_storage(settings)

    async with async_session_factory() as db:
        stmt = select(Analysis).where(Analysis.id == analysis_id)
        analysis = (await db.execute(stmt)).scalar_one_or_none()
        if analysis is None:
            return

        analysis.status = "RUNNING"
        analysis.started_at = datetime.now(UTC)
        await db.commit()

        # Define stage update callback — uses a SEPARATE session to avoid
        # concurrent commits on the main processing session (fixes "transaction is closed")
        async def update_stage(stage_name: str) -> None:
            try:
                async with async_session_factory() as stage_db:
                    stage_stmt = select(Analysis).where(Analysis.id == analysis_id)
                    stage_analysis = (await stage_db.execute(stage_stmt)).scalar_one_or_none()
                    if stage_analysis:
                        progress = STAGE_PROGRESS_MAP.get(stage_name, 0)
                        stage_analysis.stage = stage_name
                        stage_analysis.progress_pct = progress
                        await stage_db.commit()
            except Exception:
                pass  # Stage updates are non-critical, don't crash the pipeline

        loop = asyncio.get_running_loop()

        def sync_stage_callback(s: str) -> None:
            asyncio.run_coroutine_threadsafe(update_stage(s), loop)

        ctx = PipelineContext(
            stage_callback=sync_stage_callback,
            output_dir=temp_audio_path.parent,
        )

        try:
            result: PipelineResult = await run_pipeline(temp_audio_path, context=ctx)

            # Refresh the analysis from DB since stage callbacks used separate sessions
            await db.refresh(analysis)

            # Store artifacts into storage backend
            canonical_path = result.audio_meta.canonical_path
            if canonical_path.is_file():
                key = f"analyses/{analysis_id}/canonical.wav"
                storage.put(key, canonical_path.read_bytes(), content_type="audio/wav")
                art_audio = Artifact(
                    analysis_id=analysis_id,
                    kind="canonical_audio",
                    storage_key=key,
                    content_type="audio/wav",
                    size_bytes=canonical_path.stat().st_size,
                    sha256=result.audio_meta.sha256,
                )
                db.add(art_audio)

            # Store spectrogram if present
            if result.acoustic and result.acoustic.spectrogram_path.is_file():
                spec_path = result.acoustic.spectrogram_path
                key = f"analyses/{analysis_id}/spectrogram.png"
                storage.put(key, spec_path.read_bytes(), content_type="image/png")
                art_spec = Artifact(
                    analysis_id=analysis_id,
                    kind="spectrogram",
                    storage_key=key,
                    content_type="image/png",
                    size_bytes=spec_path.stat().st_size,
                    sha256=compute_sha256(spec_path),
                )
                db.add(art_spec)

            # Store heatmap and overlay if explanation present
            if result.explanation and result.explanation.windows:
                for win in result.explanation.windows:
                    w_idx = win["window_index"]
                    o_path = Path(win["overlay_path"])
                    if o_path.is_file():
                        key = f"analyses/{analysis_id}/overlay_w{w_idx}.png"
                        storage.put(key, o_path.read_bytes(), content_type="image/png")
                        art_overlay = Artifact(
                            analysis_id=analysis_id,
                            kind="gradcam_overlay",
                            storage_key=key,
                            content_type="image/png",
                            size_bytes=o_path.stat().st_size,
                            sha256=compute_sha256(o_path),
                        )
                        db.add(art_overlay)

            # Update analysis record with results
            analysis.status = "COMPLETE"
            analysis.stage = "COMPLETE"
            analysis.progress_pct = 100
            analysis.completed_at = datetime.now(UTC)
            if analysis.started_at:
                analysis.total_duration_ms = int(
                    (analysis.completed_at - analysis.started_at).total_seconds() * 1000
                )

            analysis.duration_seconds = result.audio_meta.duration_seconds
            analysis.sample_rate = result.audio_meta.sample_rate
            analysis.file_size_bytes = result.audio_meta.file_size_bytes

            analysis.quality = {
                "duration_s": result.quality.duration_s,
                "speech_ratio": result.quality.speech_ratio,
                "snr_estimate_db": result.quality.snr_estimate_db,
                "clipping_ratio": result.quality.clipping_ratio,
                "dc_offset": result.quality.dc_offset,
                "passed": result.quality.passed,
                "failures": result.quality.failures,
            }

            if result.acoustic:
                analysis.acoustic_result = {
                    "spoof_probability": result.acoustic.spoof_probability,
                    "window_scores": result.acoustic.window_scores,
                    "window_times": result.acoustic.window_times,
                    "uncertainty": result.acoustic.uncertainty,
                    "is_borderline": result.acoustic.is_borderline,
                    "model_version": result.acoustic.model_version,
                }
                analysis.acoustic_spoof_prob = result.acoustic.spoof_probability
                analysis.is_borderline = result.acoustic.is_borderline

            if result.transcript:
                analysis.transcript_result = {
                    "text": result.transcript.text,
                    "language": result.transcript.language,
                    "language_probability": result.transcript.language_probability,
                    "language_supported": result.transcript.language_supported,
                    "mean_confidence": result.transcript.mean_confidence,
                    "is_reliable": result.transcript.is_reliable,
                    "word_count": result.transcript.word_count,
                    "hallucination_flags": result.transcript.hallucination_flags,
                }
                analysis.detected_language = result.transcript.language

            if result.scam:
                analysis.scam_result = {
                    "scam_probability": result.scam.scam_probability,
                    "category_scores": result.scam.category_scores,
                    "triggered_categories": result.scam.triggered_categories,
                    "salient_spans": [
                        {"start": s.start, "end": s.end, "weight": s.weight, "text": s.text}
                        for s in result.scam.salient_spans
                    ],
                }
                analysis.scam_probability = result.scam.scam_probability

            if result.fusion:
                analysis.fusion_result = {
                    "risk_probability": result.fusion.risk_probability,
                    "verdict": result.fusion.verdict.value,
                    "confidence": result.fusion.confidence,
                    "feature_vector": result.fusion.feature_vector,
                    "contributions": result.fusion.contributions,
                    "reasons": result.fusion.reasons,
                    "overrides_applied": result.fusion.overrides_applied,
                }
                analysis.risk_probability = result.fusion.risk_probability
                analysis.verdict = result.fusion.verdict.value

            if result.explanation:
                analysis.explanation_result = {
                    "method": result.explanation.method,
                    "target_layer": result.explanation.target_layer,
                    "target_class": result.explanation.target_class,
                    "windows": result.explanation.windows,
                    "disclaimer": result.explanation.disclaimer,
                }

            analysis.stage_timings_ms = result.stage_timings_ms
            analysis.model_versions = result.model_versions
            analysis.degraded_branches = result.degraded_branches

            # Auto-issue challenge if borderline per 08 §3.1
            if auto_challenge and analysis.is_borderline:
                chal = issue_challenge()
                chal_record = Challenge(
                    analysis_id=analysis_id,
                    challenge_type=chal.challenge_type,
                    prompt_text=chal.prompt_text,
                    expected_phrase=chal.expected_phrase,
                    expected_duration_s=chal.expected_duration_s,
                    status="ISSUED",
                    expires_at=datetime.now(UTC).fromtimestamp(
                        datetime.now(UTC).timestamp() + 60.0
                    ),
                )
                db.add(chal_record)

            await db.commit()

        except Exception as e:
            log.error("analysis_processing_failed", analysis_id=analysis_id, error=str(e))
            try:
                analysis.status = "FAILED"
                analysis.error_code = "MODEL_ERROR"
                analysis.error_message = str(e)[:500]
                await db.commit()
            except Exception:
                # Session is broken — use a fresh session to record the failure
                try:
                    async with async_session_factory() as err_db:
                        err_stmt = select(Analysis).where(Analysis.id == analysis_id)
                        err_analysis = (await err_db.execute(err_stmt)).scalar_one_or_none()
                        if err_analysis:
                            err_analysis.status = "FAILED"
                            err_analysis.error_code = "MODEL_ERROR"
                            err_analysis.error_message = str(e)[:500]
                            await err_db.commit()
                except Exception:
                    log.error("failed_to_record_error", analysis_id=analysis_id)
        finally:
            # Clean up local temporary upload files
            try:
                if temp_audio_path.is_file():
                    temp_audio_path.unlink()
            except Exception:
                pass
