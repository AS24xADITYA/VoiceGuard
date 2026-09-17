"""VoiceGuard AI pipeline orchestrator.

Per 05 §7:
  - Ingestion: canonicalises audio, computes SHA-256, runs quality gate
  - Concurrency: runs acoustic detection and Whisper transcription concurrently
  - Intent classification: scores transcript with scam classifier if text exists
  - Challenge verification: integrates challenge deltas on re-fusion
  - Calibrated fusion: merges signals into 14-dim vector with hard overrides
  - Grad-CAM explainability: attaches attention heatmap to acoustic predictions (non-blocking)
  - End-to-end runnable from a script with ZERO web dependencies

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

from ai.audio.io import canonicalize, sanitize_filename
from ai.audio.quality import assess_quality
from ai.base import (
    AcousticResult,
    AudioMeta,
    ChallengeResult,
    ExplanationResult,
    FusionResult,
    PipelineResult,
    QualityReport,
    ScamIntentResult,
    TranscriptResult,
)
from ai.fusion.features import assemble_features
from ai.registry import get_registry

log = structlog.get_logger()


@dataclass
class PipelineContext:
    """Execution context for progress reporting and challenge re-fusion."""

    stage_callback: Callable[[str], None] | None = None
    challenge_result: ChallengeResult | None = None
    output_dir: Path | None = None
    stage_timings: dict[str, int] = field(default_factory=dict)

    def set_stage(self, stage_name: str) -> None:
        if self.stage_callback:
            try:
                self.stage_callback(stage_name)
            except Exception:
                pass


async def run_pipeline(
    audio_path: Path | str,
    context: PipelineContext | None = None,
) -> PipelineResult:
    """Run the complete VoiceGuard multi-signal analysis pipeline.

    Args:
        audio_path: Path to input audio file (WAV, MP3, FLAC, OGG, M4A, WEBM).
        context: Optional PipelineContext for progress reporting and artifacts.

    Returns:
        Complete PipelineResult dataclass.
    """
    ctx = context or PipelineContext()
    path = Path(audio_path)
    out_dir = ctx.output_dir or path.parent

    timings: dict[str, int] = {}
    degraded_branches: list[str] = []

    # ── Stage 1: INGESTION & QUALITY GATE ───────────────────────────
    ctx.set_stage("INGESTING")
    t0 = time.perf_counter()

    canonical_wav = out_dir / f"{path.stem}_canonical.wav"
    audio_meta = canonicalize(path, canonical_wav)

    # Load canonical audio array and assess quality
    from ai.audio.io import load_canonical
    y_audio, sr = load_canonical(canonical_wav)
    quality_report = assess_quality(y_audio, sr=sr)

    timings["ingestion_ms"] = int((time.perf_counter() - t0) * 1000)

    # ── Stage 2: CONCURRENT ACOUSTIC & TRANSCRIPTION ───────────────
    ctx.set_stage("ANALYZING_ACOUSTIC")
    t1 = time.perf_counter()

    registry = get_registry()
    loop = asyncio.get_running_loop()

    # Define tasks to run concurrently in threads
    def _run_acoustic() -> AcousticResult | None:
        try:
            detector = registry.get_acoustic()
            return detector.analyze(canonical_wav, output_dir=out_dir)
        except Exception as e:
            log.warning("acoustic_branch_failed", error=str(e))
            return None

    def _run_transcription() -> TranscriptResult | None:
        try:
            transcriber = registry.get_transcriber()
            return transcriber.transcribe(canonical_wav)
        except Exception as e:
            log.warning("transcription_branch_failed", error=str(e))
            return None

    # Run concurrently via asyncio.gather
    acoustic_res, transcript_res = await asyncio.gather(
        loop.run_in_executor(None, _run_acoustic),
        loop.run_in_executor(None, _run_transcription),
    )

    if acoustic_res is None:
        degraded_branches.append("acoustic")
    if transcript_res is None or not transcript_res.text:
        degraded_branches.append("transcription")

    timings["acoustic_transcription_ms"] = int((time.perf_counter() - t1) * 1000)

    # ── Stage 3: SCAM INTENT CLASSIFICATION ─────────────────────────
    ctx.set_stage("SCORING_INTENT")
    t2 = time.perf_counter()

    scam_res: ScamIntentResult | None = None
    if transcript_res is not None and transcript_res.text.strip():
        def _run_scam() -> ScamIntentResult | None:
            try:
                clf = registry.get_scam_classifier()
                return clf.score(transcript_res)
            except Exception as e:
                log.warning("scam_classifier_failed", error=str(e))
                return None

        scam_res = await loop.run_in_executor(None, _run_scam)
        if scam_res is None:
            degraded_branches.append("scam_classifier")

    timings["scam_intent_ms"] = int((time.perf_counter() - t2) * 1000)

    # ── Stage 4: FUSION LAYER ───────────────────────────────────────
    ctx.set_stage("FUSING")
    t3 = time.perf_counter()

    features = assemble_features(
        quality=quality_report,
        acoustic=acoustic_res,
        transcript=transcript_res,
        scam=scam_res,
        challenge=ctx.challenge_result,
    )

    fuser = registry.get_fuser()
    fusion_res = fuser.fuse(features, quality=quality_report)

    timings["fusion_ms"] = int((time.perf_counter() - t3) * 1000)

    # ── Stage 5: GRAD-CAM EXPLAINABILITY ───────────────────────────
    ctx.set_stage("EXPLAINING")
    t4 = time.perf_counter()

    explanation_res: ExplanationResult | None = None
    if acoustic_res is not None:
        def _run_explain() -> ExplanationResult | None:
            try:
                explainer = registry.get_explainer()
                # Target class 1 if spoof probability > 0.5 else 0
                target_cls = 1 if acoustic_res.spoof_probability >= 0.50 else 0
                return explainer.explain(
                    canonical_wav,
                    target_class=target_cls,
                    output_dir=out_dir,
                )
            except Exception as e:
                log.warning("explanation_failed", error=str(e))
                return None

        explanation_res = await loop.run_in_executor(None, _run_explain)

    timings["explaining_ms"] = int((time.perf_counter() - t4) * 1000)

    ctx.set_stage("COMPLETE")

    # Record component versions
    model_versions = {
        "pipeline": "0.1.0",
        "acoustic": acoustic_res.model_version if acoustic_res else "unavailable",
        "transcriber": transcript_res.model_version if transcript_res else "unavailable",
        "scam_classifier": scam_res.model_version if scam_res else "unavailable",
        "fuser": fusion_res.model_version if fusion_res else "unavailable",
    }

    return PipelineResult(
        audio_meta=audio_meta,
        quality=quality_report,
        acoustic=acoustic_res,
        transcript=transcript_res,
        scam=scam_res,
        challenge=ctx.challenge_result,
        fusion=fusion_res,
        explanation=explanation_res,
        stage_timings_ms=timings,
        model_versions=model_versions,
        degraded_branches=degraded_branches,
    )
