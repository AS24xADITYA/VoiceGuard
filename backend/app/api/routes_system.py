"""System routes: health, config, metrics.

These endpoints are public (no auth required).
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(prefix="/system", tags=["system"])

_start_time = time.time()


@router.get(
    "/health",
    summary="System health check",
    description="Reports component load status, queue state, and uptime.",
)
async def health() -> dict[str, Any]:
    """Health endpoint per 08 §6.

    Returns per-component {loaded, warm, version, load_ms}.
    status: healthy | degraded | unhealthy.
    """
    from ai.registry import get_registry

    settings = get_settings()
    registry = get_registry()
    components = registry.health_report()

    # Determine overall status
    acoustic_ok = components.get("acoustic", {}).get("loaded", False)
    whisper_ok = components.get("whisper", {}).get("loaded", False)

    if acoustic_ok or whisper_ok:
        status = "healthy"
    elif not acoustic_ok and not whisper_ok:
        # Check if any predictive branch is loaded
        has_any_predictive = acoustic_ok or whisper_ok
        status = "unhealthy" if not has_any_predictive else "degraded"
    else:
        status = "degraded"

    # Check if any non-essential component failed
    for name, info in components.items():
        if name not in ("acoustic", "whisper") and not info.get("loaded", True):
            if status == "healthy":
                status = "degraded"

    return {
        "status": status,
        "version": "0.1.0",
        "pipeline_version": "0.1.0",
        "uptime_seconds": int(time.time() - _start_time),
        "device": settings.resolve_device(),
        "components": components,
        "queue": registry.queue_status(),
    }


@router.get(
    "/config",
    summary="Public configuration",
    description="Configuration values the frontend needs: limits, formats, languages, thresholds.",
)
async def config() -> dict[str, Any]:
    """Public config endpoint per 08 §6."""
    settings = get_settings()
    return {
        "limits": {
            "max_upload_mb": settings.max_upload_mb,
            "max_duration_seconds": settings.max_duration_seconds,
            "min_duration_seconds": settings.min_duration_seconds,
            "max_concurrent_inference": settings.max_concurrent_inference,
        },
        "supported_formats": ["wav", "mp3", "m4a", "flac", "ogg", "webm"],
        "supported_languages": settings.whisper_language_list,
        "language_tiers": {
            "production": ["en", "hi", "ta"],
            "experimental_degraded": ["mr", "bn"],
            "notes": "Bengali (bn, S3 F1 0.0769) and Marathi (mr, S3 F1 0.2143) exhibit severe ASR phonetic degradation under Whisper telephone transcription per 13 §7.3.",
        },
        "verdict_thresholds": {
            "moderate": settings.verdict_threshold_moderate,
            "high": settings.verdict_threshold_high,
        },
        "challenge_types": [
            "WHISPER",
            "PITCH_UP",
            "PITCH_DOWN",
            "SLOW_SPEECH",
            "SUSTAINED_VOWEL",
            "COUNT_BACKWARD",
        ],
        "device": settings.resolve_device(),
    }


@router.get(
    "/metrics",
    summary="Published evaluation metrics",
    description="Metrics of the currently loaded models, from the evaluation artefact.",
)
async def metrics() -> dict[str, Any]:
    """Metrics endpoint per 08 §6.

    Reads from the artefact metrics.json produced by the evaluation protocol.
    Until models are trained and evaluated, returns empty metrics.
    """
    from ai.registry import get_registry

    registry = get_registry()
    return registry.get_metrics()
