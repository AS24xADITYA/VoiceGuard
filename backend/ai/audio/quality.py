"""Audio quality assessment and gate.

Per 05 §1.4: computed once, stored, and passed to fusion as features.
If any floor is violated, the analysis still runs but the verdict is
forced to INCONCLUSIVE with the specific reason returned.

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import structlog

from ai.base import QualityReport

log = structlog.get_logger()

# Quality floors from 05 §1.4
SPEECH_RATIO_FLOOR: Final[float] = 0.20
SNR_FLOOR_DB: Final[float] = 5.0
CLIPPING_CEILING: Final[float] = 0.01
DC_OFFSET_CEILING: Final[float] = 0.01
MIN_DURATION: Final[float] = 1.5


def compute_speech_ratio(
    y: np.ndarray, sr: int = 16000, aggressiveness: int = 2, frame_ms: int = 30
) -> float:
    """Compute the ratio of VAD-detected voiced frames to total frames.

    Uses webrtcvad with the specified aggressiveness level and frame size.
    """
    try:
        import webrtcvad

        vad = webrtcvad.Vad(aggressiveness)

        # Convert to 16-bit PCM for webrtcvad
        pcm = (y * 32767).astype(np.int16).tobytes()

        frame_length = int(sr * frame_ms / 1000)
        frame_bytes = frame_length * 2  # 16-bit = 2 bytes per sample
        n_frames = len(pcm) // frame_bytes

        if n_frames == 0:
            return 0.0

        voiced = 0
        for i in range(n_frames):
            start = i * frame_bytes
            frame = pcm[start : start + frame_bytes]
            if len(frame) == frame_bytes:
                try:
                    if vad.is_speech(frame, sr):
                        voiced += 1
                except Exception:
                    pass

        return voiced / n_frames if n_frames > 0 else 0.0

    except ImportError:
        log.warning("webrtcvad_not_available", msg="Falling back to energy-based VAD")
        return _energy_vad_ratio(y, sr, frame_ms)


def _energy_vad_ratio(
    y: np.ndarray, sr: int = 16000, frame_ms: int = 30
) -> float:
    """Fallback energy-based VAD when webrtcvad is unavailable."""
    frame_length = int(sr * frame_ms / 1000)
    n_frames = len(y) // frame_length
    if n_frames == 0:
        return 0.0

    energies = []
    for i in range(n_frames):
        frame = y[i * frame_length : (i + 1) * frame_length]
        energies.append(np.mean(frame ** 2))

    energies = np.array(energies)
    threshold = np.median(energies) * 0.5
    voiced = np.sum(energies > threshold)
    return float(voiced / n_frames)


def compute_snr_estimate(y: np.ndarray, sr: int = 16000) -> float:
    """Estimate SNR as 10·log₁₀(P_voiced / P_unvoiced).

    Uses a simple energy-based segmentation.
    """
    frame_length = int(sr * 0.03)  # 30ms frames
    n_frames = len(y) // frame_length
    if n_frames < 2:
        return 0.0

    energies = np.array([
        np.mean(y[i * frame_length : (i + 1) * frame_length] ** 2)
        for i in range(n_frames)
    ])

    threshold = np.median(energies)
    voiced_energy = np.mean(energies[energies >= threshold]) if np.any(energies >= threshold) else 1e-10
    unvoiced_energy = np.mean(energies[energies < threshold]) if np.any(energies < threshold) else 1e-10

    if unvoiced_energy < 1e-10:
        return 60.0  # Very clean signal

    snr = 10 * np.log10(voiced_energy / unvoiced_energy)
    return float(snr)


def compute_clipping_ratio(y: np.ndarray) -> float:
    """Fraction of samples with |y| > 0.99."""
    return float(np.mean(np.abs(y) > 0.99))


def compute_dc_offset(y: np.ndarray) -> float:
    """|mean(y)|"""
    return float(np.abs(np.mean(y)))


def assess_quality(y: np.ndarray, sr: int = 16000) -> QualityReport:
    """Run the full quality gate per 05 §1.4.

    Returns a QualityReport with pass/fail status and specific failures.
    """
    duration_s = len(y) / sr
    speech_ratio = compute_speech_ratio(y, sr)
    snr_db = compute_snr_estimate(y, sr)
    clipping = compute_clipping_ratio(y)
    dc = compute_dc_offset(y)

    failures = []

    if duration_s < MIN_DURATION:
        failures.append(f"Duration {duration_s:.1f}s below minimum {MIN_DURATION}s")

    if speech_ratio < SPEECH_RATIO_FLOOR:
        failures.append(
            f"Speech ratio {speech_ratio:.2f} below minimum {SPEECH_RATIO_FLOOR}"
        )

    if snr_db < SNR_FLOOR_DB:
        failures.append(
            f"Estimated SNR {snr_db:.1f} dB below minimum {SNR_FLOOR_DB} dB"
        )

    if clipping > CLIPPING_CEILING:
        failures.append(
            f"Clipping ratio {clipping:.3f} exceeds maximum {CLIPPING_CEILING}"
        )

    if dc > DC_OFFSET_CEILING:
        failures.append(
            f"DC offset {dc:.4f} exceeds maximum {DC_OFFSET_CEILING}"
        )

    passed = len(failures) == 0

    return QualityReport(
        duration_s=duration_s,
        speech_ratio=speech_ratio,
        snr_estimate_db=snr_db,
        clipping_ratio=clipping,
        dc_offset=dc,
        passed=passed,
        failures=failures,
    )


def compute_audio_quality_score(quality: QualityReport) -> float:
    """Compute a composite [0,1] quality score for the fusion feature vector.

    Higher = better quality.
    """
    scores = []

    # Speech ratio: 0 at floor, 1 at 0.8+
    sr_score = min(1.0, max(0.0, (quality.speech_ratio - SPEECH_RATIO_FLOOR) / 0.6))
    scores.append(sr_score)

    # SNR: 0 at floor, 1 at 30+ dB
    snr_score = min(1.0, max(0.0, (quality.snr_estimate_db - SNR_FLOOR_DB) / 25.0))
    scores.append(snr_score)

    # Clipping: 1 at 0, 0 at ceiling
    clip_score = max(0.0, 1.0 - quality.clipping_ratio / CLIPPING_CEILING)
    scores.append(clip_score)

    # DC offset: 1 at 0, 0 at ceiling
    dc_score = max(0.0, 1.0 - quality.dc_offset / DC_OFFSET_CEILING)
    scores.append(dc_score)

    return float(np.mean(scores))
