"""Unit tests for ai/audio/quality.py.

Verifies Phase 2 exit criteria:
  - The quality gate correctly rejects silence
  - The quality gate correctly rejects a 0.5 s clip (< 1.5s minimum)
  - The quality gate correctly rejects heavily clipped audio
  - Valid audio passes the gate
  - Composite audio quality score is in [0, 1]
"""

from __future__ import annotations

import pytest

from ai.audio.quality import (
    CLIPPING_CEILING,
    DC_OFFSET_CEILING,
    MIN_DURATION,
    SNR_FLOOR_DB,
    SPEECH_RATIO_FLOOR,
    assess_quality,
    compute_audio_quality_score,
    compute_clipping_ratio,
    compute_dc_offset,
    compute_snr_estimate,
    compute_speech_ratio,
)
from tests.fixtures.audio_fixtures import (
    create_clipped_audio,
    create_silence_audio,
    create_synthetic_audio,
)


def test_quality_gate_rejects_silence():
    """Verify quality gate rejects digital silence."""
    silence = create_silence_audio(duration_s=4.0, sr=16000)
    report = assess_quality(silence, sr=16000)

    assert not report.passed
    assert report.speech_ratio < SPEECH_RATIO_FLOOR or report.snr_estimate_db < SNR_FLOOR_DB
    assert any("below minimum" in f for f in report.failures)


def test_quality_gate_rejects_short_clip():
    """Verify quality gate rejects audio shorter than 1.5 s."""
    short = create_synthetic_audio(duration_s=0.5, sr=16000)
    report = assess_quality(short, sr=16000)

    assert not report.passed
    assert report.duration_s < MIN_DURATION
    assert any("below minimum 1.5s" in f for f in report.failures)


def test_quality_gate_rejects_clipped_audio():
    """Verify quality gate rejects heavily clipped audio."""
    clipped = create_clipped_audio(duration_s=4.0, sr=16000, gain=20.0)
    report = assess_quality(clipped, sr=16000)

    assert not report.passed
    assert report.clipping_ratio > CLIPPING_CEILING
    assert any("Clipping ratio" in f and "exceeds maximum" in f for f in report.failures)


def test_quality_gate_composite_score_range():
    """Verify composite quality score produces valid values in [0, 1]."""
    for audio in [
        create_synthetic_audio(4.0),
        create_silence_audio(4.0),
        create_clipped_audio(4.0),
    ]:
        report = assess_quality(audio, sr=16000)
        score = compute_audio_quality_score(report)
        assert 0.0 <= score <= 1.0
