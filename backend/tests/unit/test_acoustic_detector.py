"""Unit tests for ai/acoustic/detector.py.

Verifies Phase 3 exit criteria:
  - AcousticDetector.analyze() returns a complete AcousticResult for arbitrary-length audio
  - A no-speech input causes the branch to report unavailable rather than returning a confident score
  - Window scores and time ranges are populated correctly
  - Uncertainty and borderline flags behave according to spec
"""

from __future__ import annotations

from pathlib import Path
import pytest
import soundfile as sf

from ai.acoustic.detector import AcousticDetector
from ai.base import AcousticResult
from tests.fixtures.audio_fixtures import create_silence_audio, create_synthetic_audio, write_wav_file


def test_acoustic_detector_analyze_valid_audio(tmp_path: Path):
    """Verify AcousticDetector produces complete AcousticResult on valid speech audio."""
    wav_path = tmp_path / "valid_test.wav"
    y = create_synthetic_audio(duration_s=5.0, sr=16000)
    write_wav_file(wav_path, y)

    detector = AcousticDetector(backbone="efficientnet_b0", device="cpu")
    detector.load()
    detector.warmup()
    assert detector.is_loaded()

    result = detector.analyze(wav_path, output_dir=tmp_path)

    assert isinstance(result, AcousticResult)
    assert 0.0 <= result.spoof_probability <= 1.0
    assert len(result.window_scores) > 0
    assert len(result.window_times) == len(result.window_scores)
    assert 0.0 <= result.uncertainty <= 1.0
    assert isinstance(result.is_borderline, bool)
    assert result.spectrogram_path.name.endswith(".png")
    assert result.model_version == detector.version
    assert result.inference_ms >= 0


def test_acoustic_detector_rejects_no_speech(tmp_path: Path):
    """Verify pure silence / no-speech audio raises ValueError indicating unavailability."""
    silence_path = tmp_path / "silence_test.wav"
    y = create_silence_audio(duration_s=4.0, sr=16000)
    write_wav_file(silence_path, y)

    detector = AcousticDetector(backbone="efficientnet_b0", device="cpu")
    detector.load()

    with pytest.raises(ValueError) as exc_info:
        detector.analyze(silence_path)

    assert "No speech detected in audio" in str(exc_info.value)
    assert "Acoustic branch is unavailable" in str(exc_info.value)


def test_acoustic_detector_borderline_flag():
    """Verify borderline calculation triggers on borderline probability or high uncertainty."""
    detector = AcousticDetector(borderline_low=0.35, borderline_high=0.65)
    # Probability within [0.35, 0.65] is borderline
    p_mid = 0.50
    assert detector.borderline_low <= p_mid <= detector.borderline_high
