"""Unit tests for ai/audio/io.py.

Verifies Phase 2 exit criteria:
  - Magic byte validation identifies supported formats
  - Unsupported format raises AudioValidationError("UNSUPPORTED_FORMAT")
  - Duration limits enforce min (1.5s) and max (300s)
  - Amplitude normalization scales peak to -1 dBFS
  - Filename sanitization strips dangerous characters
  - SHA-256 computation is deterministic
  - Corrupt input raises the correct typed exception
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from ai.audio.io import (
    AudioValidationError,
    compute_sha256,
    normalize_amplitude,
    sanitize_filename,
    validate_duration,
    validate_file_size,
    validate_magic_bytes,
)
from tests.fixtures.audio_fixtures import (
    create_synthetic_audio,
    write_corrupt_file,
    write_wav_file,
)


def test_validate_magic_bytes_wav(tmp_path: Path):
    """Verify WAV magic bytes are recognized."""
    wav_path = tmp_path / "test.wav"
    y = create_synthetic_audio(2.0)
    write_wav_file(wav_path, y)

    fmt = validate_magic_bytes(wav_path)
    assert fmt == "wav"


def test_validate_magic_bytes_supported_formats(tmp_path: Path):
    """Verify magic bytes detection across mock headers for supported formats."""
    test_cases = [
        ("test.mp3", b"ID3\x03\x00\x00\x00\x00\x00\x00paddingdata", "mp3"),
        ("test.flac", b"fLaC\x00\x00\x00\"paddingdata", "flac"),
        ("test.ogg", b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00", "ogg"),
        ("test.m4a", b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00", "m4a"),
        ("test.webm", b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01", "webm"),
    ]
    for filename, header_bytes, expected_fmt in test_cases:
        p = tmp_path / filename
        p.write_bytes(header_bytes)
        assert validate_magic_bytes(p) == expected_fmt


def test_validate_magic_bytes_unsupported(tmp_path: Path):
    """Verify unsupported format raises AudioValidationError with UNSUPPORTED_FORMAT."""
    corrupt_path = tmp_path / "corrupt.bin"
    write_corrupt_file(corrupt_path)

    with pytest.raises(AudioValidationError) as exc_info:
        validate_magic_bytes(corrupt_path)
    assert exc_info.value.code == "UNSUPPORTED_FORMAT"


def test_validate_duration_bounds():
    """Verify duration bounds checking."""
    # Valid
    validate_duration(2.0, min_seconds=1.5, max_seconds=300.0)
    validate_duration(150.0, min_seconds=1.5, max_seconds=300.0)

    # Too short (< 1.5s)
    with pytest.raises(AudioValidationError) as exc_1:
        validate_duration(1.0, min_seconds=1.5, max_seconds=300.0)
    assert exc_1.value.code == "DURATION_OUT_OF_RANGE"

    # Too long (> 300s)
    with pytest.raises(AudioValidationError) as exc_2:
        validate_duration(301.0, min_seconds=1.5, max_seconds=300.0)
    assert exc_2.value.code == "DURATION_OUT_OF_RANGE"


def test_validate_file_size(tmp_path: Path):
    """Verify file size validation raises PAYLOAD_TOO_LARGE."""
    small_file = tmp_path / "small.dat"
    small_file.write_bytes(b"x" * 1024)
    validate_file_size(small_file, max_mb=25)

    with pytest.raises(AudioValidationError) as exc_info:
        validate_file_size(small_file, max_mb=0)  # max 0 MB forces failure
    assert exc_info.value.code == "PAYLOAD_TOO_LARGE"


def test_normalize_amplitude():
    """Verify peak normalisation scales peak to -1 dBFS (~0.891)."""
    y = np.array([-0.2, 0.4, -0.5, 0.3], dtype=np.float32)
    normalized = normalize_amplitude(y)
    expected_peak = 10 ** (-1.0 / 20.0)  # ~0.89125
    assert abs(np.max(np.abs(normalized)) - expected_peak) < 1e-4


def test_sanitize_filename():
    """Verify sanitize_filename strips path traversal and invalid characters."""
    assert sanitize_filename("../../../etc/passwd") == "______etc_passwd"
    assert sanitize_filename("safe_recording.wav") == "safe_recording.wav"
    assert sanitize_filename("audio\x00null.mp3") == "audio_null.mp3"
    assert sanitize_filename("") == "unnamed"


def test_compute_sha256(tmp_path: Path):
    """Verify SHA-256 calculation is correct and reproducible."""
    p = tmp_path / "test.txt"
    p.write_bytes(b"voiceguard-test-data")
    h1 = compute_sha256(p)
    h2 = compute_sha256(p)
    assert h1 == h2
    assert len(h1) == 64
