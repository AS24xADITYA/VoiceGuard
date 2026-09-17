"""Synthetic audio test fixtures for Phase 2 tests.

Provides synthetic audio generators for:
  - 1s, 4s, 10s, 65s clean audio samples
  - pure silence
  - heavily clipped audio
  - short clip (0.5s)
  - corrupt file
  - format magic-byte sample files
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import soundfile as sf


def create_synthetic_audio(
    duration_s: float = 4.0,
    sr: int = 16000,
    frequencies: tuple[float, ...] = (220.0, 440.0, 880.0),
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate multi-frequency harmonic tone simulating speech frequencies."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    y = np.zeros_like(t)
    for i, f in enumerate(frequencies):
        weight = 1.0 / (i + 1)
        y += weight * np.sin(2 * np.pi * f * t)
    # Normalize to desired peak amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) * amplitude
    return y.astype(np.float32)


def create_silence_audio(duration_s: float = 4.0, sr: int = 16000) -> np.ndarray:
    """Generate pure digital silence."""
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def create_clipped_audio(
    duration_s: float = 4.0, sr: int = 16000, gain: float = 10.0
) -> np.ndarray:
    """Generate heavily clipped audio (> 1% clipped samples)."""
    clean = create_synthetic_audio(duration_s=duration_s, sr=sr, amplitude=0.9)
    amplified = clean * gain
    clipped = np.clip(amplified, -1.0, 1.0)
    return clipped.astype(np.float32)


def write_wav_file(path: Path, y: np.ndarray, sr: int = 16000, subtype: str = "PCM_16") -> Path:
    """Write float32 audio samples to a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype=subtype)
    return path


def write_corrupt_file(path: Path) -> Path:
    """Write an unparseable corrupt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"NOT_A_VALID_HEADER_DATA_GARBAGE_1234567890" * 10)
    return path
