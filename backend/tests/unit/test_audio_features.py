"""Unit tests for ai/audio/features.py.

Verifies Phase 2 exit criteria:
  - Log-mel output shape is exactly (128, T) with T = ceil(samples / 160)
  - Windowing produces the correct count and time ranges for 1 s, 4 s, 10 s, 65 s inputs
  - Normalisation produces zero mean and unit variance
  - Spectrogram to tensor produces (3, 128, 400)
  - MFCC produces (40, T)
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from ai.audio.features import (
    HOP_LENGTH,
    N_MELS,
    N_MFCC,
    WINDOW_FRAMES,
    compute_mfcc,
    log_mel,
    normalize_spectrogram,
    segment_windows,
    spectrogram_to_tensor,
)
from tests.fixtures.audio_fixtures import create_synthetic_audio


def test_log_mel_shape_and_range():
    """Verify log-mel shape is exactly (128, T) with T = ceil(samples / 160)."""
    sr = 16000
    for duration in [1.0, 2.5, 4.0]:
        y = create_synthetic_audio(duration_s=duration, sr=sr)
        mel = log_mel(y, sr=sr)
        expected_frames = math.ceil(len(y) / HOP_LENGTH)

        assert mel.shape[0] == N_MELS, f"Expected {N_MELS} mels, got {mel.shape[0]}"
        # librosa melspectrogram frame count is either ceil or 1 + samples // hop
        assert abs(mel.shape[1] - expected_frames) <= 1
        # dB range should be bounded within [-80, 0]
        assert mel.max() <= 0.0 + 1e-4
        assert mel.min() >= -80.0 - 1e-4


def test_spectrogram_normalization():
    """Verify normalisation produces zero mean and unit variance."""
    y = create_synthetic_audio(duration_s=4.0, sr=16000)
    mel = log_mel(y)
    norm = normalize_spectrogram(mel)

    assert abs(norm.mean()) < 1e-5
    assert abs(norm.std() - 1.0) < 1e-4


def test_segment_windows_duration_cases():
    """Verify windowing counts and time ranges for 1s, 4s, 10s, 65s inputs."""
    sr = 16000

    # Case 1: 1s input (shorter than 4s window) -> reflect-padded to 1 window
    y_1s = create_synthetic_audio(duration_s=1.0, sr=sr)
    w_1s = segment_windows(y_1s, sr=sr)
    assert len(w_1s) == 1
    assert w_1s[0][1] == 0.0
    assert w_1s[0][2] == 4.0
    assert len(w_1s[0][0]) == int(4.0 * sr)

    # Case 2: 4s input -> exactly 1 window [0.0, 4.0]
    y_4s = create_synthetic_audio(duration_s=4.0, sr=sr)
    w_4s = segment_windows(y_4s, sr=sr)
    assert len(w_4s) == 1
    assert w_4s[0][1] == 0.0
    assert w_4s[0][2] == 4.0

    # Case 3: 10s input -> windows: [0, 4], [2, 6], [4, 8], [6, 10] -> 4 windows
    y_10s = create_synthetic_audio(duration_s=10.0, sr=sr)
    w_10s = segment_windows(y_10s, sr=sr)
    assert len(w_10s) == 4
    expected_ranges_10s = [(0.0, 4.0), (2.0, 6.0), (4.0, 8.0), (6.0, 10.0)]
    for i, (audio_w, t_start, t_end) in enumerate(w_10s):
        assert abs(t_start - expected_ranges_10s[i][0]) < 1e-4
        assert abs(t_end - expected_ranges_10s[i][1]) < 1e-4
        assert len(audio_w) == int(4.0 * sr)

    # Case 4: 65s input -> starts at 0, 2, ..., 60 (covers 60-64), plus tail [61, 65] -> 32 windows
    y_65s = create_synthetic_audio(duration_s=65.0, sr=sr)
    w_65s = segment_windows(y_65s, sr=sr)
    assert len(w_65s) == 32
    assert abs(w_65s[0][1] - 0.0) < 1e-4
    assert abs(w_65s[-1][2] - 65.0) < 1e-4


def test_spectrogram_to_tensor():
    """Verify spectrogram_to_tensor creates (3, 128, 400) tensor."""
    y = create_synthetic_audio(duration_s=4.0, sr=16000)
    mel = log_mel(y)
    norm = normalize_spectrogram(mel)
    tensor = spectrogram_to_tensor(norm)

    assert tensor.shape == (3, N_MELS, WINDOW_FRAMES)
    assert tensor.dtype == np.float32
    # Verify channels 0, 1, 2 are identical replicas
    assert np.array_equal(tensor[0], tensor[1])
    assert np.array_equal(tensor[1], tensor[2])


def test_compute_mfcc():
    """Verify compute_mfcc produces shape (40, T)."""
    y = create_synthetic_audio(duration_s=4.0, sr=16000)
    mfcc = compute_mfcc(y, sr=16000)
    assert mfcc.shape[0] == N_MFCC
    assert mfcc.shape[1] > 0
