"""Audio feature extraction: log-mel spectrogram, MFCC, windowing.

All parameters per 05 §2.1 — do not deviate; training and inference
must match bit for bit.

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

from typing import Final

import librosa
import numpy as np

# ── Exact spectrogram parameters from 05 §2.1 ──────────────────────
SAMPLE_RATE: Final[int] = 16000
N_FFT: Final[int] = 1024       # 64 ms window
HOP_LENGTH: Final[int] = 160   # 10 ms hop
WIN_LENGTH: Final[int] = 400   # 25 ms
WINDOW: Final[str] = "hann"
N_MELS: Final[int] = 128
F_MIN: Final[int] = 20         # Hz
F_MAX: Final[int] = 8000       # Nyquist
POWER: Final[float] = 2.0
TOP_DB: Final[float] = 80      # dynamic range floor for power_to_db

# ── Windowing parameters from 05 §2.2 ──────────────────────────────
WINDOW_SECONDS: Final[float] = 4.0
WINDOW_FRAMES: Final[int] = 400    # 4.0 s / 10 ms hop
HOP_SECONDS: Final[float] = 2.0   # 50% overlap

# ── MFCC parameters ─────────────────────────────────────────────────
N_MFCC: Final[int] = 40


def log_mel(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute log-mel spectrogram per 05 §2.1.

    Args:
        y: Audio samples as float32 numpy array.
        sr: Sample rate (must be 16000).

    Returns:
        Log-mel spectrogram of shape (128, T) with dB values in [-80, 0].
    """
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=POWER,
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)
    return S_db  # (128, T), dB in [-80, 0]


def normalize_spectrogram(S_db: np.ndarray) -> np.ndarray:
    """Per-example normalisation to zero mean, unit variance.

    Per 05 §2.1: global dataset statistics are NOT used.
    """
    return (S_db - S_db.mean()) / (S_db.std() + 1e-8)


def compute_mfcc(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute MFCCs from the same mel basis.

    Per 05 §2.1: n_mfcc=40, used by challenge–response comparator
    and quality diagnostics, NOT fed to the CNN.

    Returns:
        MFCCs of shape (40, T).
    """
    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
    )


def segment_windows(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> list[tuple[np.ndarray, float, float]]:
    """Segment audio into overlapping windows per 05 §2.2.

    Args:
        y: Audio samples.
        sr: Sample rate.
        window_seconds: Window length in seconds.
        hop_seconds: Hop between windows in seconds.

    Returns:
        List of (audio_window, t_start, t_end) tuples.
        Audio shorter than window_seconds is reflect-padded to window_seconds.
    """
    window_samples = int(window_seconds * sr)
    hop_samples = int(hop_seconds * sr)

    # Reflect-pad short audio
    if len(y) < window_samples:
        y = np.pad(y, (0, window_samples - len(y)), mode="reflect")

    windows = []
    start = 0
    while start + window_samples <= len(y):
        t_start = start / sr
        t_end = (start + window_samples) / sr
        windows.append((y[start : start + window_samples], t_start, t_end))
        start += hop_samples

    # If we didn't cover the end, add a final window
    if start < len(y) and len(y) - start >= sr:  # at least 1 second remaining
        end_window = y[-window_samples:]
        t_start = (len(y) - window_samples) / sr
        t_end = len(y) / sr
        windows.append((end_window, t_start, t_end))

    return windows


def spectrogram_to_tensor(S_norm: np.ndarray) -> np.ndarray:
    """Convert normalised spectrogram to 3-channel tensor for CNN input.

    Per 05 §2.2: Input tensor per window is (1, 128, 400) → replicated
    to (3, 128, 400) for the pretrained backbone.

    Args:
        S_norm: Normalised spectrogram of shape (128, W) where W is the
                window frame count.

    Returns:
        numpy array of shape (3, 128, W).
    """
    # Ensure exactly WINDOW_FRAMES columns
    if S_norm.shape[1] < WINDOW_FRAMES:
        pad_width = WINDOW_FRAMES - S_norm.shape[1]
        S_norm = np.pad(S_norm, ((0, 0), (0, pad_width)), mode="reflect")
    elif S_norm.shape[1] > WINDOW_FRAMES:
        S_norm = S_norm[:, :WINDOW_FRAMES]

    # Replicate to 3 channels
    return np.stack([S_norm, S_norm, S_norm], axis=0).astype(np.float32)
