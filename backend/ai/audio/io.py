"""Audio ingestion, validation, canonicalisation, and hashing.

Per 05 §1: all downstream stages consume only the canonical 16 kHz
mono 16-bit PCM WAV form. No stage re-decodes the original.

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import hashlib
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Final

import numpy as np
import soundfile as sf
import structlog

from ai.base import AudioMeta

log = structlog.get_logger()

# Supported container formats by magic bytes
MAGIC_BYTES: dict[str, list[bytes]] = {
    "wav": [b"RIFF"],
    "mp3": [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3"],
    "flac": [b"fLaC"],
    "ogg": [b"OggS"],
    "m4a": [b"\x00\x00\x00"],  # ftyp box — first 4 bytes vary, check offset 4
    "webm": [b"\x1a\x45\xdf\xa3"],
}

# m4a/mp4 uses an ftyp box — "ftyp" appears at offset 4
M4A_FTYP_OFFSET: Final[int] = 4
M4A_FTYP_MAGIC: Final[bytes] = b"ftyp"

CANONICAL_SAMPLE_RATE: Final[int] = 16000
CANONICAL_CHANNELS: Final[int] = 1
CANONICAL_SUBTYPE: Final[str] = "PCM_16"


class AudioValidationError(Exception):
    """Raised when audio fails validation."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


def validate_magic_bytes(filepath: Path) -> str:
    """Validate file magic bytes against supported formats.

    Returns the detected format string.
    Raises AudioValidationError with UNSUPPORTED_FORMAT if no match.
    """
    with open(filepath, "rb") as f:
        header = f.read(32)

    if len(header) < 4:
        raise AudioValidationError("UNSUPPORTED_FORMAT", "File is too small to be valid audio.")

    # Check standard magic bytes
    for fmt, magics in MAGIC_BYTES.items():
        if fmt == "m4a":
            # m4a: check for ftyp at offset 4
            if len(header) > 8 and header[M4A_FTYP_OFFSET : M4A_FTYP_OFFSET + 4] == M4A_FTYP_MAGIC:
                return "m4a"
            continue
        for magic in magics:
            if header[: len(magic)] == magic:
                return fmt

    raise AudioValidationError(
        "UNSUPPORTED_FORMAT",
        "File does not match any supported audio format (WAV, MP3, M4A, FLAC, OGG, WEBM).",
    )


def validate_file_size(filepath: Path, max_mb: int = 25) -> None:
    """Validate file size does not exceed the maximum.

    Raises AudioValidationError with PAYLOAD_TOO_LARGE.
    """
    size_bytes = filepath.stat().st_size
    max_bytes = max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise AudioValidationError(
            "PAYLOAD_TOO_LARGE",
            f"File size {size_bytes / (1024*1024):.1f} MB exceeds the maximum of {max_mb} MB.",
        )


def probe_duration(filepath: Path) -> float:
    """Use ffprobe to get the duration of an audio file.

    Falls back to ffmpeg stream timestamp scanning for live-streamed containers
    (e.g., browser MediaRecorder WebM/Opus blobs without duration container header).
    Raises AudioValidationError with CORRUPT_AUDIO on failure.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(filepath),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            stdout_str = result.stdout.strip()
            if stdout_str and stdout_str != "N/A":
                try:
                    return float(stdout_str)
                except ValueError:
                    pass

        # Fallback for streamed media (browser WebM without container duration header)
        null_res = subprocess.run(
            ["ffmpeg", "-i", str(filepath), "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        import re

        matches = re.findall(r"time=(\d+):(\d+):(\d+\.?\d*)", null_res.stderr)
        if matches:
            hours, minutes, seconds = matches[-1]
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

        if result.returncode != 0:
            raise AudioValidationError(
                "CORRUPT_AUDIO",
                f"ffprobe failed: {result.stderr.strip()[:200]}",
            )
        raise AudioValidationError("CORRUPT_AUDIO", "Cannot determine audio duration from stream.")
    except AudioValidationError:
        raise
    except subprocess.TimeoutExpired as e:
        raise AudioValidationError("CORRUPT_AUDIO", f"Audio probe timed out: {e}")
    except Exception as e:
        raise AudioValidationError("CORRUPT_AUDIO", f"Cannot determine audio duration: {e}")


def validate_duration(
    duration: float,
    min_seconds: float = 1.5,
    max_seconds: float = 300.0,
) -> None:
    """Validate audio duration is within bounds.

    Raises AudioValidationError with DURATION_OUT_OF_RANGE.
    """
    if duration < min_seconds:
        raise AudioValidationError(
            "DURATION_OUT_OF_RANGE",
            f"Audio duration {duration:.1f}s is below the minimum of {min_seconds}s.",
        )
    if duration > max_seconds:
        raise AudioValidationError(
            "DURATION_OUT_OF_RANGE",
            f"Audio duration {duration:.1f}s exceeds the maximum of {max_seconds}s.",
        )


def canonicalize(src: Path, dst: Path) -> AudioMeta:
    """Transcode audio to the canonical form: 16 kHz mono 16-bit PCM WAV.

    Per 05 §1.2:
        ffmpeg -y -i {src} -ac 1 -ar 16000 -sample_fmt s16 -f wav {dst}
    Then load with soundfile and compute metadata.

    Returns AudioMeta with canonical path, hash, duration, etc.
    Raises AudioValidationError with CORRUPT_AUDIO on failure.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src),
                "-ac", "1",
                "-ar", str(CANONICAL_SAMPLE_RATE),
                "-sample_fmt", "s16",
                "-f", "wav",
                str(dst),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise AudioValidationError(
                "CORRUPT_AUDIO",
                f"ffmpeg transcoding failed: {result.stderr.strip()[:200]}",
            )
    except subprocess.TimeoutExpired:
        raise AudioValidationError("CORRUPT_AUDIO", "Audio transcoding timed out.")
    except FileNotFoundError:
        try:
            y, sr = sf.read(str(src), dtype="float32")
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != CANONICAL_SAMPLE_RATE:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=CANONICAL_SAMPLE_RATE)
            sf.write(str(dst), y, CANONICAL_SAMPLE_RATE, subtype="PCM_16")
        except Exception:
            raise AudioValidationError(
                "CORRUPT_AUDIO",
                "ffmpeg not found. Ensure ffmpeg is installed and on PATH.",
            )

    # Load canonical audio and compute metadata
    y, sr = sf.read(str(dst), dtype="float32")
    assert sr == CANONICAL_SAMPLE_RATE

    # Compute SHA-256 of the canonical PCM bytes
    sha256 = compute_sha256(dst)

    return AudioMeta(
        canonical_path=dst,
        original_path=src,
        sha256=sha256,
        duration_seconds=len(y) / sr,
        sample_rate=sr,
        n_samples=len(y),
        file_size_bytes=dst.stat().st_size,
    )


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_canonical(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load a canonical WAV file as a float32 numpy array.

    Returns (samples, sample_rate).
    """
    y, sr = sf.read(str(audio_path), dtype="float32")
    return y, sr


def normalize_amplitude(y: np.ndarray) -> np.ndarray:
    """Peak normalisation to −1 dBFS per 05 §1.3.

    Applied after transcoding.
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        target = 10 ** (-1.0 / 20.0)  # −1 dBFS
        y = y * target / peak
    return y


def sanitize_filename(filename: str) -> str:
    """Sanitise a user-supplied filename per 14 §3.

    Strips path separators, null bytes, control characters.
    Truncates to 255 characters.
    """
    # Remove path separators and null bytes
    for char in ["/", "\\", "\x00", "..", ":"]:
        filename = filename.replace(char, "_")

    # Remove control characters
    filename = "".join(c for c in filename if ord(c) >= 32)

    # Truncate
    return filename[:255] if filename else "unnamed"


def ingest(
    src: Path,
    work_dir: Path,
    max_mb: int = 25,
    min_seconds: float = 1.5,
    max_seconds: float = 300.0,
) -> AudioMeta:
    """Full ingestion pipeline: validate → transcode → normalise → hash.

    Args:
        src: Path to the uploaded audio file.
        work_dir: Directory for canonical output.
        max_mb: Maximum file size in MB.
        min_seconds: Minimum duration.
        max_seconds: Maximum duration.

    Returns:
        AudioMeta for the canonical file.

    Raises:
        AudioValidationError on any validation failure.
    """
    # 1. Magic byte validation
    validate_magic_bytes(src)

    # 2. Size check
    validate_file_size(src, max_mb)

    # 3. Duration probe
    duration = probe_duration(src)
    validate_duration(duration, min_seconds, max_seconds)

    # 4. Canonicalise
    canonical_path = work_dir / "canonical.wav"
    meta = canonicalize(src, canonical_path)

    # 5. Amplitude normalisation
    y, sr = load_canonical(canonical_path)
    y = normalize_amplitude(y)
    sf.write(str(canonical_path), y, sr, subtype=CANONICAL_SUBTYPE)

    # 6. Recompute hash after normalisation
    meta = AudioMeta(
        canonical_path=canonical_path,
        original_path=src,
        sha256=compute_sha256(canonical_path),
        duration_seconds=len(y) / sr,
        sample_rate=sr,
        n_samples=len(y),
        file_size_bytes=canonical_path.stat().st_size,
    )

    log.info(
        "audio_ingested",
        duration=f"{meta.duration_seconds:.1f}s",
        sha256=meta.sha256[:16],
    )
    return meta
