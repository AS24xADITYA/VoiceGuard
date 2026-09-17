"""Spectrogram and Grad-CAM visualization rendering.

Per 05 §6.4:
  - Output 1: spectrogram.png — log-mel, viridis, labelled time (s) and freq (Hz)
  - Output 2: heatmap.png — Grad-CAM, inferno colormap, same axes
  - Output 3: overlay.png — spectrogram grayscale base + heatmap at alpha 0.45,
                            colorbar with 'model attention (relative)' label

Matplotlib with the Agg backend, 150 dpi, fixed figure size.
Axis extents returned as structured data for the React viewer.
Peak region extraction via connected components on thresholded CAM.

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# ── Mandatory disclaimer per 05 §6.5 ────────────────────────────────
MANDATORY_DISCLAIMER: Final[str] = (
    "This visualisation shows which regions of the audio spectrogram most influenced "
    "the model's classification. It indicates where the model attended, not where "
    "manipulation definitively occurred. Attention maps on audio spectrograms can highlight "
    "recording artefacts, silence, or channel characteristics rather than synthesis artefacts."
)

FIGURE_SIZE: Final[tuple[float, float]] = (10.0, 4.0)
FIGURE_DPI: Final[int] = 150


@dataclass(frozen=True)
class AxisExtents:
    """Axis extents for overlaying coordinates in UI."""

    t_min: float
    t_max: float
    f_min: float
    f_max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class PeakRegion:
    """A high-attention region in the spectrogram."""

    t_start_s: float
    t_end_s: float
    f_low_hz: float
    f_high_hz: float
    mean_weight: float

    def describe(self) -> str:
        """Textual description for accessibility and UI summary."""
        return (
            f"attention concentrated at {self.t_start_s:.1f}–{self.t_end_s:.1f} s "
            f"in the {self.f_low_hz:.0f}–{self.f_high_hz:.0f} Hz band "
            f"(weight: {self.mean_weight:.2f})"
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def render_spectrogram(
    s_db: np.ndarray,
    output_path: Path | str,
    sr: int = 16000,
    hop_length: int = 160,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    title: str | None = "Log-Mel Spectrogram",
) -> AxisExtents:
    """Render log-mel spectrogram with viridis colormap and labeled axes.

    Args:
        s_db: Log-mel spectrogram array of shape (N_MELS, T).
        output_path: Path to write the PNG file.
        sr: Sample rate.
        hop_length: Hop length in samples.
        f_min: Minimum frequency (Hz).
        f_max: Maximum frequency (Hz).
        title: Optional title string.

    Returns:
        AxisExtents dataclass with physical coordinates.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_mels, t_frames = s_db.shape
    duration_s = (t_frames * hop_length) / sr

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    extent = [0.0, duration_s, f_min, f_max]

    im = ax.imshow(
        s_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        interpolation="bilinear",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title, fontsize=11, fontweight="medium")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Power (dB)")

    plt.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)

    return AxisExtents(t_min=0.0, t_max=duration_s, f_min=f_min, f_max=f_max)


def render_waveform(
    y: np.ndarray,
    output_path: Path | str,
    sr: int = 16000,
    title: str | None = "Waveform",
) -> AxisExtents:
    """Render audio waveform with amplitude [-1, 1] and labeled time axis.

    Args:
        y: 1D audio samples.
        output_path: Path to write the PNG file.
        sr: Sample rate.
        title: Optional title string.

    Returns:
        AxisExtents dataclass.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    duration_s = len(y) / sr
    time_axis = np.linspace(0.0, duration_s, num=len(y))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    ax.plot(time_axis, y, color="#22D3EE", linewidth=0.6, alpha=0.9)
    ax.set_xlim(0.0, duration_s)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle="--", alpha=0.3)
    if title:
        ax.set_title(title, fontsize=11, fontweight="medium")

    plt.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)

    return AxisExtents(t_min=0.0, t_max=duration_s, f_min=-1.0, f_max=1.0)


def render_heatmap(
    cam: np.ndarray,
    output_path: Path | str,
    t_start: float = 0.0,
    t_end: float = 4.0,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    title: str | None = "Grad-CAM Attention Heatmap",
) -> AxisExtents:
    """Render Grad-CAM attention heatmap with inferno colormap.

    Args:
        cam: Grad-CAM attention map in [0, 1] of shape (128, W).
        output_path: Path to write the PNG file.
        t_start: Window start time in seconds.
        t_end: Window end time in seconds.
        f_min: Minimum frequency (Hz).
        f_max: Maximum frequency (Hz).
        title: Optional title string.

    Returns:
        AxisExtents dataclass.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    extent = [t_start, t_end, f_min, f_max]

    im = ax.imshow(
        cam,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
        interpolation="bilinear",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title, fontsize=11, fontweight="medium")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Attention Weight")

    plt.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)

    return AxisExtents(t_min=t_start, t_max=t_end, f_min=f_min, f_max=f_max)


def render_overlay(
    s_db: np.ndarray,
    cam: np.ndarray,
    output_path: Path | str,
    t_start: float = 0.0,
    t_end: float = 4.0,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    alpha: float = 0.45,
    title: str | None = "Spectrogram with Grad-CAM Attention Overlay",
) -> AxisExtents:
    """Render grayscale spectrogram base with color attention heatmap overlay.

    Per 05 §6.4:
    - Base: grayscale spectrogram
    - Overlay: inferno colormap at alpha 0.45
    - Colorbar label: 'model attention (relative)'

    Returns:
        AxisExtents dataclass.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    extent = [t_start, t_end, f_min, f_max]

    # Base: grayscale spectrogram
    ax.imshow(
        s_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="gray",
        interpolation="bilinear",
    )

    # Overlay: heatmap at alpha=0.45
    im_cam = ax.imshow(
        cam,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
        alpha=alpha,
        interpolation="bilinear",
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title, fontsize=11, fontweight="medium")

    cbar = fig.colorbar(im_cam, ax=ax, pad=0.02)
    cbar.set_label("model attention (relative)")

    plt.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI)
    plt.close(fig)

    return AxisExtents(t_min=t_start, t_max=t_end, f_min=f_min, f_max=f_max)


def extract_peak_regions(
    cam: np.ndarray,
    t_start: float = 0.0,
    t_end: float = 4.0,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    threshold: float = 0.7,
    min_pixels: int = 10,
) -> list[PeakRegion]:
    """Extract salient attention regions from Grad-CAM map.

    Per 05 §6.4:
    - Threshold the CAM at 0.7
    - Find connected components
    - Return each component's bounding box converted to
      (t_start_s, t_end_s, f_low_hz, f_high_hz, mean_weight).

    Args:
        cam: Grad-CAM 2D array of shape (N_MELS, T_FRAMES) in [0, 1].
        t_start: Window start time in seconds.
        t_end: Window end time in seconds.
        f_min: Minimum frequency (Hz).
        f_max: Maximum frequency (Hz).
        threshold: CAM intensity threshold (default 0.7).
        min_pixels: Minimum pixel area to exclude isolated noisy pixels.

    Returns:
        List of PeakRegion objects sorted by mean_weight descending.
    """
    binary_mask = cam >= threshold
    if not np.any(binary_mask):
        return []

    labeled_array, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return []

    n_mels, n_frames = cam.shape
    dt = (t_end - t_start) / max(n_frames, 1)
    df = (f_max - f_min) / max(n_mels, 1)

    regions: list[PeakRegion] = []
    slices = ndimage.find_objects(labeled_array)

    for i, s in enumerate(slices):
        if s is None:
            continue
        mel_slice, frame_slice = s
        component_mask = labeled_array[s] == (i + 1)
        pixel_count = np.sum(component_mask)
        if pixel_count < min_pixels:
            continue

        region_cam = cam[s]
        mean_val = float(np.mean(region_cam[component_mask]))

        # Mel index 0 is f_min, index n_mels-1 is f_max
        f_low = f_min + mel_slice.start * df
        f_high = f_min + mel_slice.stop * df
        t_s = t_start + frame_slice.start * dt
        t_e = t_start + frame_slice.stop * dt

        regions.append(
            PeakRegion(
                t_start_s=round(t_s, 2),
                t_end_s=round(t_e, 2),
                f_low_hz=round(f_low, 1),
                f_high_hz=round(f_high, 1),
                mean_weight=round(mean_val, 3),
            )
        )

    # Sort strongest attention first
    regions.sort(key=lambda r: r.mean_weight, reverse=True)
    return regions
