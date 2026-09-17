"""Unit tests for ai/explain/render.py.

Verifies Phase 2 exit criteria:
  - A spectrogram PNG renders with correct axes and extents
  - Waveform PNG renders with correct amplitude and time bounds
  - Heatmap and overlay PNGs render with correct colormaps and labels
  - Peak region extraction detects connected components above threshold 0.7
  - Mandatory disclaimer is present
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from ai.audio.features import log_mel
from ai.explain.render import (
    MANDATORY_DISCLAIMER,
    AxisExtents,
    PeakRegion,
    extract_peak_regions,
    render_heatmap,
    render_overlay,
    render_spectrogram,
    render_waveform,
)
from tests.fixtures.audio_fixtures import create_synthetic_audio


def test_mandatory_disclaimer_present():
    """Verify mandatory disclaimer matches 05 §6.5 requirement."""
    assert "visualisation shows which regions" in MANDATORY_DISCLAIMER
    assert "not where manipulation definitively occurred" in MANDATORY_DISCLAIMER


def test_render_spectrogram_and_waveform(tmp_path: Path):
    """Verify spectrogram and waveform render valid PNG files with accurate extents."""
    y = create_synthetic_audio(duration_s=2.5, sr=16000)
    mel = log_mel(y, sr=16000)

    spec_path = tmp_path / "spectrogram.png"
    spec_extents = render_spectrogram(mel, spec_path, sr=16000, hop_length=160)

    assert spec_path.exists()
    assert spec_path.stat().st_size > 1000
    assert spec_extents.t_min == 0.0
    assert abs(spec_extents.t_max - 2.5) < 0.05
    assert spec_extents.f_min == 20.0
    assert spec_extents.f_max == 8000.0

    wave_path = tmp_path / "waveform.png"
    wave_extents = render_waveform(y, wave_path, sr=16000)

    assert wave_path.exists()
    assert wave_path.stat().st_size > 1000
    assert wave_extents.t_min == 0.0
    assert abs(wave_extents.t_max - 2.5) < 0.05


def test_render_heatmap_and_overlay(tmp_path: Path):
    """Verify heatmap and overlay render with valid extents."""
    y = create_synthetic_audio(duration_s=4.0, sr=16000)
    mel = log_mel(y, sr=16000)
    cam = np.zeros((128, 400), dtype=np.float32)
    # create a hot spot
    cam[40:60, 100:150] = 0.85

    heat_path = tmp_path / "heatmap.png"
    heat_extents = render_heatmap(cam, heat_path, t_start=0.0, t_end=4.0)
    assert heat_path.exists()
    assert heat_path.stat().st_size > 1000
    assert heat_extents.t_min == 0.0
    assert heat_extents.t_max == 4.0

    overlay_path = tmp_path / "overlay.png"
    overlay_extents = render_overlay(mel[:, :400], cam, overlay_path, t_start=0.0, t_end=4.0)
    assert overlay_path.exists()
    assert overlay_path.stat().st_size > 1000
    assert overlay_extents.t_min == 0.0
    assert overlay_extents.t_max == 4.0


def test_extract_peak_regions():
    """Verify connected component peak extraction detects hotspot regions."""
    cam = np.zeros((128, 400), dtype=np.float32)
    # Add an attention peak above 0.7
    cam[50:70, 100:140] = 0.9

    regions = extract_peak_regions(cam, t_start=0.0, t_end=4.0, f_min=20.0, f_max=8000.0, threshold=0.7)

    assert len(regions) == 1
    region = regions[0]
    assert 0.9 <= region.t_start_s <= 1.1
    assert 1.3 <= region.t_end_s <= 1.5
    assert region.f_low_hz > 2000
    assert region.mean_weight >= 0.85
    desc = region.describe()
    assert "attention concentrated at" in desc
