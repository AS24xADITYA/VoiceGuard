"""Unit tests for ai/explain/gradcam.py.

Verifies Phase 4 exit criteria:
  - CAM output shape matches input spectrogram (128, 400)
  - CAM values lie in [0, 1] and are not constant
  - Both classes (0: bona fide, 1: spoof) can be explained
  - Calling without gradients enabled raises AssertionError
  - GradCAMExplainer.explain() returns valid ExplanationResult
  - Forced internal failure degrades gracefully to None without crashing
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from ai.acoustic.model import AcousticDeepfakeCNN
from ai.base import ExplanationResult
from ai.explain.gradcam import GradCAM, GradCAMExplainer
from tests.fixtures.audio_fixtures import create_synthetic_audio, write_wav_file


def test_gradcam_shape_and_range():
    """Verify CAM output shape is exactly (128, 400) and bounded in [0, 1]."""
    model = AcousticDeepfakeCNN(backbone="efficientnet_b0", pretrained=False)
    model.eval()

    gradcam = GradCAM(model, model.gradcam_target_layer)
    x = torch.randn(1, 3, 128, 400, requires_grad=True)

    with torch.enable_grad():
        cam = gradcam(x, class_idx=1)

    assert cam.shape == (128, 400)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0
    # Values must not be all identical
    assert cam.max() > cam.min()


def test_gradcam_explains_both_classes():
    """Verify Grad-CAM can compute explanations for both class 0 and class 1."""
    model = AcousticDeepfakeCNN(backbone="efficientnet_b0", pretrained=False)
    model.eval()

    gradcam = GradCAM(model, model.gradcam_target_layer)
    x = torch.randn(1, 3, 128, 400, requires_grad=True)

    with torch.enable_grad():
        cam_0 = gradcam(x, class_idx=0)
        cam_1 = gradcam(x, class_idx=1)

    assert cam_0.shape == (128, 400)
    assert cam_1.shape == (128, 400)
    assert not np.isnan(cam_0).any()
    assert not np.isnan(cam_1).any()


def test_gradcam_asserts_gradients_enabled():
    """Verify Grad-CAM explicitly raises AssertionError if called without gradients."""
    model = AcousticDeepfakeCNN(backbone="efficientnet_b0", pretrained=False)
    model.eval()

    gradcam = GradCAM(model, model.gradcam_target_layer)
    x = torch.randn(1, 3, 128, 400)

    with torch.no_grad():
        with pytest.raises(AssertionError) as exc_info:
            gradcam(x, class_idx=1)

    assert "Grad-CAM requires gradients enabled" in str(exc_info.value)


def test_explainer_end_to_end(tmp_path: Path):
    """Verify GradCAMExplainer generates ExplanationResult with image artifacts."""
    wav_path = tmp_path / "speech.wav"
    y = create_synthetic_audio(duration_s=4.5, sr=16000)
    write_wav_file(wav_path, y)

    model = AcousticDeepfakeCNN(backbone="efficientnet_b0", pretrained=False)
    explainer = GradCAMExplainer(acoustic_model=model)
    explainer.load()

    result = explainer.explain(wav_path, output_dir=tmp_path)

    assert result is not None
    assert isinstance(result, ExplanationResult)
    assert result.method == "grad-cam"
    assert len(result.windows) >= 1
    assert "visualisation shows which regions" in result.disclaimer

    first_win = result.windows[0]
    assert Path(first_win["spectrogram_path"]).exists()
    assert Path(first_win["heatmap_path"]).exists()
    assert Path(first_win["overlay_path"]).exists()


def test_explainer_graceful_degradation():
    """Verify forced failure gracefully returns None without throwing."""
    # Explainer without a model should fail gracefully to None
    explainer = GradCAMExplainer(acoustic_model=None)
    result = explainer.explain("nonexistent_path.wav")
    assert result is None
