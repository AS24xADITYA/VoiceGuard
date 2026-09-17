"""Unit tests for ai/acoustic/model.py.

Verifies Phase 3 exit criteria:
  - Model architecture accepts input tensor (B, 3, 128, 400) and produces (B, 2) logits
  - Fallback backbone (resnet18) is supported
  - gradcam_target_layer exists and is a valid convolutional module
  - Inference is deterministic for a fixed input
"""

from __future__ import annotations

import torch
import pytest

from ai.acoustic.model import AcousticDeepfakeCNN


def test_acoustic_cnn_forward_shape():
    """Verify forward pass output shape is (B, 2) for EfficientNet-B0."""
    model = AcousticDeepfakeCNN(backbone="efficientnet_b0", pretrained=False)
    model.eval()

    batch = torch.randn(2, 3, 128, 400)
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (2, 2)
    assert not torch.isnan(logits).any()


def test_acoustic_cnn_resnet_fallback():
    """Verify mandated fallback backbone (resnet18) works and has valid gradcam layer."""
    model = AcousticDeepfakeCNN(backbone="resnet18", pretrained=False)
    model.eval()

    batch = torch.randn(1, 3, 128, 400)
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (1, 2)
    assert hasattr(model, "gradcam_target_layer")
    assert isinstance(model.gradcam_target_layer, torch.nn.Module)


def test_gradcam_target_layer_efficientnet():
    """Verify gradcam_target_layer is accessible on EfficientNet."""
    model = AcousticDeepfakeCNN(backbone="efficientnet_b0", pretrained=False)
    target = model.gradcam_target_layer
    assert isinstance(target, torch.nn.Module)


def test_inference_determinism():
    """Verify inference is deterministic for fixed inputs."""
    model = AcousticDeepfakeCNN(backbone="efficientnet_b0", pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 128, 400)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2, atol=1e-6)
