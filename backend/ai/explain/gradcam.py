"""Grad-CAM explainability for the acoustic classifier.

Per 05 §6.2 - §6.3:
  - Attached to the acoustic CNN only
  - Runs in eval() mode with gradients enabled (explicit assertion against inference_mode)
  - Target layer: conv_head for EfficientNet, layer4[-1] for ResNet
  - Window selection: highest-scoring window + up to 2 windows within 0.1 of max
  - Renders spectrogram.png, heatmap.png, overlay.png
  - Peak-region extraction via connected components
  - Graceful fallback: on ANY error, returns None and logs warning, never fails pipeline

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import structlog
import torch
import torch.nn.functional as F

from ai.acoustic.model import AcousticDeepfakeCNN
from ai.audio.features import log_mel, normalize_spectrogram, segment_windows, spectrogram_to_tensor
from ai.audio.io import load_canonical
from ai.audio.quality import compute_speech_ratio
from ai.base import Component, ExplanationResult
from ai.explain.render import (
    MANDATORY_DISCLAIMER,
    AxisExtents,
    extract_peak_regions,
    render_heatmap,
    render_overlay,
    render_spectrogram,
)

log = structlog.get_logger()


class GradCAM:
    """Core Grad-CAM implementation hooked into a model's target convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "acts", o.detach())
        )
        self.target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grads", go[0].detach())
        )

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Compute Grad-CAM attention heatmap for input tensor x.

        Args:
            x: Input tensor of shape (1, 3, 128, 400).
            class_idx: Target class index (0 = bona fide, 1 = spoof).
                       If None, targets the predicted class (logits.argmax).

        Returns:
            CAM heatmap array of shape (128, 400) with values in [0, 1].
        """
        assert torch.is_grad_enabled(), (
            "Grad-CAM requires gradients enabled. Do not call within torch.inference_mode() "
            "or torch.no_grad()."
        )

        self.model.zero_grad()
        logits = self.model(x)  # (1, 2)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        if self.grads is None or self.acts is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Global average pool of gradients across spatial dimensions (H, W) -> (1, K, 1, 1)
        weights = self.grads.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))

        # Bilinear interpolation to match input spectrogram shape (128, 400)
        cam = F.interpolate(
            cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalize to [0, 1]
        c_min, c_max = float(cam_np.min()), float(cam_np.max())
        if c_max - c_min > 1e-8:
            cam_norm = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm.astype(np.float32)


class GradCAMExplainer(Component):
    """Component providing Grad-CAM visualizations and peak attention regions."""

    def __init__(
        self,
        acoustic_model: AcousticDeepfakeCNN | None = None,
        acoustic_detector: Any | None = None,
        version: str = "gradcam-v0.1.0",
    ) -> None:
        self._name: Final[str] = "explainer"
        self._version = version
        self.model = acoustic_model
        self.detector = acoustic_detector
        self._gradcam: GradCAM | None = None
        self._is_loaded = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        """Initialize hooks on the model's target convolutional layer."""
        if self.model is None and self.detector is not None:
            if hasattr(self.detector, "model") and self.detector.model is not None:
                self.model = self.detector.model
            elif hasattr(self.detector, "load"):
                if not self.detector.is_loaded():
                    self.detector.load()
                self.model = getattr(self.detector, "model", None)

        if self.model is None:
            raise RuntimeError("GradCAMExplainer requires an instantiated AcousticDeepfakeCNN.")

        target_layer = self.model.gradcam_target_layer
        self._gradcam = GradCAM(self.model, target_layer)
        self._is_loaded = True

    def warmup(self) -> None:
        """Warmup hook for the explainer."""
        if not self._is_loaded:
            self.load()

    def explain(
        self,
        audio_path: Path | str,
        target_class: int | None = None,
        output_dir: Path | str | None = None,
        max_windows: int = 3,
    ) -> ExplanationResult | None:
        """Generate Grad-CAM heatmaps and peak regions.

        Per 05 §6.3:
          - Highest-scoring window + up to two windows with score within 0.1 of max.
          - Returns ExplanationResult with image paths, axis extents, and peak regions.
          - On failure: logs warning and returns None; NEVER raises exception to pipeline.

        Args:
            audio_path: Path to canonical WAV audio.
            target_class: Class to explain (0 or 1, default: model prediction).
            output_dir: Output directory for PNG artifacts.
            max_windows: Maximum windows to explain (default 3).

        Returns:
            ExplanationResult or None on failure.
        """
        try:
            if not self._is_loaded or self._gradcam is None:
                self.load()

            path = Path(audio_path)
            out_dir = Path(output_dir) if output_dir else path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            y, sr = load_canonical(path)
            raw_windows = segment_windows(y, sr=sr, window_seconds=4.0, hop_seconds=2.0)

            valid_windows = []
            for w_audio, t_start, t_end in raw_windows:
                if compute_speech_ratio(w_audio, sr=sr) >= 0.15:
                    valid_windows.append((w_audio, t_start, t_end))

            if not valid_windows:
                valid_windows = [raw_windows[0]]

            # Window inference to obtain scores
            window_data = []
            assert self.model is not None

            self.model.eval()
            for idx, (w_audio, t_start, t_end) in enumerate(valid_windows):
                mel = log_mel(w_audio, sr=sr)
                norm = normalize_spectrogram(mel)
                tensor = spectrogram_to_tensor(norm)
                x = torch.from_numpy(tensor).unsqueeze(0)

                with torch.no_grad():
                    logits = self.model(x)
                    prob_spoof = float(F.softmax(logits, dim=-1)[0, 1].item())
                    pred_class = int(logits.argmax(dim=1).item())

                window_data.append({
                    "index": idx,
                    "audio": w_audio,
                    "mel": mel,
                    "tensor": x,
                    "t_start": t_start,
                    "t_end": t_end,
                    "score": prob_spoof,
                    "pred_class": pred_class,
                })

            if not window_data:
                return None

            # Window selection per 05 §6.3:
            # Highest scoring window + up to 2 within 0.1 of max
            window_data.sort(key=lambda w: w["score"], reverse=True)
            max_score = window_data[0]["score"]
            selected_windows = [window_data[0]]

            for w in window_data[1:]:
                if len(selected_windows) >= max_windows:
                    break
                if abs(max_score - w["score"]) <= 0.10:
                    selected_windows.append(w)

            # Sort selected windows chronologically for consistent display
            selected_windows.sort(key=lambda w: w["t_start"])

            explained_windows_meta = []
            global_extents: dict[str, float] = {}

            # Render CAM for selected windows with gradients enabled
            assert self._gradcam is not None
            with torch.enable_grad():
                for win in selected_windows:
                    w_idx = win["index"]
                    x_input = win["tensor"].clone().requires_grad_(True)
                    cls_to_explain = target_class if target_class is not None else win["pred_class"]

                    cam = self._gradcam(x_input, class_idx=cls_to_explain)

                    prefix = f"{path.stem}_w{w_idx}"
                    spec_png = out_dir / f"{prefix}_spectrogram.png"
                    heat_png = out_dir / f"{prefix}_heatmap.png"
                    overlay_png = out_dir / f"{prefix}_overlay.png"

                    # Render PNGs per 05 §6.4
                    extents = render_spectrogram(
                        win["mel"],
                        spec_png,
                        sr=sr,
                        f_min=20.0,
                        f_max=8000.0,
                    )
                    render_heatmap(
                        cam,
                        heat_png,
                        t_start=win["t_start"],
                        t_end=win["t_end"],
                        f_min=20.0,
                        f_max=8000.0,
                    )
                    render_overlay(
                        win["mel"][:, :400],
                        cam,
                        overlay_png,
                        t_start=win["t_start"],
                        t_end=win["t_end"],
                        f_min=20.0,
                        f_max=8000.0,
                    )

                    global_extents = extents.to_dict()

                    # Peak region extraction
                    peaks = extract_peak_regions(
                        cam,
                        t_start=win["t_start"],
                        t_end=win["t_end"],
                        f_min=20.0,
                        f_max=8000.0,
                        threshold=0.7,
                    )

                    explained_windows_meta.append({
                        "window_index": w_idx,
                        "t_start_s": win["t_start"],
                        "t_end_s": win["t_end"],
                        "start_time_s": win["t_start"],
                        "end_time_s": win["t_end"],
                        "score": win["score"],
                        "spectrogram_path": str(spec_png.resolve()),
                        "heatmap_path": str(heat_png.resolve()),
                        "overlay_path": str(overlay_png.resolve()),
                        "axis_extents": global_extents,
                        "peak_regions": [p.to_dict() for p in peaks],
                        "peak_descriptions": [p.describe() for p in peaks],
                    })

            target_layer_name = self.model.gradcam_target_layer.__class__.__name__
            first_cls = target_class if target_class is not None else selected_windows[0]["pred_class"]

            return ExplanationResult(
                method="grad-cam",
                target_layer=target_layer_name,
                target_class=first_cls,
                windows=explained_windows_meta,
                axis_extents=global_extents,
                disclaimer=MANDATORY_DISCLAIMER,
            )

        except Exception as e:
            log.warning("gradcam_explanation_failed", error=str(e))
            return None
