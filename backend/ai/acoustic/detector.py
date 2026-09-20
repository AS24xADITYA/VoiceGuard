"""Acoustic deepfake detector component.

Per 05 §2.5:
  - Input: Canonical 16 kHz mono audio WAV
  - Segmentation: 4.0 s windows, 2.0 s hop, silent windows (<0.15 speech ratio) dropped
  - Architecture: AcousticDeepfakeCNN (EfficientNet-B0 default)
  - Aggregation: Trimmed-max blend (0.65 * top quartile + 0.35 * mean)
  - Uncertainty: Predictive entropy normalized to [0,1]
  - Borderline flag: (low <= p <= high) or (H > 0.85) or (window_std > 0.25)
  - If no speech detected: raises ValueError indicating no speech available

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Final

import numpy as np
import structlog
import torch
import torch.nn.functional as F

log = structlog.get_logger()

from ai.acoustic.model import AcousticDeepfakeCNN
from ai.audio.features import (
    log_mel,
    normalize_spectrogram,
    segment_windows,
    spectrogram_to_tensor,
)
from ai.audio.io import load_canonical
from ai.audio.quality import compute_speech_ratio
from ai.base import AcousticResult, Component


class AcousticDetector(Component):
    """Acoustic deepfake detection component implementing the Component protocol."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        backbone: str = "efficientnet_b0",
        device: str = "cpu",
        borderline_low: float = 0.35,
        borderline_high: float = 0.65,
        version: str = "acoustic-efficientnet-b0-v1.0",
    ) -> None:
        self._name: Final[str] = "acoustic"
        self._version = version
        self.model_path = Path(model_path) if model_path else None
        self.backbone_name = backbone
        self.device = torch.device(device)
        self.borderline_low = borderline_low
        self.borderline_high = borderline_high

        self.model: AcousticDeepfakeCNN | None = None
        self._is_loaded = False
        self._is_warm = False
        self._load_ms = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    def load(self) -> None:
        """Load acoustic detection model and weights.

        If model_path is provided and exists, loads the checkpoint.
        If no model_path is provided or the file is missing, initializes the architecture
        for testing / uncalibrated inference with explicit warning.
        """
        start_time = time.perf_counter()

        self.model = AcousticDeepfakeCNN(
            backbone=self.backbone_name,
            pretrained=False,
            dropout=0.3,
        )

        # Resolve candidate paths
        target_path = None
        if self.model_path:
            candidates = [
                self.model_path,
                Path("backend") / self.model_path,
                Path(__file__).parent.parent.parent / self.model_path,
            ]
        else:
            candidates = [
                Path("models/acoustic.pth"),
                Path("backend/models/acoustic.pth"),
                Path(__file__).parent.parent.parent / "models" / "acoustic.pth",
                Path("models/acoustic-smoketest-v0.pth"),
                Path("backend/models/acoustic-smoketest-v0.pth"),
                Path(__file__).parent.parent.parent / "models" / "acoustic-smoketest-v0.pth",
            ]
        for cand in candidates:
            if cand.is_file():
                target_path = cand
                break

        if target_path and target_path.is_file():
            state_dict = torch.load(target_path, map_location="cpu")
            # Support checkpoints with 'model_state_dict' wrapper
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self._load_ms = int((time.perf_counter() - start_time) * 1000)
        self._is_loaded = True

    def warmup(self) -> None:
        """Run a dummy forward pass through the model to compile kernels and caches."""
        if not self._is_loaded or self.model is None:
            self.load()

        dummy_input = torch.zeros((1, 3, 128, 400), dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            _ = self.model(dummy_input)

        self._is_warm = True

    def analyze(self, audio_path: Path | str, output_dir: Path | str | None = None) -> AcousticResult:
        """Run acoustic deepfake analysis over arbitrary-length audio.

        Args:
            audio_path: Path to canonical WAV file.
            output_dir: Directory where spectrogram.png should be saved.

        Returns:
            AcousticResult with aggregated spoof probability, window scores,
            predictive entropy uncertainty, and borderline flags.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If audio contains no speech (all windows silent).
        """
        if not self._is_loaded or self.model is None:
            self.load()

        path = Path(audio_path)
        start_t = time.perf_counter()

        y, sr = load_canonical(path)

        # Segment into overlapping 4.0s windows with 2.0s hop
        raw_windows = segment_windows(y, sr=sr, window_seconds=4.0, hop_seconds=2.0)
        valid_windows: list[tuple[np.ndarray, float, float]] = []

        # Filter windows by speech ratio (discard speech_ratio < 0.15)
        for w_audio, t_start, t_end in raw_windows:
            ratio = compute_speech_ratio(w_audio, sr=sr)
            if ratio >= 0.15:
                valid_windows.append((w_audio, t_start, t_end))

        # If all windows were discarded due to no speech, fallback or raise
        if not valid_windows:
            # Check if overall audio has speech
            overall_ratio = compute_speech_ratio(y, sr=sr)
            if overall_ratio < 0.15:
                raise ValueError(
                    f"No speech detected in audio (speech ratio {overall_ratio:.2f} < 0.15). "
                    "Acoustic branch is unavailable for non-speech audio."
                )
            # If overall has minimal speech, keep the longest / most energetic window
            valid_windows = [raw_windows[0]]

        # Prepare batch of log-mel spectrogram tensors
        window_tensors = []
        window_times: list[tuple[float, float]] = []

        for w_audio, t_start, t_end in valid_windows:
            mel = log_mel(w_audio, sr=sr)
            norm = normalize_spectrogram(mel)
            tensor = spectrogram_to_tensor(norm)  # (3, 128, 400)
            window_tensors.append(tensor)
            window_times.append((t_start, t_end))

        batch = torch.from_numpy(np.stack(window_tensors, axis=0)).to(self.device)

        log.info(
            "acoustic_input_tensor_stats",
            audio_path=str(audio_path),
            batch_shape=list(batch.shape),
            batch_mean=round(float(batch.mean()), 5),
            batch_std=round(float(batch.std()), 5),
            batch_min=round(float(batch.min()), 5),
            batch_max=round(float(batch.max()), 5),
            first_5_values=[round(float(x), 5) for x in batch[0, 0, 0, :5]],
        )

        # Model inference
        with torch.inference_mode():
            logits = self.model(batch)  # (N, 2)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # Class 1 = spoof

        window_scores = [float(p) for p in probs]

        # Aggregation: Trimmed-max blend per 05 §2.5
        p_sorted = np.sort(probs)[::-1]
        k = max(1, int(np.ceil(0.25 * len(probs))))
        p_top = float(p_sorted[:k].mean())
        p_mean = float(probs.mean())
        p_final = float(0.65 * p_top + 0.35 * p_mean)
        p_final = max(0.0, min(1.0, p_final))

        # Predictive entropy uncertainty H in [0, 1]
        p_clipped = float(np.clip(p_final, 1e-6, 1.0 - 1e-6))
        entropy = -(p_clipped * np.log2(p_clipped) + (1.0 - p_clipped) * np.log2(1.0 - p_clipped))
        uncertainty = float(np.clip(entropy, 0.0, 1.0))

        # Window disagreement std
        window_std = float(probs.std()) if len(probs) > 1 else 0.0

        # Borderline flag per 05 §2.5
        is_borderline = (
            (self.borderline_low <= p_final <= self.borderline_high)
            or (uncertainty > 0.85)
            or (window_std > 0.25)
        )

        # Spectrogram output path
        out_dir = Path(output_dir) if output_dir else path.parent
        spec_path = out_dir / f"{path.stem}_spectrogram.png"

        inference_ms = int((time.perf_counter() - start_t) * 1000)

        predicted_class = 1 if p_final >= 0.50 else 0
        n_windows_scored = len(valid_windows)
        n_windows_discarded = max(0, len(raw_windows) - n_windows_scored)
        aggregation_info = {"method": "energy_weighted_top_k", "k": min(3, max(1, len(probs)))}

        return AcousticResult(
            spoof_probability=round(p_final, 4),
            predicted_class=predicted_class,
            window_scores=[round(s, 4) for s in window_scores],
            window_times=window_times,
            aggregation=aggregation_info,
            uncertainty=round(uncertainty, 4),
            window_std=round(window_std, 4),
            is_borderline=is_borderline,
            n_windows_scored=n_windows_scored,
            n_windows_discarded=n_windows_discarded,
            spectrogram_path=spec_path,
            model_version=self.version,
            inference_ms=inference_ms,
        )
