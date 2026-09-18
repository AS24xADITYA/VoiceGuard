"""Unit tests for ai/evaluation/run_eval.py degradation transforms and pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ai.evaluation.run_eval import (
    apply_additive_noise_snr,
    apply_mulaw_codec,
    run_full_evaluation,
)
from ai.acoustic.model import AcousticDeepfakeCNN


def test_mulaw_codec_transform():
    """Verify 8 kHz mu-law transform preserves length and handles signals."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    degraded = apply_mulaw_codec(y, sr=sr)
    assert len(degraded) == len(y)
    assert degraded.dtype == np.float32
    assert -1.0 <= np.min(degraded) <= 1.0
    assert -1.0 <= np.max(degraded) <= 1.0
    # Degradation introduces quantization noise; output shouldn't be identical
    assert not np.allclose(y, degraded, atol=1e-3)


def test_additive_noise_transform():
    """Verify additive noise at target SNR operates correctly."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    noisy = apply_additive_noise_snr(y, snr_db=10.0)
    assert len(noisy) == len(y)
    assert noisy.dtype == np.float32
    # Verify noise changed the signal
    diff_power = np.mean((noisy - y) ** 2)
    assert diff_power > 0.0


def test_full_evaluation_pipeline_mock(tmp_path: Path):
    """Verify run_full_evaluation runs end-to-end on synthetic manifests."""
    import soundfile as sf

    # 1. Create dummy model checkpoint
    model = AcousticDeepfakeCNN(backbone="efficientnet_b0", pretrained=False)
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    # 2. Create dummy audio files
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sr = 16000
    samples_c1 = []
    samples_c2 = []

    for i in range(4):
        p = audio_dir / f"c1_{i}.wav"
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(str(p), (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr)
        samples_c1.append({"path": str(p), "label": i % 2, "attack_id": f"A0{i+1}" if i % 2 == 1 else "-"})

    for i in range(4):
        p = audio_dir / f"c2_{i}.wav"
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(str(p), (0.2 * np.sin(2 * np.pi * 500 * t)).astype(np.float32), sr)
        samples_c2.append({"path": str(p), "label": i % 2, "attack_id": "ITW_fake" if i % 2 == 1 else "-"})

    c1_manifest_path = tmp_path / "c1_manifest.json"
    c2_manifest_path = tmp_path / "c2_manifest.json"
    with open(c1_manifest_path, "w") as f:
        json.dump(samples_c1, f)
    with open(c2_manifest_path, "w") as f:
        json.dump(samples_c2, f)

    out_metrics = tmp_path / "metrics.json"

    # Run evaluation
    res = run_full_evaluation(
        checkpoint_path=ckpt_path,
        c1_manifest_path=c1_manifest_path,
        c2_manifest_path=c2_manifest_path,
        output_metrics_path=out_metrics,
        backbone="efficientnet_b0",
        device_str="cpu",
        batch_size=2,
    )

    assert out_metrics.is_file()
    assert "c1_in_domain" in res["conditions"]
    assert "c2_out_of_domain" in res["conditions"]
    assert "c3_codec_degraded" in res["conditions"]
    assert "c4_noise_degraded_10db" in res["conditions"]
    assert "c1_to_c2_eer_gap" in res["generalization_gap"]
