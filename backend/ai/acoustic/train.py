"""Acoustic deepfake CNN training pipeline.

Per 05 §2.4 and 06 §2:
  - Full loop with Automatic Mixed Precision (AMP)
  - Checkpoint and resume capability (epoch, model, optimizer, scaler, val_eer)
  - Waveform and spectrogram augmentations (SpecAugment, noise, gain, speed)
  - Class-weighted CrossEntropyLoss with label smoothing (0.05)
  - Early stopping on validation Equal Error Rate (EER) with patience=6
  - Deterministic seeding (seed=42)

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from ai.acoustic.model import AcousticDeepfakeCNN
from ai.audio.features import HOP_LENGTH, N_MELS, WINDOW_FRAMES, log_mel, normalize_spectrogram, spectrogram_to_tensor
from ai.evaluation.metrics import compute_all_metrics, compute_eer


def seed_everything(seed: int = 42) -> None:
    """Ensure reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Augmentation implementations per 05 §2.4 ─────────────────────────


def apply_waveform_gain(y: np.ndarray, max_db: float = 6.0) -> np.ndarray:
    """Apply random gain adjustment within [-max_db, +max_db]."""
    gain_db = np.random.uniform(-max_db, max_db)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return np.clip(y * gain_linear, -1.0, 1.0)


def apply_additive_noise(y: np.ndarray, min_snr: float = 5.0, max_snr: float = 25.0) -> np.ndarray:
    """Add white / pink noise at random SNR uniform in [5, 25] dB."""
    target_snr_db = np.random.uniform(min_snr, max_snr)
    signal_power = np.mean(y ** 2)
    if signal_power <= 1e-10:
        return y
    target_noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(target_noise_power), size=len(y)).astype(np.float32)
    return np.clip(y + noise, -1.0, 1.0)


def apply_spec_augment(
    spec: np.ndarray,
    freq_masks: int = 2,
    freq_width: int = 16,
    time_masks: int = 2,
    time_width: int = 40,
) -> np.ndarray:
    """Apply SpecAugment frequency and time masking to spectrogram in-place."""
    spec_aug = spec.copy()
    n_mels, t_frames = spec_aug.shape

    # Frequency masking
    for _ in range(freq_masks):
        f = np.random.randint(0, freq_width)
        f0 = np.random.randint(0, max(1, n_mels - f))
        spec_aug[f0 : f0 + f, :] = 0.0

    # Time masking
    for _ in range(time_masks):
        t = np.random.randint(0, time_width)
        t0 = np.random.randint(0, max(1, t_frames - t))
        spec_aug[:, t0 : t0 + t] = 0.0

    return spec_aug


# ── Dataset & Manifest Loader ─────────────────────────────────────────


class AudioSpectrogramDataset(Dataset):
    """PyTorch Dataset yielding 3-channel spectrogram tensors and binary labels."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        item = self.samples[idx]
        label = int(item["label"])  # 0: bona fide, 1: spoof

        # Check if precomputed spectrogram array is available
        if "npy_path" in item and Path(item["npy_path"]).is_file():
            s_norm = np.load(item["npy_path"])
        else:
            # Load audio on the fly
            import soundfile as sf

            y, sr = sf.read(item["path"], dtype="float32")
            if self.augment:
                if np.random.rand() < 0.5:
                    y = apply_waveform_gain(y)
                if np.random.rand() < 0.4:
                    y = apply_additive_noise(y)

            # Ensure 4.0s window length (64,000 samples)
            target_len = int(4.0 * 16000)
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)), mode="reflect")
            elif len(y) > target_len:
                start = np.random.randint(0, len(y) - target_len) if self.augment else 0
                y = y[start : start + target_len]

            mel = log_mel(y, sr=16000)
            s_norm = normalize_spectrogram(mel)

        if self.augment and np.random.rand() < 0.5:
            s_norm = apply_spec_augment(s_norm)

        tensor = spectrogram_to_tensor(s_norm)  # (3, 128, 400)
        return torch.from_numpy(tensor), label


# ── Training Loop & Checkpointing ─────────────────────────────────────


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one epoch of training with AMP."""
    model.train()
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == "cuda")):
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(y_batch)

    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate model and compute validation loss, EER, and accuracy."""
    model.eval()
    all_scores: list[float] = []
    all_labels: list[int] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        logits = model(x_batch)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        all_scores.extend(probs)
        all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)

    eer, threshold = compute_eer(y_true, y_score)
    preds = (y_score >= threshold).astype(int)
    acc = float(np.mean(preds == y_true))

    return eer, threshold, acc


def run_training(
    train_manifest: list[dict[str, Any]],
    val_manifest: list[dict[str, Any]],
    output_dir: Path | str,
    backbone: str = "efficientnet_b0",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 6,
    resume_checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    """Execute full acoustic model training loop."""
    seed_everything(42)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets & loaders
    train_ds = AudioSpectrogramDataset(train_manifest, augment=True)
    val_ds = AudioSpectrogramDataset(val_manifest, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" else 2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0 if os.name == "nt" else 2,
    )

    # Class balance weights
    train_labels = [s["label"] for s in train_manifest]
    n_bonafide = sum(1 for l in train_labels if l == 0)
    n_spoof = sum(1 for l in train_labels if l == 1)
    weight_bonafide = (len(train_labels) / (2.0 * max(n_bonafide, 1)))
    weight_spoof = (len(train_labels) / (2.0 * max(n_spoof, 1)))
    class_weights = torch.tensor([weight_bonafide, weight_spoof], dtype=torch.float32, device=device)

    # Criterion, Model, Optimizer, Scaler per 05 §2.4
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    model = AcousticDeepfakeCNN(backbone=backbone, pretrained=True, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    best_eer = float("inf")
    patience_counter = 0

    # Resume from checkpoint if provided
    if resume_checkpoint and Path(resume_checkpoint).is_file():
        ckpt = torch.load(resume_checkpoint, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scaler_state_dict" in ckpt and device.type == "cuda":
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_eer = ckpt.get("val_eer", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best EER: {best_eer:.4f}")

    training_history: list[dict[str, Any]] = []

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_eer, val_thresh, val_acc = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_eer": round(val_eer, 4),
            "val_threshold": round(val_thresh, 4),
            "val_accuracy": round(val_acc, 4),
            "duration_s": round(elapsed, 1),
        }
        training_history.append(log_entry)
        print(f"Epoch {epoch:02d} [{elapsed:.1f}s] - Loss: {train_loss:.4f} - Val EER: {val_eer:.4f} (Acc: {val_acc:.4f})")

        # Save latest checkpoint for resumption
        latest_path = out_path / "checkpoint_latest.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if device.type == "cuda" else {},
                "val_eer": val_eer,
                "backbone": backbone,
            },
            latest_path,
        )

        # Early stopping & best model tracking
        if val_eer < best_eer:
            best_eer = val_eer
            patience_counter = 0
            best_path = out_path / "model_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_eer": val_eer,
                    "val_threshold": val_thresh,
                    "backbone": backbone,
                },
                best_path,
            )
            print(f"  ★ New best model saved with Val EER: {val_eer:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without EER improvement.")
                break

    # Save training history and config
    config_summary = {
        "backbone": backbone,
        "epochs_trained": len(training_history),
        "best_val_eer": best_eer,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "history": training_history,
    }
    with open(out_path / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(config_summary, f, indent=2)

    return config_summary
