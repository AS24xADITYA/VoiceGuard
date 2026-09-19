"""Official ASVspoof 2019 LA Evaluation Partition Evaluator (Condition C1 per 06 §2.1 & 13 §6.1).

Evaluates existing acoustic checkpoint (backend/models/acoustic.pth) strictly on the
official ASVspoof 2019 LA evaluation partition (71,237 trials, 7,355 bona fide, 63,882 spoof
across unseen synthesis/VC attacks A07–A19).
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

# Ensure backend modules are resolvable
repo_root = Path(__file__).resolve().parent.parent
backend_dir = repo_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ai.acoustic.model import AcousticDeepfakeCNN
from ai.audio.features import log_mel, normalize_spectrogram, spectrogram_to_tensor
from ai.evaluation.metrics import compute_all_metrics, compute_eer


def load_official_eval_protocol(protocol_path: Path) -> dict[str, dict[str, Any]]:
    """Parse official ASVspoof2019.LA.cm.eval.trl.txt into a lookup map.

    Format: <speaker_id> <audio_file_name> <system_id> <attack_id> <key>
    """
    protocol: dict[str, dict[str, Any]] = {}
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                spk, fname, _, atk, key = parts[:5]
                protocol[fname] = {
                    "speaker_id": spk,
                    "attack_id": atk,
                    "label": 0 if key == "bonafide" else 1,
                    "key": key,
                }
    return protocol


def run_c1_evaluation(
    parquet_path: Path,
    protocol_path: Path,
    checkpoint_path: Path,
    metrics_json_path: Path,
    batch_size: int = 64,
    device_str: str | None = None,
) -> dict[str, Any]:
    print(f"=== ASVspoof 2019 LA Official Evaluation Partition Evaluator ===")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Parquet    : {parquet_path}")
    print(f"Protocol   : {protocol_path}")

    # 1. Load protocol
    protocol = load_official_eval_protocol(protocol_path)
    print(f"✓ Loaded {len(protocol)} protocol trials (A07–A19 & bona fide).")

    # 2. Setup device & model
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Inference device: {device}")

    model = AcousticDeepfakeCNN(backbone="efficientnet_b0", pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 3. Stream / read parquet rows and run batch inference
    pq_file = pq.ParquetFile(parquet_path)
    total_rows = pq_file.metadata.num_rows
    print(f"Total rows in Parquet: {total_rows}")

    all_scores: list[float] = []
    all_labels: list[int] = []
    scores_by_attack: dict[str, list[float]] = {}
    labels_by_attack: dict[str, list[int]] = {}

    batch_tensors: list[torch.Tensor] = []
    batch_labels: list[int] = []
    batch_attacks: list[str] = []

    processed = 0
    t_start = time.time()

    def flush_batch() -> None:
        nonlocal batch_tensors, batch_labels, batch_attacks
        if not batch_tensors:
            return
        x_batch = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(x_batch)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        all_scores.extend(probs.tolist())
        all_labels.extend(batch_labels)
        for p, l, a in zip(probs, batch_labels, batch_attacks):
            scores_by_attack.setdefault(a, []).append(float(p))
            labels_by_attack.setdefault(a, []).append(l)

        batch_tensors = []
        batch_labels = []
        batch_attacks = []

    # Iterate over row groups / batches in pyarrow
    for batch in pq_file.iter_batches(batch_size=1024, columns=["audio", "audio_file_name"]):
        pydict = batch.to_pydict()
        audios = pydict["audio"]
        fnames = pydict["audio_file_name"]

        for audio_struct, fname in zip(audios, fnames):
            stem = Path(fname).stem
            meta = protocol.get(stem) or protocol.get(fname)
            if not meta:
                # Fallback if stem not directly found
                continue

            # Decode audio waveform
            raw_bytes = audio_struct["bytes"]
            y, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")

            # Pad or truncate to 4.0s @ 16kHz
            target_samples = int(4.0 * sr)
            if len(y) < target_samples:
                y = np.pad(y, (0, target_samples - len(y)), mode="reflect")
            elif len(y) > target_samples:
                y = y[:target_samples]

            mel = log_mel(y, sr=sr)
            norm_spec = normalize_spectrogram(mel)
            tensor = torch.from_numpy(spectrogram_to_tensor(norm_spec)).unsqueeze(0)

            batch_tensors.append(tensor)
            batch_labels.append(meta["label"])
            batch_attacks.append(meta["attack_id"])
            processed += 1

            if len(batch_tensors) >= batch_size:
                flush_batch()

            if processed % 5000 == 0:
                elapsed = time.time() - t_start
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_rows - processed) / rate if rate > 0 else 0
                print(f"  Processed {processed}/{total_rows} ({rate:.1f} samples/s, ETA: {eta:.0f}s)...")

    flush_batch()
    total_time = time.time() - t_start
    print(f"✓ Inference complete: {len(all_scores)} samples evaluated in {total_time:.1f}s.")

    # 4. Compute comprehensive C1 metrics
    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)

    c1_metrics = compute_all_metrics(y_true, y_scores)
    print(f"\n--- Corrected C1 In-Domain Evaluation Results ---")
    print(f"Total Samples    : {c1_metrics.total_samples} ({c1_metrics.n_bonafide} bona fide, {c1_metrics.n_spoof} spoof)")
    print(f"EER              : {c1_metrics.eer * 100:.2f}% (Threshold: {c1_metrics.eer_threshold:.4f})")
    print(f"min t-DCF        : {c1_metrics.min_tdcf:.4f}")
    print(f"AUC-ROC          : {c1_metrics.auc_roc:.4f}")
    print(f"F1 Score         : {c1_metrics.f1:.4f}")
    print(f"Accuracy         : {c1_metrics.accuracy * 100:.2f}%")
    print(f"ECE              : {c1_metrics.ece:.4f}")
    print(f"Confusion Matrix : {c1_metrics.confusion_matrix}")

    # 5. Compute per-attack breakdown (A07–A19)
    bonafide_scores = [s for s, l in zip(all_scores, all_labels) if l == 0]
    c1_attacks: dict[str, Any] = {}
    print("\n--- Per-Attack Breakdown (Unseen Synthesis/VC A07–A19) ---")
    print(f"| Attack | Type Description                         | Samples | Mean Score | Attack EER |")
    print(f"|--------|------------------------------------------|---------|------------|------------|")

    attack_descs = {
        "A07": "Vocoder (Neural / WaveNet)",
        "A08": "Neural vocoder (WaveRNN)",
        "A09": "Vocoder (World)",
        "A10": "Neural vocoder (MelGAN)",
        "A11": "Griffin-Lim vocoder",
        "A12": "Neural vocoder (WaveGlow)",
        "A13": "VC (CycleGAN)",
        "A14": "VC (StarGAN)",
        "A15": "VC (Spectral mapping)",
        "A16": "WaveNet TTS + Griffin-Lim",
        "A17": "Neural TTS (Tacotron 2 + WaveGlow)",
        "A18": "Neural TTS (Transformer + WaveNet)",
        "A19": "Neural TTS (FastSpeech + MelGAN)",
    }

    for atk_id in sorted([k for k in scores_by_attack.keys() if k != "-"]):
        atk_scores = scores_by_attack[atk_id]
        combined_scores = np.array(bonafide_scores + atk_scores)
        combined_labels = np.array([0] * len(bonafide_scores) + [1] * len(atk_scores))
        try:
            atk_eer, _ = compute_eer(combined_labels, combined_scores)
        except Exception:
            atk_eer = float("nan")

        desc = attack_descs.get(atk_id, "Unknown attack")
        c1_attacks[atk_id] = {
            "samples": len(atk_scores),
            "description": desc,
            "mean_score": round(float(np.mean(atk_scores)), 4),
            "eer_or_miss_rate": round(float(atk_eer), 4),
        }
        print(f"| {atk_id:6} | {desc:40} | {len(atk_scores):7} | {np.mean(atk_scores):10.4f} | {atk_eer * 100:9.2f}% |")

    # 6. Load existing metrics.json to update C1 and recompute generalization gap
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    # Condition C2 (In-the-Wild)
    c2_data = metrics_data["conditions"]["c2_out_of_domain"]
    c2_eer = c2_data["eer"]
    c2_auc = c2_data["auc_roc"]

    c1_to_c2_eer_gap = round(c2_eer - c1_metrics.eer, 4)
    c1_to_c2_auc_gap = round(c1_metrics.auc_roc - c2_auc, 4)

    print(f"\n========================================================================================")
    print(f"Corrected C1 -> C2 Generalization Gap (Per 06 §1 & 13 §6.3):")
    print(f"  In-Domain C1 EER (ASVspoof 2019 LA Eval, A07–A19): {c1_metrics.eer * 100:.2f}%")
    print(f"  Out-of-Domain C2 EER (In-the-Wild):               {c2_eer * 100:.2f}%")
    print(f"  Corrected EER Gap:                                +{c1_to_c2_eer_gap * 100:.2f}%")
    print(f"  In-Domain C1 AUC:                                 {c1_metrics.auc_roc:.4f}")
    print(f"  Out-of-Domain C2 AUC:                             {c2_auc:.4f}")
    print(f"  Corrected AUC Gap:                                -{c1_to_c2_auc_gap:.4f}")
    print(f"========================================================================================")

    # Update payload
    metrics_data["model"]["operating_threshold"] = round(float(c1_metrics.eer_threshold), 4)
    metrics_data["conditions"]["c1_in_domain"] = c1_metrics.to_dict()
    metrics_data["generalization_gap"]["c1_to_c2_eer_gap"] = c1_to_c2_eer_gap
    metrics_data["generalization_gap"]["c1_to_c2_auc_gap"] = c1_to_c2_auc_gap
    metrics_data["c1_per_attack_metrics"] = c1_attacks

    # Write to backend/models/metrics.json
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✓ Updated metrics written to: {metrics_json_path}")

    # Also update app metrics if present
    app_metrics_path = repo_root / "backend" / "app" / "metrics.json"
    if app_metrics_path.is_file():
        with open(app_metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"✓ Updated app metrics written to: {app_metrics_path}")

    return metrics_data


if __name__ == "__main__":
    cache_hub = Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub"
    parquet_matches = list(cache_hub.glob("datasets--Bisher--ASVspoof_2019_LA/**/test-00000-of-00001.parquet"))
    if not parquet_matches:
        # Check blobs
        blob_dir = cache_hub / "datasets--Bisher--ASVspoof_2019_LA" / "blobs"
        parquet_matches = [p for p in blob_dir.glob("*") if not p.name.endswith(".incomplete") and p.stat().st_size > 4 * 1024**3]

    if not parquet_matches:
        print("ERROR: Official eval parquet file not yet completely downloaded.")
        sys.exit(1)

    parquet_file = parquet_matches[0]
    protocol_file = list(cache_hub.glob("datasets--Nemez1z--asvspoof-2019-la/**/ASVspoof2019.LA.cm.eval.trl.txt"))[0]
    ckpt_file = repo_root / "backend" / "models" / "acoustic.pth"
    metrics_file = repo_root / "backend" / "models" / "metrics.json"

    run_c1_evaluation(
        parquet_path=parquet_file,
        protocol_path=protocol_file,
        checkpoint_path=ckpt_file,
        metrics_json_path=metrics_file,
    )
