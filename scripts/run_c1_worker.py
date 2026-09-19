"""Parallel Worker for ASVspoof 2019 LA Official Evaluation Partition."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

repo_root = Path(__file__).resolve().parent.parent
backend_dir = repo_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ai.acoustic.model import AcousticDeepfakeCNN
from ai.audio.features import normalize_spectrogram, spectrogram_to_tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--start-rg", type=int, required=True)
    parser.add_argument("--end-rg", type=int, required=True)
    parser.add_argument("--parquet-path", type=str, required=True)
    parser.add_argument("--protocol-path", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    # 1. Load protocol map
    protocol: dict[str, dict] = {}
    with open(args.protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                spk, fname, _, atk, key = parts[:5]
                protocol[fname] = {
                    "spk": spk,
                    "atk": atk,
                    "lbl": 0 if key == "bonafide" else 1,
                    "key": key,
                }

    # 2. Setup model
    model = AcousticDeepfakeCNN(backbone="efficientnet_b0", pretrained=False)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # 3. Precompute mel filterbank and STFT window
    mel_basis = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=20, fmax=8000)
    fft_window = librosa.filters.get_window("hann", 400)

    # 4. Open Parquet and iterate through assigned row groups
    pq_file = pq.ParquetFile(args.parquet_path)
    out_file = Path(args.output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Check already processed items if resuming
    processed_files = set()
    if out_file.is_file():
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    processed_files.add(d["fname"])

    print(f"[Worker {args.worker_id}] Starting row groups {args.start_rg} to {args.end_rg} (already processed: {len(processed_files)})...")
    t_start = time.time()
    total_processed = len(processed_files)

    with open(out_file, "a", encoding="utf-8") as out_fp:
        for rg_idx in range(args.start_rg, args.end_rg):
            rg = pq_file.read_row_group(rg_idx, columns=["audio", "audio_file_name"]).to_pydict()
            audios = rg["audio"]
            fnames = rg["audio_file_name"]

            batch_tensors = []
            batch_meta = []

            for audio_struct, fname in zip(audios, fnames):
                stem = Path(fname).stem
                if stem in processed_files or fname in processed_files:
                    continue

                meta = protocol.get(stem) or protocol.get(fname)
                if not meta:
                    continue

                raw_bytes = audio_struct["bytes"]
                y, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")

                target_samples = int(4.0 * sr)
                if len(y) < target_samples:
                    y = np.pad(y, (0, target_samples - len(y)), mode="reflect")
                else:
                    y = y[:target_samples]

                D = librosa.stft(y, n_fft=1024, hop_length=160, win_length=400, window=fft_window)
                S = np.dot(mel_basis, np.abs(D) ** 2)
                S_db = librosa.power_to_db(S, ref=np.max, top_db=80.0)
                norm_spec = normalize_spectrogram(S_db)
                t = torch.from_numpy(spectrogram_to_tensor(norm_spec)).unsqueeze(0)

                batch_tensors.append(t)
                batch_meta.append((stem, meta["spk"], meta["atk"], meta["lbl"]))

                if len(batch_tensors) >= args.batch_size:
                    x_batch = torch.cat(batch_tensors, dim=0)
                    with torch.no_grad():
                        logits = model(x_batch)
                        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()

                    for (st, spk, atk, lbl), p in zip(batch_meta, probs):
                        record = {
                            "fname": st,
                            "speaker_id": spk,
                            "attack_id": atk,
                            "label": int(lbl),
                            "score": float(p),
                        }
                        out_fp.write(json.dumps(record) + "\n")
                        processed_files.add(st)
                        total_processed += 1

                    out_fp.flush()
                    batch_tensors = []
                    batch_meta = []

            # Flush any remaining items in row group
            if batch_tensors:
                x_batch = torch.cat(batch_tensors, dim=0)
                with torch.no_grad():
                    logits = model(x_batch)
                    probs = torch.softmax(logits, dim=-1)[:, 1].numpy()

                for (st, spk, atk, lbl), p in zip(batch_meta, probs):
                    record = {
                        "fname": st,
                        "speaker_id": spk,
                        "attack_id": atk,
                        "label": int(lbl),
                        "score": float(p),
                    }
                    out_fp.write(json.dumps(record) + "\n")
                    processed_files.add(st)
                    total_processed += 1

                out_fp.flush()

            if (rg_idx - args.start_rg + 1) % 25 == 0 or rg_idx == args.end_rg - 1:
                elapsed = time.time() - t_start
                rate = (total_processed - len(processed_files) + total_processed) / elapsed if elapsed > 0 else 0
                print(f"[Worker {args.worker_id}] RG {rg_idx+1}/{args.end_rg} | Processed: {total_processed} items ({elapsed:.1f}s)")

    print(f"[Worker {args.worker_id}] Finished! Total records in {out_file.name}: {total_processed}")


if __name__ == "__main__":
    main()
