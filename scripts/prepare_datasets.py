"""Dataset preparation and verification pipeline.

Per 06 §2.6:
  1. Resolves dataset protocols into (path, label, speaker_id, attack_id)
  2. Canonicalises audio to 16 kHz mono 16-bit PCM WAV
  3. Strict speaker-disjointness assertion across train/val/test splits (aborts on leakage)
  4. Computes per-file quality checks (drops below floor)
  5. Precomputes log-mel spectrogram shards (.npy)
  6. Emits manifests (manifest.csv, manifest.json) and class/split summaries
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# Add backend directory to sys.path to access ai modules
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ai.audio.features import log_mel, normalize_spectrogram
from ai.audio.io import canonicalize, validate_magic_bytes
from ai.audio.quality import assess_quality


def verify_speaker_disjointness(manifest_entries: list[dict[str, Any]]) -> None:
    """Assert that no speaker appears in more than one partition.

    Per 06 §1 & §2.6: Speaker leakage between train, validation, and evaluation
    invalidates anti-spoofing results. This assertion halts preprocessing if violated.
    """
    speakers_by_split: dict[str, set[str]] = {}

    for entry in manifest_entries:
        split = entry["split"]
        speaker = entry["speaker_id"]
        if split not in speakers_by_split:
            speakers_by_split[split] = set()
        speakers_by_split[split].add(speaker)

    splits = list(speakers_by_split.keys())
    violations = []

    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = speakers_by_split[s1].intersection(speakers_by_split[s2])
            if overlap:
                violations.append((s1, s2, overlap))

    if violations:
        error_msg = ["FATAL: Speaker disjointness violation detected!"]
        for s1, s2, overlap in violations:
            error_msg.append(f"  Overlap between '{s1}' and '{s2}': {len(overlap)} speakers: {list(overlap)[:5]}...")
        error_msg.append("Aborting dataset preparation to prevent test set contamination.")
        raise ValueError("\n".join(error_msg))


def process_dataset(
    protocol_path: Path | str,
    audio_dir: Path | str,
    output_dir: Path | str,
    precompute_specs: bool = True,
) -> dict[str, Any]:
    """Process an acoustic anti-spoofing dataset protocol and canonicalize files.

    Expected protocol format: CSV or whitespace-delimited table:
      [speaker_id] [audio_filename] [system_id/attack_id] [-] [key/label: bonafide | spoof]
    Or generic CSV with columns: (path, speaker_id, split, label, attack_id).
    """
    proto = Path(protocol_path)
    src_dir = Path(audio_dir)
    out_dir = Path(output_dir)
    canonical_dir = out_dir / "canonical"
    specs_dir = out_dir / "specs"

    canonical_dir.mkdir(parents=True, exist_ok=True)
    if precompute_specs:
        specs_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []

    with open(proto, "r", encoding="utf-8") as f:
        # Detect delimiter: comma or whitespace
        sample_line = f.readline()
        f.seek(0)
        delimiter = "," if "," in sample_line else None

        if delimiter:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append({
                    "rel_path": row.get("filename") or row.get("path"),
                    "speaker_id": row.get("speaker_id", "unknown"),
                    "split": row.get("split", "train"),
                    "label": 1 if row.get("label", "").lower() in ("spoof", "1") else 0,
                    "attack_id": row.get("attack_id", "-"),
                })
        else:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id, filename, _, _, key = parts[:5]
                    entries.append({
                        "rel_path": filename if filename.endswith(".wav") else f"{filename}.wav",
                        "speaker_id": speaker_id,
                        "split": "train" if "train" in str(proto).lower() else "dev" if "dev" in str(proto).lower() else "eval",
                        "label": 0 if key.lower() == "bonafide" else 1,
                        "attack_id": parts[3] if len(parts) > 3 else "-",
                    })

    print(f"Loaded {len(entries)} protocol records. Verifying speaker disjointness...")
    verify_speaker_disjointness(entries)
    print("✓ Speaker disjointness verified across splits.")

    processed_manifest: list[dict[str, Any]] = []
    dropped_quality_count = 0

    for i, item in enumerate(entries):
        src_file = src_dir / item["rel_path"]
        if not src_file.is_file():
            continue

        target_wav = canonical_dir / f"{src_file.stem}.wav"

        # Canonicalize if not already present
        if not target_wav.is_file():
            try:
                meta = canonicalize(src_file, target_wav)
            except Exception as e:
                print(f"Warning: Transcoding failed for {src_file}: {e}")
                continue

        # Quality check
        y, sr = sf.read(str(target_wav), dtype="float32")
        q_report = assess_quality(y, sr=sr)
        if not q_report.passed:
            dropped_quality_count += 1
            continue

        entry: dict[str, Any] = {
            "path": str(target_wav.resolve()),
            "speaker_id": item["speaker_id"],
            "split": item["split"],
            "label": item["label"],
            "attack_id": item["attack_id"],
            "duration_s": round(len(y) / sr, 2),
            "speech_ratio": round(q_report.speech_ratio, 3),
            "snr_db": round(q_report.snr_estimate_db, 1),
        }

        # Precompute normalized log-mel spectrogram tensor if requested
        if precompute_specs:
            spec_file = specs_dir / f"{src_file.stem}.npy"
            if not spec_file.is_file():
                mel = log_mel(y, sr=sr)
                norm_spec = normalize_spectrogram(mel)
                np.save(spec_file, norm_spec)
            entry["npy_path"] = str(spec_file.resolve())

        processed_manifest.append(entry)

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(entries)} audio files...")

    # Write manifests
    manifest_csv = out_dir / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        if processed_manifest:
            writer = csv.DictWriter(f, fieldnames=list(processed_manifest[0].keys()))
            writer.writeheader()
            writer.writerows(processed_manifest)

    manifest_json = out_dir / "manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(processed_manifest, f, indent=2)

    # Class and split distribution summary
    split_counts = Counter(m["split"] for m in processed_manifest)
    label_counts = Counter(m["label"] for m in processed_manifest)
    attack_counts = Counter(m["attack_id"] for m in processed_manifest)

    summary = {
        "total_files_accepted": len(processed_manifest),
        "total_dropped_quality": dropped_quality_count,
        "splits": dict(split_counts),
        "classes": {"bonafide": label_counts[0], "spoof": label_counts[1]},
        "attacks": dict(attack_counts),
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nProcessing complete! Accepted: {len(processed_manifest)}, Dropped (quality): {dropped_quality_count}")
    print(f"Summary: {json.dumps(summary, indent=2)}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceGuard acoustic dataset preprocessor per 06 §2.6")
    parser.add_argument("--protocol", type=str, required=True, help="Path to protocol table or CSV")
    parser.add_argument("--audio-dir", type=str, required=True, help="Path to raw audio directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to write canonical audio and manifests")
    parser.add_argument("--precompute-specs", action="store_true", default=True, help="Precompute spectrogram shards")
    args = parser.parse_args()

    process_dataset(
        protocol_path=args.protocol,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        precompute_specs=args.precompute_specs,
    )


if __name__ == "__main__":
    main()
