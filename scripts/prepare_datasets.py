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
import re
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


def build_wavefake_protocol(audio_dir: Path | str, output_protocol: Path | str) -> Path:
    """Generate speaker/chapter-disjoint protocol for WaveFake corpus (16 §1.2).

    Assigns chapters/subsets disjointly:
      - Train split: LJ001 - LJ040 (~80%)
      - Dev split: LJ041 - LJ050 (~20%)
    """
    root = Path(audio_dir)
    out_proto = Path(output_protocol)
    out_proto.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    wav_files = list(root.rglob("*.wav")) + list(root.rglob("*.flac"))

    print(f"Discovered {len(wav_files)} audio files in WaveFake root: {root}")

    for w in wav_files:
        rel = w.relative_to(root)
        stem = w.stem
        # Detect speaker / chapter prefix e.g. LJ001, LJ025
        match = re.match(r"(LJ\d{3})", stem, re.IGNORECASE)
        speaker_id = match.group(1).upper() if match else f"SPK_{abs(hash(stem)) % 50:03d}"

        # Determine split based on chapter ID to guarantee strict disjointness
        chap_num = int(speaker_id[2:]) if speaker_id.startswith("LJ") and speaker_id[2:].isdigit() else int(speaker_id[-2:])
        split = "train" if chap_num <= 40 else "dev"

        # Determine label and attack
        parent_name = w.parent.name.lower()
        if any(b in parent_name for b in ("real", "bona_fide", "ljspeech-1.1", "clean")):
            label = 0
            attack_id = "-"
        else:
            label = 1
            # Infer vocoder type from folder name
            attack_id = parent_name.replace("ljspeech_", "").replace("_", "-")

        entries.append({
            "path": str(w.resolve()),
            "speaker_id": speaker_id,
            "split": split,
            "label": label,
            "attack_id": attack_id,
        })

    with open(out_proto, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "speaker_id", "split", "label", "attack_id"])
        writer.writeheader()
        writer.writerows(entries)

    print(f"✓ Generated WaveFake protocol at {out_proto} ({len(entries)} items)")
    return out_proto


def build_in_the_wild_protocol(audio_dir: Path | str, output_protocol: Path | str) -> Path:
    """Generate evaluation protocol for In-the-Wild corpus (16 §1.3).

    All utterances belong to out-of-domain evaluation (split='eval').
    """
    root = Path(audio_dir)
    out_proto = Path(output_protocol)
    out_proto.parent.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []

    # Check for meta.csv
    meta_csvs = list(root.rglob("*.csv"))
    meta_file = None
    for m in meta_csvs:
        if "meta" in m.name.lower() or "label" in m.name.lower():
            meta_file = m
            break

    if meta_file and meta_file.is_file():
        print(f"Parsing In-the-Wild metadata CSV: {meta_file}")
        with open(meta_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("file") or row.get("filename") or row.get("path")
                spk = row.get("speaker") or "itw_speaker"
                lbl_raw = str(row.get("label", "")).lower()
                is_spoof = 0 if "bona" in lbl_raw or lbl_raw in ("0", "real") else 1

                # Locate actual file
                cand = root / fname if fname else None
                if not cand or not cand.is_file():
                    found = list(root.rglob(f"{Path(fname).stem}.*")) if fname else []
                    cand = found[0] if found else None

                if cand and cand.is_file():
                    entries.append({
                        "path": str(cand.resolve()),
                        "speaker_id": spk,
                        "split": "eval",
                        "label": is_spoof,
                        "attack_id": "in_the_wild_spoof" if is_spoof else "-",
                    })
    else:
        # Infer from folder names: bona_fide/real vs spoof/fake
        wav_files = list(root.rglob("*.wav")) + list(root.rglob("*.mp3")) + list(root.rglob("*.flac"))
        print(f"Scanning directory hierarchy for {len(wav_files)} files...")
        for w in wav_files:
            parent_name = w.parent.name.lower()
            is_spoof = 0 if any(b in parent_name for b in ("real", "bona_fide", "genuine")) else 1
            spk = w.parent.parent.name if is_spoof else w.parent.name
            entries.append({
                "path": str(w.resolve()),
                "speaker_id": spk,
                "split": "eval",
                "label": is_spoof,
                "attack_id": "in_the_wild_spoof" if is_spoof else "-",
            })

    with open(out_proto, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "speaker_id", "split", "label", "attack_id"])
        writer.writeheader()
        writer.writerows(entries)

    print(f"✓ Generated In-the-Wild evaluation protocol at {out_proto} ({len(entries)} items)")
    return out_proto


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
                p_val = row.get("filename") or row.get("path") or row.get("rel_path")
                entries.append({
                    "rel_path": p_val,
                    "speaker_id": row.get("speaker_id", "unknown"),
                    "split": row.get("split", "train"),
                    "label": 1 if str(row.get("label", "")).lower() in ("spoof", "1") else 0,
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
        rel = item["rel_path"]
        # If absolute path already provided, use directly
        candidate_p = Path(rel)
        if candidate_p.is_file():
            src_file = candidate_p
        else:
            src_file = src_dir / rel
            if not src_file.is_file():
                stem = Path(rel).stem
                for ext in [".flac", ".wav", ".mp3", ".ogg"]:
                    cand = src_dir / f"{stem}{ext}"
                    if cand.is_file():
                        src_file = cand
                        break

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
        try:
            y, sr = sf.read(str(target_wav), dtype="float32")
            q_report = assess_quality(y, sr=sr)
            if not q_report.passed:
                dropped_quality_count += 1
                continue
        except Exception:
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


def prepare_huggingface_asvspoof(output_dir: Path | str, precompute_specs: bool = True) -> dict[str, Any]:
    """Load official Bisher/ASVspoof_2019_LA from Hugging Face and prepare canonical audio and specs."""
    from datasets import load_dataset
    out_dir = Path(output_dir)
    canonical_dir = out_dir / "canonical"
    specs_dir = out_dir / "specs"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    if precompute_specs:
        specs_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    print("Loading Bisher/ASVspoof_2019_LA train & validation splits from Hugging Face...")
    for hf_split, split_label in [("train", "train"), ("validation", "dev")]:
        ds = load_dataset("Bisher/ASVspoof_2019_LA", split=hf_split)
        print(f"Loaded {len(ds)} samples for partition: {split_label}")
        for i, item in enumerate(ds):
            fname = item.get("audio_file_name") or f"{split_label}_{i:06d}.flac"
            stem = Path(fname).stem
            wav_path = canonical_dir / f"{stem}.wav"

            y = np.array(item["audio"]["array"], dtype=np.float32)
            sr = item["audio"]["sampling_rate"]
            if not wav_path.is_file():
                sf.write(str(wav_path), y, sr)

            label = 0 if item["key"] == 0 or str(item["key"]).lower() == "bonafide" else 1
            entry: dict[str, Any] = {
                "path": str(wav_path.resolve()),
                "speaker_id": item["speaker_id"],
                "split": split_label,
                "label": label,
                "attack_id": item.get("system_id", "-"),
                "duration_s": round(len(y) / sr, 2),
            }
            if precompute_specs:
                spec_path = specs_dir / f"{stem}.npy"
                if not spec_path.is_file():
                    mel = log_mel(y, sr=sr)
                    norm_spec = normalize_spectrogram(mel)
                    np.save(spec_path, norm_spec)
                entry["npy_path"] = str(spec_path.resolve())

            manifest_entries.append(entry)
            if (i + 1) % 2000 == 0:
                print(f"  [{split_label}] Processed {i + 1}/{len(ds)} samples...")

    verify_speaker_disjointness(manifest_entries)
    print("✓ Speaker disjointness verified across splits.")

    manifest_json = out_dir / "manifest.json"
    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    manifest_csv = out_dir / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        if manifest_entries:
            writer = csv.DictWriter(f, fieldnames=list(manifest_entries[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_entries)

    return {"total_files_accepted": len(manifest_entries)}


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceGuard acoustic dataset preprocessor per 06 §2.6")
    parser.add_argument("--protocol", type=str, default=None, help="Path to protocol table or CSV")
    parser.add_argument("--audio-dir", type=str, default=None, help="Path to raw audio directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to write canonical audio and manifests")
    parser.add_argument("--dataset-type", type=str, default="protocol", choices=["protocol", "wavefake", "in_the_wild", "hf_asvspoof"], help="Dataset format type")
    parser.add_argument("--precompute-specs", action="store_true", default=True, help="Precompute spectrogram shards")
    args = parser.parse_args()

    if args.dataset_type == "hf_asvspoof":
        prepare_huggingface_asvspoof(args.output_dir, precompute_specs=args.precompute_specs)
        return

    if not args.audio_dir:
        raise ValueError("--audio-dir is required when not using hf_asvspoof")

    protocol_file = args.protocol
    if args.dataset_type == "wavefake":
        proto_path = Path(args.output_dir) / "wavefake_protocol.csv"
        protocol_file = str(build_wavefake_protocol(args.audio_dir, proto_path))
    elif args.dataset_type == "in_the_wild":
        proto_path = Path(args.output_dir) / "in_the_wild_protocol.csv"
        protocol_file = str(build_in_the_wild_protocol(args.audio_dir, proto_path))

    if not protocol_file:
        raise ValueError("Must provide either --protocol or --dataset-type with 'wavefake' or 'in_the_wild'")

    process_dataset(
        protocol_path=protocol_file,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        precompute_specs=args.precompute_specs,
    )


if __name__ == "__main__":
    main()
