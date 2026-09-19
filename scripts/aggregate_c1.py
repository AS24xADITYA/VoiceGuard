"""Aggregator for ASVspoof 2019 LA Official Evaluation Partition Results."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

repo_root = Path(__file__).resolve().parent.parent
backend_dir = repo_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ai.evaluation.metrics import compute_all_metrics, compute_eer


def main() -> None:
    parts_dir = repo_root / "scratch"
    part_files = sorted(list(parts_dir.glob("c1_part_*.jsonl")))

    if not part_files:
        print("ERROR: No c1_part_*.jsonl files found in scratch/")
        sys.exit(1)

    all_records = []
    seen_fnames = set()

    for p in part_files:
        print(f"Reading {p.name}...")
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec["fname"] not in seen_fnames:
                        seen_fnames.add(rec["fname"])
                        all_records.append(rec)

    total = len(all_records)
    print(f"\nTotal unique evaluated trials: {total} (Expected: 71237)")
    if total != 71237:
        print(f"WARNING: Total count {total} does not match official 71,237. Evaluation may still be running.")

    y_true = np.array([r["label"] for r in all_records], dtype=int)
    y_scores = np.array([r["score"] for r in all_records], dtype=float)

    c1_metrics = compute_all_metrics(y_true, y_scores)

    print("\n" + "=" * 80)
    print("CORRECTED CONDITION C1 IN-DOMAIN (OFFICIAL EVALUATION PARTITION)")
    print("Corpus: ASVspoof 2019 LA Official Evaluation Split (Unseen Attacks A07–A19)")
    print("=" * 80)
    print(f"Total Trials     : {c1_metrics.total_samples}")
    print(f"Bona Fide        : {c1_metrics.n_bonafide}")
    print(f"Spoof            : {c1_metrics.n_spoof}")
    print(f"EER              : {c1_metrics.eer * 100:.2f}%")
    print(f"EER Threshold    : {c1_metrics.eer_threshold:.4f}")
    print(f"min t-DCF        : {c1_metrics.min_tdcf:.4f}")
    print(f"AUC-ROC          : {c1_metrics.auc_roc:.4f}")
    print(f"F1 Score         : {c1_metrics.f1:.4f}")
    print(f"Accuracy         : {c1_metrics.accuracy * 100:.2f}%")
    print(f"Precision        : {c1_metrics.precision:.4f}")
    print(f"Recall           : {c1_metrics.recall:.4f}")
    print(f"ECE              : {c1_metrics.ece:.4f}")
    print(f"Confusion Matrix : {c1_metrics.confusion_matrix}  [[TN, FP], [FN, TP]]")
    print("=" * 80)

    # Per-attack breakdown
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

    bonafide_scores = [r["score"] for r in all_records if r["label"] == 0]
    scores_by_attack: dict[str, list[float]] = {}
    for r in all_records:
        if r["label"] == 1:
            scores_by_attack.setdefault(r["attack_id"], []).append(r["score"])

    c1_attacks: dict[str, Any] = {}
    print("\n" + "=" * 80)
    print("C1 PER-ATTACK BREAKDOWN (UNSEEN ATTACKS A07–A19)")
    print("=" * 80)
    print(f"| Attack | Description                              | Samples | Mean Score | Attack EER |")
    print(f"|--------|------------------------------------------|---------|------------|------------|")

    for atk in sorted(scores_by_attack.keys()):
        atk_scores = scores_by_attack[atk]
        comb_scores = np.array(bonafide_scores + atk_scores)
        comb_labels = np.array([0] * len(bonafide_scores) + [1] * len(atk_scores))
        try:
            atk_eer, _ = compute_eer(comb_labels, comb_scores)
        except Exception:
            atk_eer = float("nan")

        desc = attack_descs.get(atk, "Unseen Attack")
        c1_attacks[atk] = {
            "samples": len(atk_scores),
            "description": desc,
            "mean_score": round(float(np.mean(atk_scores)), 4),
            "eer_or_miss_rate": round(float(atk_eer), 4),
        }
        print(f"| {atk:6} | {desc:40} | {len(atk_scores):7} | {np.mean(atk_scores):10.4f} | {atk_eer * 100:9.2f}% |")

    print("=" * 80)

    # Generalization gap update
    metrics_path = repo_root / "backend" / "models" / "metrics.json"
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    c2_eer = metrics_data["conditions"]["c2_out_of_domain"]["eer"]
    c2_auc = metrics_data["conditions"]["c2_out_of_domain"]["auc_roc"]

    c1_to_c2_eer_gap = round(c2_eer - c1_metrics.eer, 4)
    c1_to_c2_auc_gap = round(c1_metrics.auc_roc - c2_auc, 4)

    print(f"\n========================================================================================")
    print(f"CORRECTED C1 -> C2 GENERALIZATION GAP (Per 06 §1 & 13 §6.3):")
    print(f"  C1 In-Domain EER (Official Eval, A07–A19): {c1_metrics.eer * 100:.2f}%")
    print(f"  C2 Out-of-Domain EER (In-the-Wild):        {c2_eer * 100:.2f}%")
    print(f"  Corrected EER Gap:                         +{c1_to_c2_eer_gap * 100:.2f}%")
    print(f"  C1 In-Domain AUC:                          {c1_metrics.auc_roc:.4f}")
    print(f"  C2 Out-of-Domain AUC:                      {c2_auc:.4f}")
    print(f"  Corrected AUC Gap:                         -{c1_to_c2_auc_gap:.4f}")
    print(f"========================================================================================\n")

    # Update metrics_data
    metrics_data["model"]["operating_threshold"] = round(float(c1_metrics.eer_threshold), 4)
    metrics_data["conditions"]["c1_in_domain"] = c1_metrics.to_dict()
    metrics_data["generalization_gap"]["c1_to_c2_eer_gap"] = c1_to_c2_eer_gap
    metrics_data["generalization_gap"]["c1_to_c2_auc_gap"] = c1_to_c2_auc_gap
    metrics_data["c1_per_attack_metrics"] = c1_attacks

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[SUCCESS] Updated {metrics_path}")

    app_metrics = repo_root / "backend" / "app" / "metrics.json"
    if app_metrics.is_file():
        with open(app_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"[SUCCESS] Updated {app_metrics}")


if __name__ == "__main__":
    main()
