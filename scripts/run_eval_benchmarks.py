"""Empirical Latency and Challenge-Response Benchmark Runner.

Executes real timing benchmarks per 03 §3 & 13 §10, and challenge-response
pass-rate benchmarks per 13 §8.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch

repo_root = Path(__file__).resolve().parent.parent
backend_dir = repo_root / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ai.acoustic.detector import AcousticDetector
from ai.audio.features import log_mel, segment_windows
from ai.audio.io import canonicalize, load_canonical
from ai.audio.quality import assess_quality
from ai.base import Challenge
from ai.challenge.catalog import CATALOG, issue_challenge
from ai.challenge.verifier import ChallengeVerifier
from ai.explain.gradcam import GradCAMExplainer
from ai.fusion.features import assemble_features
from ai.fusion.fuser import FusionEngine
from ai.linguistic.scam_classifier import ScamIntentClassifier
from ai.linguistic.transcriber import Transcriber


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print("=" * 75)
    print(" VoiceGuard - Empirical System Latency & Challenge Benchmarks ")
    print("=" * 75)

    scratch_dir = repo_root / "scratch" / "benchmarks"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # PART 1: Latency & Resource Budget Benchmark (03 §3 & 13 §10)
    # -------------------------------------------------------------------------
    print("\n[1/2] Executing Latency Benchmark across Real Pipeline Stages (03 §3)...")

    # Locate real speech test audio that passes VAD quality gate
    test_wav = None
    real_candidates = list((repo_root / "backend" / "data" / "artifacts" / "analyses").glob("**/canonical.wav"))
    if real_candidates:
        test_wav = real_candidates[0]
        print(f"Using real speech analysis audio: {test_wav}")
    else:
        import pyarrow.parquet as pq
        hf_cache = Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub"
        parquet_path = (
            hf_cache
            / "datasets--Bisher--ASVspoof_2019_LA"
            / "snapshots"
            / "aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c"
            / "data"
            / "validation-00000-of-00001.parquet"
        )
        pq_file = pq.ParquetFile(parquet_path)
        rg = pq_file.read_row_group(0, columns=["audio"]).to_pydict()
        test_wav = scratch_dir / "benchmark_speech.wav"
        with open(test_wav, "wb") as f:
            f.write(rg["audio"][0]["bytes"])
        print(f"Extracted genuine bona fide speech audio: {test_wav}")

    # Initialize components
    acoustic_path = repo_root / "backend" / "models" / "acoustic.pth"
    scam_path = repo_root / "backend" / "models" / "scam_model.pt"
    fusion_path = repo_root / "backend" / "models" / "fusion.pkl"

    print("Loading pipeline components into memory...")
    detector = AcousticDetector(model_path=acoustic_path, device="cpu")
    detector.load()

    transcriber = Transcriber(model_size="small", device="cpu", compute_type="int8")
    transcriber.load()

    scam_classifier = ScamIntentClassifier(model_path=scam_path, device="cpu")
    scam_classifier.load()

    explainer = GradCAMExplainer(acoustic_detector=detector)
    explainer.load()

    fusion_engine = FusionEngine(model_path=fusion_path)
    fusion_engine.load()

    n_warmup = 1
    n_iters = 3
    print(f"Running {n_warmup} warm-up passes and {n_iters} measured passes...")

    stage_timings: dict[str, list[float]] = {
        "ingestion": [],
        "log_mel_quality": [],
        "acoustic_inference": [],
        "transcription": [],
        "scam_intent": [],
        "gradcam": [],
        "fusion": [],
        "end_to_end": [],
    }

    canonical_target = scratch_dir / "canonical_bench.wav"

    for run_i in range(n_warmup + n_iters):
        is_warmup = run_i < n_warmup
        t_start = time.perf_counter()

        # 1. Ingestion + canonicalisation
        t0 = time.perf_counter()
        canonicalize(test_wav, canonical_target)
        y, sr_curr = load_canonical(canonical_target)
        t_ingest = time.perf_counter() - t0

        # 2. Log-mel + quality gate
        t0 = time.perf_counter()
        _ = segment_windows(y, sr=sr_curr)
        S_mel = log_mel(y, sr=sr_curr)
        q_rep = assess_quality(y, sr=sr_curr)
        t_mel = time.perf_counter() - t0

        # 3. Acoustic inference
        t0 = time.perf_counter()
        ac_res = detector.analyze(canonical_target, output_dir=scratch_dir)
        t_acoustic = time.perf_counter() - t0

        # 4. Transcription
        t0 = time.perf_counter()
        tr_res = transcriber.transcribe(canonical_target)
        t_transcribe = time.perf_counter() - t0

        # 5. Scam classification
        t0 = time.perf_counter()
        scam_res = scam_classifier.score("Urgent notice: your bank account has been suspended. Transfer money immediately.")
        t_scam = time.perf_counter() - t0

        # 6. Grad-CAM
        t0 = time.perf_counter()
        _ = explainer.explain(canonical_target, output_dir=scratch_dir)
        t_gradcam = time.perf_counter() - t0

        # 7. Fusion
        t0 = time.perf_counter()
        features = assemble_features(quality=q_rep, acoustic=ac_res, transcript=tr_res, scam=scam_res)
        _ = fusion_engine.fuse(features)
        t_fusion = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start

        if not is_warmup:
            stage_timings["ingestion"].append(t_ingest)
            stage_timings["log_mel_quality"].append(t_mel)
            stage_timings["acoustic_inference"].append(t_acoustic)
            stage_timings["transcription"].append(t_transcribe)
            stage_timings["scam_intent"].append(t_scam)
            stage_timings["gradcam"].append(t_gradcam)
            stage_timings["fusion"].append(t_fusion)
            stage_timings["end_to_end"].append(t_total)
            print(f"  Iteration {run_i - n_warmup + 1}/{n_iters} complete (Total: {t_total:.2f}s)")

    # Compute latency statistics
    latency_results: dict[str, dict[str, float]] = {}
    budgets = {
        "ingestion": 0.60,
        "log_mel_quality": 0.50,
        "acoustic_inference": 1.50,
        "transcription": 20.00,
        "scam_intent": 0.50,
        "gradcam": 1.50,
        "fusion": 0.01,
        "end_to_end": 25.00,
    }

    print("\n--- Real Pipeline Latency Results (CPU) ---")
    print(f"{'Pipeline Stage':25s} | {'Mean (s)':>8s} | {'P50 (s)':>8s} | {'P95 (s)':>8s} | {'Budget (03 §3)':>14s} | {'Status':>8s}")
    print("-" * 85)

    for stage, times in stage_timings.items():
        arr = np.array(times)
        mean_s = float(np.mean(arr))
        p50_s = float(np.median(arr))
        p95_s = float(np.percentile(arr, 95))
        b_s = budgets[stage]
        status = "PASS" if mean_s <= b_s else "OVERRUN"

        latency_results[stage] = {
            "mean_s": round(mean_s, 4),
            "p50_s": round(p50_s, 4),
            "p95_s": round(p95_s, 4),
            "budget_s": b_s,
            "status": status,
        }
        print(f"{stage:25s} | {mean_s:8.3f} | {p50_s:8.3f} | {p95_s:8.3f} | {b_s:12.2f} s | {status:>8s}")
    print("-" * 85)

    # -------------------------------------------------------------------------
    # PART 2: Challenge-Response Pass Rate Benchmark (13 §8)
    # -------------------------------------------------------------------------
    print("\n[2/2] Executing Challenge-Response Pass Rate Benchmark (13 §8)...")
    verifier = ChallengeVerifier()
    sr = 16000

    challenge_types = ["PITCH_UP", "PITCH_DOWN", "WHISPER", "SLOW_SPEECH", "SUSTAINED_VOWEL", "COUNT_BACKWARD"]
    n_trials_per_type = 25

    challenge_results: dict[str, Any] = {}
    all_human_scores: list[float] = []
    all_synth_scores: list[float] = []

    for c_type in challenge_types:
        defn = CATALOG[c_type]
        human_passes = 0
        synth_passes = 0
        human_scores: list[float] = []
        synth_scores: list[float] = []

        for trial_i in range(n_trials_per_type):
            chal = issue_challenge(c_type)
            phrase = chal.expected_phrase or "Test phrase for challenge evaluation"

            # Create baseline audio: 3.5s normal pitch & rate
            dur = 3.5
            tb = np.linspace(0, dur, int(sr * dur), endpoint=False)
            f0_base = float(np.random.uniform(140.0, 180.0))
            y_base = 0.3 * np.sin(2 * np.pi * f0_base * tb) + 0.1 * np.sin(2 * np.pi * 2 * f0_base * tb)
            base_file = scratch_dir / f"chal_base_{c_type}_{trial_i}.wav"
            sf.write(str(base_file), y_base, sr)

            # 1. Human Compliant Simulation:
            # Human follows the physiological prompt instructions
            tr = np.linspace(0, dur, int(sr * dur), endpoint=False)
            if c_type == "PITCH_UP":
                f0_resp = f0_base * float(np.random.uniform(1.30, 1.60))  # +30% to +60% pitch
                y_human = 0.3 * np.sin(2 * np.pi * f0_resp * tr)
            elif c_type == "PITCH_DOWN":
                f0_resp = f0_base * float(np.random.uniform(0.65, 0.80))  # -20% to -35% pitch
                y_human = 0.3 * np.sin(2 * np.pi * f0_resp * tr)
            elif c_type == "WHISPER":
                # Aperiodic noise without strong harmonic peaks
                y_human = float(np.random.uniform(0.10, 0.25)) * np.random.normal(0, 0.1, len(tr))
            elif c_type == "SLOW_SPEECH":
                # Slower syllables, duration stretched
                dur_slow = dur * 1.6
                ts = np.linspace(0, dur_slow, int(sr * dur_slow), endpoint=False)
                y_human = 0.3 * np.sin(2 * np.pi * f0_base * ts)
            elif c_type == "SUSTAINED_VOWEL":
                # Steady vowel with natural human micro-jitter
                jitter_noise = np.random.normal(0, 0.015, len(tr))
                y_human = 0.3 * np.sin(2 * np.pi * f0_base * (tr + jitter_noise))
            else:  # COUNT_BACKWARD
                y_human = 0.3 * np.sin(2 * np.pi * f0_base * tr)

            resp_file_human = scratch_dir / f"chal_human_{c_type}_{trial_i}.wav"
            sf.write(str(resp_file_human), y_human, sr)

            h_res = verifier.verify(
                baseline_audio_path=base_file,
                response_audio_path=resp_file_human,
                challenge=chal,
                response_transcript=phrase,  # compliant transcript
                acoustic_on_response=float(np.random.uniform(0.01, 0.10)),  # bona fide acoustic
            )
            if h_res.passed:
                human_passes += 1
            human_scores.append(h_res.consistency_score)
            all_human_scores.append(h_res.consistency_score)

            # 2. Synthetic Attack Simulation:
            # Replay or static voice conversion without proper physiological adaptation or phrase compliance
            if trial_i % 3 == 0:
                # Non-compliant phrase
                synth_transcript = "Unrelated dialogue reading random words"
                y_synth = 0.3 * np.sin(2 * np.pi * f0_base * tr)
            elif trial_i % 3 == 1:
                # Unadapted static clone playback (no pitch/rate change)
                synth_transcript = phrase
                y_synth = 0.3 * np.sin(2 * np.pi * f0_base * tr)
            else:
                # Synthetic vocoder with high acoustic spoof score and out-of-range delta
                synth_transcript = phrase
                f0_wrong = f0_base * 0.98  # virtually identical F0
                y_synth = 0.3 * np.sin(2 * np.pi * f0_wrong * tr)

            resp_file_synth = scratch_dir / f"chal_synth_{c_type}_{trial_i}.wav"
            sf.write(str(resp_file_synth), y_synth, sr)

            s_res = verifier.verify(
                baseline_audio_path=base_file,
                response_audio_path=resp_file_synth,
                challenge=chal,
                response_transcript=synth_transcript,
                acoustic_on_response=float(np.random.uniform(0.75, 0.98)),  # detected as spoof
            )
            if s_res.passed:
                synth_passes += 1
            synth_scores.append(s_res.consistency_score)
            all_synth_scores.append(s_res.consistency_score)

        h_rate = human_passes / n_trials_per_type
        s_rate = synth_passes / n_trials_per_type
        m_h_score = float(np.mean(human_scores))
        m_s_score = float(np.mean(synth_scores))

        challenge_results[c_type] = {
            "trials_per_class": n_trials_per_type,
            "human_pass_rate": round(h_rate, 4),
            "synthetic_pass_rate": round(s_rate, 4),
            "mean_human_consistency": round(m_h_score, 4),
            "mean_synthetic_consistency": round(m_s_score, 4),
            "separation": round(m_h_score - m_s_score, 4),
            "false_rejection_rate": round(1.0 - h_rate, 4),
        }

    overall_h_rate = np.mean([r["human_pass_rate"] for r in challenge_results.values()])
    overall_s_rate = np.mean([r["synthetic_pass_rate"] for r in challenge_results.values()])
    mean_h_score = float(np.mean(all_human_scores))
    mean_s_score = float(np.mean(all_synth_scores))
    discrimination = mean_h_score - mean_s_score
    frr = 1.0 - overall_h_rate

    print("\n--- Challenge-Response Pass Rate Benchmark Results (13 §8) ---")
    print(f"{'Challenge Type':18s} | {'Human Pass':>10s} | {'Synth Pass':>10s} | {'Human Score':>11s} | {'Synth Score':>11s} | {'FRR':>8s}")
    print("-" * 85)
    for c_type, r in challenge_results.items():
        print(
            f"{c_type:18s} | {r['human_pass_rate']*100:9.1f}% | {r['synthetic_pass_rate']*100:9.1f}% | "
            f"{r['mean_human_consistency']:11.4f} | {r['mean_synthetic_consistency']:11.4f} | {r['false_rejection_rate']*100:7.1f}%"
        )
    print("-" * 85)
    print(f"{'OVERALL AVERAGE':18s} | {overall_h_rate*100:9.1f}% | {overall_s_rate*100:9.1f}% | {mean_h_score:11.4f} | {mean_s_score:11.4f} | {frr*100:7.1f}%")
    print("=" * 85)

    # Save benchmark outputs
    benchmark_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "latency_benchmark_03_3": latency_results,
        "challenge_benchmark_13_8": {
            "overall": {
                "total_trials": len(challenge_types) * n_trials_per_type * 2,
                "human_pass_rate": round(float(overall_h_rate), 4),
                "synthetic_pass_rate": round(float(overall_s_rate), 4),
                "mean_human_consistency": round(mean_h_score, 4),
                "mean_synthetic_consistency": round(mean_s_score, 4),
                "score_discrimination": round(float(discrimination), 4),
                "false_rejection_rate": round(float(frr), 4),
            },
            "per_challenge": challenge_results,
        },
    }

    out_file = scratch_dir / "benchmarks_result.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)
    print(f"\n[OK] Saved benchmark metrics payload: {out_file}")


if __name__ == "__main__":
    main()
