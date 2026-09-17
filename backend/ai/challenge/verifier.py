"""Challenge-response verification engine.

Per 05 §4.3-§4.5:
  - Comparative feature extraction between baseline and challenge response:
      * f0_median, f0_std (via librosa.yin)
      * hnr (harmonic-to-noise ratio)
      * spectral_flatness, spectral_centroid
      * rms_energy, zcr
      * speaking_rate, jitter, shimmer
  - Graded falloff scoring against expected relative delta ranges
  - Anti-replay & compliance checking: phrase similarity >= 0.70
  - Non-compliant responses produce consistency_score = 0.0 with zero pass contribution

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Final

import librosa
import numpy as np
import structlog

from ai.audio.io import load_canonical
from ai.base import Challenge, ChallengeResult, Component
from ai.challenge.catalog import CATALOG, issue_challenge

log = structlog.get_logger()


def harmonic_to_noise_ratio(y: np.ndarray, sr: int = 16000) -> float:
    """Compute harmonic-to-noise ratio via harmonic-percussive separation."""
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_power = float(np.mean(y_harm ** 2))
        perc_power = float(np.mean(y_perc ** 2)) + 1e-8
        return float(harm_power / perc_power)
    except Exception:
        return 1.0


def extract_f0_features(y: np.ndarray, sr: int = 16000) -> tuple[float, float, float, float]:
    """Extract median F0, F0 standard deviation, jitter, and shimmer."""
    try:
        f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr)
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 5:
            f0_med = float(np.nanmedian(valid_f0))
            f0_std = float(np.nanstd(valid_f0))
            # Local jitter: relative cycle-to-cycle F0 difference
            diffs = np.abs(np.diff(valid_f0))
            jitter = float(np.mean(diffs) / (f0_med + 1e-8))
            # Local shimmer: relative cycle-to-cycle peak envelope difference
            peaks = np.abs(y[:: max(1, int(sr / f0_med))])
            shimmer = float(np.mean(np.abs(np.diff(peaks))) / (np.mean(peaks) + 1e-8)) if len(peaks) > 5 else 0.02
            return f0_med, f0_std, jitter, shimmer
    except Exception:
        pass
    return 150.0, 20.0, 0.015, 0.03


def estimate_speaking_rate(y: np.ndarray, sr: int = 16000) -> float:
    """Estimate speaking rate (syllables/sec) using envelope peak counting."""
    try:
        envelope = np.abs(y)
        hop = int(sr * 0.02)  # 20ms smoothing
        smoothed = np.convolve(envelope, np.ones(hop) / hop, mode="same")
        # Peak detection above median
        thresh = np.median(smoothed) * 1.5
        peaks = (smoothed[1:-1] > smoothed[:-2]) & (smoothed[1:-1] > smoothed[2:]) & (smoothed[1:-1] > thresh)
        n_peaks = np.sum(peaks)
        duration = len(y) / sr
        return float(n_peaks / max(duration, 0.1))
    except Exception:
        return 3.0


def extract_acoustic_features(y: np.ndarray, sr: int = 16000) -> dict[str, float]:
    """Compute dictionary of comparative acoustic features per 05 §4.3."""
    f0_med, f0_std, jitter, shimmer = extract_f0_features(y, sr)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    hnr = harmonic_to_noise_ratio(y, sr)
    sp_rate = estimate_speaking_rate(y, sr)

    return {
        "f0_median": f0_med,
        "f0_std": f0_std,
        "hnr": hnr,
        "spectral_flatness": flatness,
        "spectral_centroid": centroid,
        "rms_energy": rms,
        "speaking_rate": sp_rate,
        "jitter": jitter,
        "shimmer": shimmer,
    }


def relative_change(v_base: float, v_resp: float) -> float:
    """Compute relative delta (response - baseline) / baseline."""
    denom = abs(v_base) if abs(v_base) > 1e-6 else 1.0
    return float((v_resp - v_base) / denom)


def graded_falloff(val: float, lo: float, hi: float) -> float:
    """Score relative delta: 1.0 within [lo, hi], decaying exponentially outside."""
    if lo <= val <= hi:
        return 1.0
    dist = (lo - val) if val < lo else (val - hi)
    span = max(abs(hi - lo), 0.2)
    return float(np.exp(-2.0 * (dist / span)))


class ChallengeVerifier(Component):
    """Component managing challenge issuance and comparative behavioral verification."""

    def __init__(self, version: str = "challenge-v0.1.0") -> None:
        self._name: Final[str] = "challenge"
        self._version = version
        self._is_loaded = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        self._is_loaded = True

    def warmup(self) -> None:
        pass

    def issue(self, challenge_type: str | None = None) -> Challenge:
        """Issue a challenge per 05 §4.2."""
        return issue_challenge(challenge_type)

    def verify(
        self,
        baseline_audio_path: Path | str,
        response_audio_path: Path | str,
        challenge: Challenge,
        response_transcript: str | None = None,
        acoustic_on_response: float | None = None,
    ) -> ChallengeResult:
        """Verify challenge response against baseline audio.

        Args:
            baseline_audio_path: Path to baseline canonical WAV.
            response_audio_path: Path to challenge response canonical WAV.
            challenge: The Challenge instance that was issued.
            response_transcript: Transcript of response audio (for compliance).
            acoustic_on_response: Optional acoustic spoof probability of response.

        Returns:
            ChallengeResult with consistency score and per-feature notes.
        """
        defn = CATALOG.get(challenge.challenge_type)
        if defn is None:
            raise ValueError(f"Unknown challenge type: {challenge.challenge_type}")

        # Check compliance if expected phrase is specified
        compliance_meta = {"required_phrase": challenge.expected_phrase, "matched": True, "similarity": 1.0}
        if challenge.expected_phrase and response_transcript:
            norm_expected = challenge.expected_phrase.lower().strip()
            norm_actual = response_transcript.lower().strip()
            similarity = difflib.SequenceMatcher(None, norm_expected, norm_actual).ratio()
            compliance_meta["similarity"] = round(similarity, 3)

            # Strict phrase match threshold >= 0.70
            if similarity < 0.70:
                compliance_meta["matched"] = False
                return ChallengeResult(
                    consistency_score=0.0,
                    feature_deltas={},
                    expected_deltas=defn.expected_range,
                    passed=False,
                    compliance=compliance_meta,
                    notes=[
                        f"NON_COMPLIANT: Spoken phrase similarity ({similarity:.2f}) fell below 0.70 threshold. "
                        "Response did not match requested challenge phrase."
                    ],
                )

        y_base, sr_base = load_canonical(Path(baseline_audio_path))
        y_resp, sr_resp = load_canonical(Path(response_audio_path))

        fb = extract_acoustic_features(y_base, sr_base)
        fr = extract_acoustic_features(y_resp, sr_resp)

        deltas = {
            k: relative_change(fb[k], fr[k])
            for k in defn.relevant_features
            if k in fb and k in fr
        }

        per_feature_scores = {}
        notes = []

        for k, d in deltas.items():
            if k in defn.expected_range:
                lo, hi = defn.expected_range[k]
                score = graded_falloff(d, lo, hi)
                per_feature_scores[k] = score

                direction = "increased" if d > 0 else "decreased"
                notes.append(
                    f"{k}: {direction} by {abs(d)*100:.1f}% (expected [{lo*100:+.0f}%, {hi*100:+.0f}%], score: {score:.2f})"
                )

        # Behavioral score: weighted mean across relevant features
        weights = [defn.feature_weights.get(k, 1.0) for k in per_feature_scores]
        scores = list(per_feature_scores.values())

        if scores and sum(weights) > 0:
            behavioural = float(np.average(scores, weights=weights))
        else:
            behavioural = 0.50

        # Combine with acoustic detector opinion if available per 05 §4.4
        if acoustic_on_response is not None:
            # 0.6 * behavioural + 0.4 * (1.0 - acoustic_spoof_probability)
            consistency = 0.6 * behavioural + 0.4 * (1.0 - acoustic_on_response)
        else:
            consistency = behavioural

        consistency = max(0.0, min(1.0, consistency))
        passed = consistency >= 0.50

        return ChallengeResult(
            consistency_score=round(consistency, 4),
            feature_deltas={k: round(v, 4) for k, v in deltas.items()},
            expected_deltas=defn.expected_range,
            passed=passed,
            compliance=compliance_meta,
            notes=notes,
        )
