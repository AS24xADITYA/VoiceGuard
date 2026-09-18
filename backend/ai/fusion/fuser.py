"""Calibrated multi-signal fusion layer.

Per 05 §5.1-§5.3:
  - Trained L2-regularised logistic regression with isotonic probability calibration
  - Linear interpretable feature contributions for the UI
  - Verdict classification: LOW (<0.30), MODERATE (0.30-0.65), HIGH (>=0.65)
  - Hard Override 1 (Single-branch cap): If only 1 predictive branch is available,
    verdict cannot exceed MODERATE.
  - Hard Override 2 (Inconclusive floor): If quality gate failed, verdict is forced
    to INCONCLUSIVE with specific reasons returned.

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

try:
    import structlog

    log = structlog.get_logger()
except ImportError:
    import logging

    log = logging.getLogger(__name__)

from ai.base import Component, FusionFeatures, FusionResult, QualityReport, Verdict
from ai.fusion.features import FEATURE_NAMES

DEFAULT_THRESHOLD_MODERATE: Final[float] = 0.30
DEFAULT_THRESHOLD_HIGH: Final[float] = 0.65


class FusionEngine(Component):
    """Fusion component combining acoustic, linguistic, and challenge features."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        threshold_moderate: float = DEFAULT_THRESHOLD_MODERATE,
        threshold_high: float = DEFAULT_THRESHOLD_HIGH,
        version: str = "fusion-lr-calibrated-smoketest-v0",
    ) -> None:
        self._name: Final[str] = "fusion"
        self._version = version
        self.model_path = Path(model_path) if model_path else Path("models/fusion-smoketest-v0.pkl")
        self.threshold_moderate = threshold_moderate
        self.threshold_high = threshold_high

        self.model: Any = None
        self.calibrator: Any = None
        self.coefficients: dict[str, float] = {}
        self.intercept: float = 0.0
        self._is_loaded = False
        self._load_ms = 0
        self._load_error: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def is_loaded(self) -> bool:
        return self._is_loaded and bool(self.coefficients)

    def load(self) -> None:
        """Load trained fusion model & calibrator from trained artifact."""
        start = time.perf_counter()

        target_path = None
        if self.model_path:
            candidates = [
                self.model_path,
                Path("backend") / self.model_path,
                Path(__file__).parent.parent.parent / self.model_path,
                Path("models/fusion-smoketest-v0.pkl"),
                Path("backend/models/fusion-smoketest-v0.pkl"),
                Path(__file__).parent.parent.parent / "models" / "fusion-smoketest-v0.pkl",
                Path("models/fusion.pkl"),
                Path("backend/models/fusion.pkl"),
            ]
            for cand in candidates:
                if cand.is_file():
                    target_path = cand
                    break

        if not target_path or not target_path.is_file():
            self._is_loaded = False
            self.model = None
            self.calibrator = None
            self.coefficients = {}
            self._load_error = f"no trained artifact at {self.model_path}"
            log.warning("fusion_artifact_missing", path=str(self.model_path))
            return

        try:
            with open(target_path, "rb") as f:
                data = pickle.load(f)
                self.model = data.get("model")
                self.calibrator = data.get("calibrator")
                if hasattr(self.model, "coef_"):
                    self.coefficients = dict(zip(FEATURE_NAMES, self.model.coef_[0]))
                    self.intercept = float(self.model.intercept_[0])
            self._is_loaded = True
            self._load_error = None
            self._load_ms = int((time.perf_counter() - start) * 1000)
        except Exception as e:
            self._is_loaded = False
            self.model = None
            self.calibrator = None
            self.coefficients = {}
            self._load_error = f"failed to load artifact at {self.model_path}: {e}"
            log.error("fusion_checkpoint_load_failed", error=str(e))

    def warmup(self) -> None:
        if not self._is_loaded:
            self.load()

    def fuse(
        self,
        features: FusionFeatures,
        quality: QualityReport | None = None,
    ) -> FusionResult:
        """Compute calibrated probability, feature contributions, and final verdict with overrides.

        Args:
            features: 14-element FusionFeatures instance.
            quality: Optional QualityReport from the audio quality gate.

        Returns:
            FusionResult with calibrated risk probability, verdict, contributions, and overrides.

        Raises:
            RuntimeError: If fusion model artifact is not loaded.
        """
        if not self.is_loaded():
            err = self._load_error or f"no trained artifact at {self.model_path}"
            raise RuntimeError(f"MODEL_ERROR: Fusion layer is not loaded ({err})")

        feat_vector = features.to_vector()
        feat_dict = dict(zip(FEATURE_NAMES, feat_vector))

        # 1. Compute raw log-odds score
        log_odds = self.intercept
        contributions: list[dict[str, Any]] = []

        for name, val in feat_dict.items():
            coef = self.coefficients.get(name, 0.0)
            contrib = coef * (val - 0.5)  # zero-centered contribution
            log_odds += contrib

            direction = "increases_risk" if contrib > 0 else "decreases_risk"
            contributions.append({
                "feature": name,
                "value": round(val, 3),
                "contribution": round(contrib, 3),
                "direction": direction,
            })

        # Sort contributions by absolute magnitude descending
        contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

        # 2. Probability computation & calibration
        raw_prob = float(1.0 / (1.0 + np.exp(-log_odds)))

        if self.calibrator is not None:
            try:
                calibrated_prob = float(self.calibrator.predict([raw_prob])[0])
            except Exception:
                calibrated_prob = raw_prob
        else:
            calibrated_prob = raw_prob

        calibrated_prob = float(np.clip(calibrated_prob, 0.0, 1.0))

        # 3. Base verdict mapping per 05 §5.3
        if calibrated_prob >= self.threshold_high:
            verdict = Verdict.HIGH
        elif calibrated_prob >= self.threshold_moderate:
            verdict = Verdict.MODERATE
        else:
            verdict = Verdict.LOW

        reasons: list[str] = []
        overrides_applied: list[str] = []

        # 4. Hard Override 1: Single-branch cap per 05 §5.3
        # If only one predictive branch is available, verdict cannot exceed MODERATE
        num_predictive_branches = int(features.acoustic_available + features.linguistic_available)
        if num_predictive_branches <= 1 and verdict == Verdict.HIGH:
            verdict = Verdict.MODERATE
            overrides_applied.append("SINGLE_BRANCH_CAP")
            reasons.append(
                "Single-branch cap applied: Risk probability exceeds high threshold, but assessment "
                "relies on only one predictive branch. Verdict capped at MODERATE."
            )

        # 5. Hard Override 2: Inconclusive floor per 05 §5.3
        # If the audio quality gate failed, force verdict to INCONCLUSIVE
        if quality is not None and not quality.passed:
            verdict = Verdict.INCONCLUSIVE
            overrides_applied.append("INCONCLUSIVE_QUALITY_FLOOR")
            reasons.append(
                f"Audio quality gate failed: {'; '.join(quality.failures)}. "
                "Verdict forced to INCONCLUSIVE."
            )

        # Confidence: distance from nearest decision boundary
        if verdict == Verdict.HIGH:
            conf = min(1.0, (calibrated_prob - self.threshold_high) / (1.0 - self.threshold_high + 1e-6))
        elif verdict == Verdict.MODERATE:
            mid = (self.threshold_high + self.threshold_moderate) / 2.0
            conf = 1.0 - abs(calibrated_prob - mid) / (mid - self.threshold_moderate + 1e-6)
        else:
            conf = min(1.0, (self.threshold_moderate - calibrated_prob) / (self.threshold_moderate + 1e-6))

        return FusionResult(
            risk_probability=round(calibrated_prob, 4),
            verdict=verdict,
            confidence=round(float(conf), 3),
            feature_vector=feat_dict,
            contributions=contributions,
            reasons=reasons,
            overrides_applied=overrides_applied,
            thresholds={
                "moderate": self.threshold_moderate,
                "high": self.threshold_high,
            },
            model_version=self.version,
        )
