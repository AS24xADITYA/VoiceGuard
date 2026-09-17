"""Fusion layer calibration and training pipeline.

Per 05 §5.1 & 06 §4:
  - Fits L2-regularised Logistic Regression with Isotonic probability calibration
  - Compares performance against HistGradientBoostingClassifier
  - Evaluates Brier score, Log Loss, ROC-AUC, and Expected Calibration Error (ECE)
  - Exports trained calibrator artifact

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from ai.evaluation.metrics import compute_ece
from ai.fusion.features import FEATURE_NAMES


def train_fusion_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Train logistic regression with isotonic calibration and benchmark against GBDT."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Base Logistic Regression (L2 penalty)
    base_lr = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    base_lr.fit(x_train, y_train)

    # 2. Isotonic Calibration
    calibrated_lr = CalibratedClassifierCV(estimator=base_lr, method="isotonic", cv="prefit")
    calibrated_lr.fit(x_val, y_val)

    # Predictions on validation set
    p_val_raw = base_lr.predict_proba(x_val)[:, 1]
    p_val_cal = calibrated_lr.predict_proba(x_val)[:, 1]

    # Benchmark: HistGradientBoostingClassifier per 05 §5.1
    gbdt = HistGradientBoostingClassifier(random_state=42)
    gbdt.fit(x_train, y_train)
    p_val_gbdt = gbdt.predict_proba(x_val)[:, 1]

    # Metrics
    metrics_summary = {
        "logistic_regression": {
            "raw_brier": round(float(brier_score_loss(y_val, p_val_raw)), 4),
            "calibrated_brier": round(float(brier_score_loss(y_val, p_val_cal)), 4),
            "calibrated_log_loss": round(float(log_loss(y_val, p_val_cal)), 4),
            "calibrated_auc": round(float(roc_auc_score(y_val, p_val_cal)), 4),
            "calibrated_ece": round(float(compute_ece(y_val, p_val_cal)), 4),
            "coefficients": dict(zip(FEATURE_NAMES, [round(float(c), 4) for c in base_lr.coef_[0]])),
            "intercept": round(float(base_lr.intercept_[0]), 4),
        },
        "gradient_boosting_benchmark": {
            "brier": round(float(brier_score_loss(y_val, p_val_gbdt)), 4),
            "log_loss": round(float(log_loss(y_val, p_val_gbdt)), 4),
            "auc": round(float(roc_auc_score(y_val, p_val_gbdt)), 4),
            "ece": round(float(compute_ece(y_val, p_val_gbdt)), 4),
        },
    }

    # Save model artifact
    model_artifact = {
        "model": base_lr,
        "calibrator": calibrated_lr,
        "metrics": metrics_summary,
    }
    with open(out_path / "fusion_model.pkl", "wb") as f:
        pickle.dump(model_artifact, f)

    with open(out_path / "fusion_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Fusion training complete. Calibrated LR Brier score: {metrics_summary['logistic_regression']['calibrated_brier']}")
    return metrics_summary
