"""Evaluation metrics computation: EER, min t-DCF, F1, AUC, ECE, and confusion matrix.

Per 05 §2.4, 06 §1, 10 §Phase 3, and 13 §1.
All metrics are computed on held-out evaluation sets.

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn import metrics


@dataclass(frozen=True)
class EvaluationMetrics:
    """Comprehensive evaluation metrics report."""

    eer: float
    eer_threshold: float
    min_tdcf: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    ece: float  # Expected Calibration Error
    confusion_matrix: list[list[int]]  # [[TN, FP], [FN, TP]]
    total_samples: int
    n_bonafide: int
    n_spoof: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and optimal threshold.

    Args:
        y_true: Binary ground truth labels (0 = bona fide, 1 = spoof).
        y_score: Continuous predicted scores (spoof probabilities).

    Returns:
        (eer, threshold) where eer is in [0, 1].
    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    # Find point where FPR and FNR intersect
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx])
    return eer, threshold


def compute_min_tdcf(
    y_true: np.ndarray,
    y_score: np.ndarray,
    p_spoof: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> float:
    """Compute normalized minimum detection cost function (min t-DCF approximation).

    Per standard biometric detection cost formula:
      C_det(theta) = C_miss * P_spoof * P_miss(theta) + C_fa * (1 - P_spoof) * P_fa(theta)
      C_default = min(C_miss * P_spoof, C_fa * (1 - P_spoof))
      min_tdcf = min_theta (C_det(theta) / C_default)

    Returns:
        Normalized min t-DCF float.
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr

    c_det = c_miss * p_spoof * fnr + c_fa * (1.0 - p_spoof) * fpr
    c_default = min(c_miss * p_spoof, c_fa * (1.0 - p_spoof))
    if c_default <= 0:
        return 0.0

    min_cost = float(np.min(c_det) / c_default)
    return round(min_cost, 4)


def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        y_true: Binary ground truth (0 or 1).
        y_prob: Predicted probability for class 1.
        n_bins: Number of equal-width probability bins.

    Returns:
        ECE value in [0, 1].
    """
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_limits[i]
        bin_upper = bin_limits[i + 1]
        mask = (y_prob >= bin_lower) & (
            y_prob <= bin_upper if i == n_bins - 1 else y_prob < bin_upper
        )
        bin_count = np.sum(mask)

        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_count / n_samples) * abs(bin_acc - bin_conf)

    return round(float(ece), 4)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
) -> EvaluationMetrics:
    """Compute the full suite of evaluation metrics per specification.

    Args:
        y_true: Ground truth binary labels (0 = bona fide, 1 = spoof).
        y_score: Predicted spoof probabilities in [0, 1].
        threshold: Decision threshold for discrete classification.
                   If None, uses the EER threshold.

    Returns:
        EvaluationMetrics instance.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    eer, eer_thresh = compute_eer(y_true, y_score)
    eval_threshold = threshold if threshold is not None else eer_thresh

    y_pred = (y_score >= eval_threshold).astype(int)

    acc = float(metrics.accuracy_score(y_true, y_pred))
    prec = float(metrics.precision_score(y_true, y_pred, zero_division=0))
    rec = float(metrics.recall_score(y_true, y_pred, zero_division=0))
    f1 = float(metrics.f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(metrics.roc_auc_score(y_true, y_score))
    except ValueError:
        auc = 0.0

    cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    ece = compute_ece(y_true, y_score)
    min_tdcf = compute_min_tdcf(y_true, y_score)

    return EvaluationMetrics(
        eer=round(eer, 4),
        eer_threshold=round(eer_thresh, 4),
        min_tdcf=min_tdcf,
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        auc_roc=round(auc, 4),
        ece=ece,
        confusion_matrix=cm,
        total_samples=len(y_true),
        n_bonafide=int(np.sum(y_true == 0)),
        n_spoof=int(np.sum(y_true == 1)),
    )


# Backward compatibility aliases
compute_min_t_dcf = compute_min_tdcf
evaluate_binary_predictions = compute_all_metrics
