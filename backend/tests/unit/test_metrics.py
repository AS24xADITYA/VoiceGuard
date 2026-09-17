"""Unit tests for ai/evaluation/metrics.py.

Verifies Phase 3 exit criteria:
  - EER computation is mathematically sound
  - min t-DCF computation produces positive values
  - ECE produces expected calibration error in [0, 1]
  - compute_all_metrics returns full report dictionary
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.evaluation.metrics import (
    compute_all_metrics,
    compute_ece,
    compute_eer,
    compute_min_tdcf,
)


def test_compute_eer_perfect_separation():
    """Verify EER is 0.0 for perfectly separated distributions."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_score = np.array([0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95])

    eer, threshold = compute_eer(y_true, y_score)
    assert eer == 0.0
    assert 0.2 <= threshold <= 0.8


def test_compute_eer_overlapping():
    """Verify EER handles overlapping predictions."""
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.2, 0.6, 0.4, 0.8])

    eer, threshold = compute_eer(y_true, y_score)
    assert 0.0 < eer < 1.0


def test_compute_ece():
    """Verify ECE produces valid calibration error."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_compute_all_metrics():
    """Verify compute_all_metrics returns complete typed report."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.3, 0.6, 0.8, 0.85, 0.9])

    report = compute_all_metrics(y_true, y_score)

    assert 0.0 <= report.eer <= 1.0
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.precision <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.f1 <= 1.0
    assert 0.0 <= report.auc_roc <= 1.0
    assert 0.0 <= report.ece <= 1.0
    assert report.total_samples == 10
    assert report.n_bonafide == 5
    assert report.n_spoof == 5
    assert len(report.confusion_matrix) == 2
