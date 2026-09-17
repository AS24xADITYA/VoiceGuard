# Model Card: Calibrated Multi-Signal Fusion Engine (`FusionEngine`)

## Model Details
- **Architecture**: L2-regularized logistic regression over the exact 14-dimensional feature vector, followed by non-parametric Isotonic Regression probability calibration.
- **Input Feature Vector (Exact Order)**:
  0. `acoustic_available` {0, 1}
  1. `acoustic_spoof_prob` [0, 1]
  2. `acoustic_uncertainty` [0, 1]
  3. `acoustic_window_std` [0, 1]
  4. `linguistic_available` {0, 1}
  5. `scam_prob` [0, 1]
  6. `scam_max_category` [0, 1]
  7. `scam_n_categories` [0, 1]
  8. `transcript_reliable` {0, 1}
  9. `language_supported` {0, 1}
  10. `transcript_length_norm` [0, 1]
  11. `challenge_available` {0, 1}
  12. `challenge_consistency` [0, 1]
  13. `audio_quality_score` [0, 1]
- **Safety Overrides**:
  - **Single-Branch Cap**: If only one predictive branch is available, verdict is capped at `MODERATE`.
  - **Inconclusive Floor**: If quality gate fails, verdict is forced to `INCONCLUSIVE` regardless of model probabilities.
- **Version**: `0.1.0-fusion-isotonic`

## Intended Use
- **Primary Use**: Fusing orthogonal acoustic, linguistic, and behavioral challenge signals into a calibrated risk score with directional feature contributions.
- **Out-of-Scope Use**:
  - Autonomous criminal determinations.
  - Operation without a trained calibration artifact.

## Training Data & Procedure
- **Training Corpus**: Multi-modal fusion dataset pairing acoustic and linguistic branch predictions across agreement and disagreement conditions.
- **Pipeline**: `StandardScaler -> LogisticRegression(C=1.0, class_weight='balanced') -> CalibratedClassifierCV(method='isotonic', cv=5)`.
- **Training Script**: `backend/ai/fusion/train_fusion.py` & `notebooks/04_train_fusion.ipynb`.

## Performance & Calibration Status
> **PENDING** — No trained fusion artifact exists yet. Model fitting and isotonic calibration will be executed on Google Colab using `notebooks/04_train_fusion.ipynb`. Per `13-TESTING-AND-EVALUATION.md` §12, rule 1, no performance metric is recorded until produced by an empirical evaluation run.

- **Fused EER (Acoustic + Linguistic)**: PENDING
- **Fused EER (with Challenge Verification)**: PENDING
- **Expected Calibration Error (ECE)**: PENDING
- **Brier Score**: PENDING
