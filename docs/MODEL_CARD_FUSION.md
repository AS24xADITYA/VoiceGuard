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
Evaluated per `13-TESTING-AND-EVALUATION.md` §9 on 330 held-out crossed test samples (165 unique held-out transcripts from `val.json`, 165 disagreement cases, strictly grouped by `transcript_id` with 5-fold `GroupKFold` calibration):

- **Dataset Composition**: 1,650 total rows (825 unique transcripts from `val.json` strictly, 2x reuse cap, 825 disagreement cases [50.0%], 1,650 unique ASVspoof 2019 LA DEV clips).
- **Cross-Validation**: 5-fold `GroupKFold` keyed on `transcript_id` (zero fold leakage).
- **Pre-Calibration Brier Score**: 0.0004 | **Post-Isotonic Brier Score**: **0.0000**
- **Pre-Calibration ECE**: 0.0062 | **Post-Isotonic ECE**: **0.0014**
- **Calibrated AUC-ROC**: **1.0000**
- **Benchmark HistGBDT**: Brier = 0.0000, ECE = 0.0001, AUC = 1.0000

### 4-Condition Ablation Study (13 §9.1)
- **F1 (Acoustic Only, $\tau=0.0049$)**: EER = 22.11%, AUC-ROC = 0.8429, Macro F1 = 0.7415, Accuracy = 76.67%
- **F2 (Linguistic Only, $\tau=0.5000$)**: EER = 25.91%, AUC-ROC = 0.8238, Macro F1 = 0.7264, Accuracy = 73.94%
- **F3 (Acoustic + Linguistic Fused)**: EER = **0.00%**, AUC-ROC = **1.0000**, Macro F1 = **1.0000**, Accuracy = **100.00%**
- **F4 (Full Stack Fused + Challenge)**: EER = **0.00%**, AUC-ROC = **1.0000**, Macro F1 = **1.0000**, Accuracy = **100.00%**

> **Real-World Generalization Caveat**:
> Note: F3/F4's near-zero EER partly reflects the OR-based construction of the crossed disagreement dataset (06 §4.1/§4.2) and should be read as a demonstration that fusion correctly combines two branches when at least one branch's signal is reliable for a given case - not a claim of zero real-world error. The acoustic branch's true real-world error rate is documented separately in C1 (14.96% EER, Part B).
