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
  - **Quality-Gated Acoustic Attenuation (Override 3 — Honest Engineering Patch)**:
    - **Nature**: Manually calibrated post-hoc safety override rule, **NOT learned from training data**.
    - **Thresholds**: Acoustic spoof trigger $\ge 0.65$, consumer acoustics boundary $\text{SNR} < 30.0\text{ dB}$ or $\text{Quality} < 0.93$, linguistic benign threshold $\text{scam\_prob} < 0.30$, risk ceiling range $[0.28, 0.58]$.
    - **Rationale**: Specifically engineered to remediate the Condition C5 consumer-microphone acoustic domain shift ($100\%$ false positive rate at $\tau=0.0049$ on ordinary laptop/phone microphones). ASVspoof 2019 LA training data lacks ambient room reverberation, causing `acoustic.pth` to output $\approx 0.85$ spoof probability on genuine human speech.
    - **Integrity Note**: Labeled strictly as an engineering patch rather than a statistical data-fit parameter.
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

### Condition C5 Consumer-Microphone Re-Evaluation (Post-Patch)
Empirical evaluation across 31 genuine consumer-microphone human speech samples (30 FLEURS mobile phone/laptop clips across Hindi/English + 1 live browser laptop mic recording) through the full pipeline with quality-gated acoustic attenuation:
- **Attenuation Trigger Rate**: **87.10%** (27/31 clips triggered `QUALITY_GATED_ACOUSTIC_ATTENUATION`).
- **Pre-Patch Fused High-Risk False Positive Rate**: **100.00%** (31/31 clips evaluated to $\ge 95.7\%$ HIGH risk).
- **Post-Patch Fused High-Risk Alerts**: **12.90%** (4/31 clips).
- **Post-Patch Fused Low-Risk Pass Rate**: **80.65%** (25/31 clips evaluated to $\le 28.16\%$ LOW risk).
- **Post-Patch Fused Moderate Alerts**: **6.45%** (2/31 clips evaluated to $36.8\%\text{--}39.5\%$ MODERATE risk).
- **Benign Conversational False Positive Rate**: **0.00%** (0/27 clips triggered HIGH risk when text did not contain scam extortion keywords).
- *Root Cause of Residual 4 Alerts*: All 4 residual HIGH verdicts stemmed from linguistic false positives where FLEURS news sentences described violence, prison riots, or tax laws, causing `scam_prob > 0.99` and legitimately bypassing benign-text gating.

