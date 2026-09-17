# VoiceGuard — Comprehensive Evaluation Report

This report documents the empirical evaluation framework for the VoiceGuard multi-signal deepfake voice and scam detection pipeline, following the testing and evaluation protocol specified in `13-TESTING-AND-EVALUATION.md`.

> **CRITICAL STATUS NOTICE**:
> **PENDING** — No trained model artifact exists yet. Model training will be executed on Google Colab using `notebooks/02_train_acoustic.ipynb`, `notebooks/03_train_scam_intent.ipynb`, and `notebooks/04_train_fusion.ipynb`. Per `13-TESTING-AND-EVALUATION.md` §12, rule 1: *No metric appears anywhere in this repository without a corresponding artifact from a real evaluation run.* All metric tables below define the standardized evaluation protocol and will be populated upon completion of the Colab training and evaluation run (`notebooks/05_evaluation_report.ipynb`).

---

## 1. Experimental Setup

- **Hardware Target**: Google Colab GPU runtime (NVIDIA T4 / V100).
- **Random Seeds**: Pinned to `42` across NumPy, PyTorch, and scikit-learn.
- **Acoustic Training Partition**: ASVspoof 2019 Logical Access (LA) train set (25,380 utterances, 20 speakers).
- **Acoustic Validation Partition**: ASVspoof 2019 LA dev set (24,844 utterances, 10 speakers, strictly speaker-disjoint).
- **Acoustic Evaluation Partition**: ASVspoof 2019 LA eval set (71,196 utterances, 48 speakers, strictly speaker-disjoint).
- **Out-of-Domain Corpus**: In-the-Wild dataset (unseen modern generative architectures and neural vocoders).
- **Scam-Intent Corpus**: Multi-turn synthesized and curated extortion dialogue transcripts across English, Hindi, and regional languages.

---

## 2. Acoustic Classifier Evaluation (Conditions C1 to C4)

Evaluation conditions specified in `13 §6.1` to assess in-domain discrimination, zero-shot generalization, and acoustic transmission degradation:

| Condition | EER | AUC-ROC | F1-Score | min t-DCF | Evaluation Status |
|---|---|---|---|---|---|
| **C1 In-domain** | PENDING | PENDING | PENDING | PENDING | Pending execution of Colab notebook `02` |
| **C2 Out-of-domain** | PENDING | PENDING | PENDING | PENDING | Pending execution of Colab notebook `02` |
| **C3 Codec-degraded** | PENDING | PENDING | PENDING | PENDING | Pending execution of Colab notebook `05` |
| **C4 Noise-degraded** | PENDING | PENDING | PENDING | PENDING | Pending execution of Colab notebook `05` |

### C1 → C2 Generalization Gap Analysis
- **Status**: PENDING Colab evaluation.
- **Objective**: Measure performance divergence between the controlled ASVspoof corpus (older synthesizers) and unseen modern in-the-wild audio (neural vocoders like HiFi-GAN, BigVGAN, and diffusion models).

---

## 3. Scam-Intent Classifier Evaluation (Conditions S1 to S3)

Dual-head XLM-RoBERTa evaluation protocol for overall extortion classification and 8 tactic categories:

| Condition | Accuracy | Macro F1 | Evaluation Status |
|---|---|---|---|
| **S1 Held-out generated** | PENDING | PENDING | Pending execution of Colab notebook `03` |
| **S2 Held-out real-style** | PENDING | PENDING | Pending execution of Colab notebook `03` |
| **S3 ASR-transcribed** | PENDING | PENDING | Pending execution of Colab notebook `03` |

### Per-Language Breakdown (Condition S3 End-to-End)

| Language | Audio WER | Scam Detection F1 | Status |
|---|---|---|---|
| English (`en`) | PENDING | PENDING | Supported |
| Hindi (`hi`) | PENDING | PENDING | Supported |
| Marathi (`mr`) | PENDING | PENDING | Supported |
| Bengali (`bn`) | PENDING | PENDING | Supported |
| Tamil (`ta`) | PENDING | PENDING | Documented Degraded Reliability |

---

## 4. Challenge–Response Evaluation

Interactive physiological challenge evaluation protocol:

- **Human Pass Rate**: PENDING
- **Synthetic Voice Conversion Pass Rate**: PENDING
- **Mean Human Consistency Score**: PENDING
- **Mean Synthetic Consistency Score**: PENDING

---

## 5. Fusion Layer Evaluation

### 5.1 Ablation Study

| Configuration | EER | AUC-ROC | Macro F1 |
|---|---|---|---|
| Acoustic only | PENDING | PENDING | PENDING |
| Linguistic only | PENDING | PENDING | PENDING |
| **Acoustic + Linguistic (Fused)** | PENDING | PENDING | PENDING |
| **Acoustic + Linguistic + Challenge (Full Pipeline)** | PENDING | PENDING | PENDING |

### 5.2 Probability Calibration
- **Pre-calibration Expected Calibration Error (ECE)**: PENDING
- **Post-isotonic Calibration ECE**: PENDING
- **Brier Score**: PENDING

---

## 6. End-to-End Latency & Resource Budgets

Target budgets defined in `03 §3` against which real latency will be measured during smoke testing:

| Component | CPU Baseline | GPU Accelerated | Target Budget (03 §3) |
|---|---|---|---|
| Audio Prep (10s audio) | PENDING | PENDING | ≤ 0.5 s |
| Acoustic CNN (10s audio) | PENDING | PENDING | ≤ 2.0 s |
| Whisper Transcription (10s audio) | PENDING | PENDING | ≤ 20.0 s |
| Scam Intent Classification | PENDING | PENDING | ≤ 1.5 s |
| Grad-CAM Generation | PENDING | PENDING | ≤ 3.0 s |
| **Total End-to-End Latency** | PENDING | PENDING | **≤ 30.0 s** |
| Peak RAM Usage | PENDING | PENDING | ≤ 4.0 GB |

---

## 7. Limitations & Scope

1. **Non-Real-Time Protection**: VoiceGuard is designed for post-call or recorded file analysis. It does not tap or monitor live telephony calls.
2. **Probabilistic Forensics**: All outputs represent estimated risks based on empirical training correlations, not judicial or biometric proof of identity.
3. **Codec Vulnerability**: Heavy cellular compression (e.g., AMR-NB @ 4.75 kbps) limits acoustic reliability, necessitating multi-modal linguistic and challenge verification.
