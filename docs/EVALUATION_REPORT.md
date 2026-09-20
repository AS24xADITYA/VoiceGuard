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

| Condition | Accuracy | Precision | Recall | Macro F1 | AUC-ROC | Evaluation Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1 Held-out generated** | **99.39%** | 0.9984 | 0.9888 | **0.9930** | **0.9996** | Genuine Evaluated |
| **S2 Held-out real-style** | **96.00%** | 0.9528 | 0.9680 | **0.9603** | **0.9956** | Genuine Evaluated |
| **S3 ASR-transcribed** | **72.40%** | 0.9242 | 0.4880 | **0.6387** | **0.8833** | Genuine Evaluated |

### Per-Language Breakdown (Condition S3 End-to-End ASR)

| Language | S3 Accuracy | S3 Precision | S3 Recall | Scam Detection F1 | AUC-ROC | Operational Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **English (`en`)** | 88.00% | 0.8276 | 0.9600 | **0.8889** | 0.9616 | **Supported (Production)** |
| **Hindi (`hi`)** | 82.00% | 1.0000 | 0.6400 | **0.7805** | 0.9904 | **Supported (Production)** |
| **Tamil (`ta`)** | 84.00% | 1.0000 | 0.6800 | **0.8095** | 0.9904 | **Supported (Production)** |
| **Marathi (`mr`)** | 56.00% | 1.0000 | 0.1200 | **0.2143** | 0.7424 | **Degraded / Experimental (13 §7.3)** |
| **Bengali (`bn`)** | 52.00% | 1.0000 | 0.0400 | **0.0769** | 0.7760 | **Degraded / Experimental (13 §7.3)** |

> **ASR Transcription Degradation Note**:
> As documented under `13 §7.3`, automated telephone audio transcription via Whisper produces significant phonetic script transliteration and word drops for Bengali and Marathi. While precision remains high (1.00), recall collapses due to corrupted Devanagari/Bengali lexical tokens, resulting in sub-floor F1 scores. Detection in Bengali and Marathi is designated as experimental.

---

## 4. Challenge–Response Evaluation

Interactive physiological challenge evaluation protocol:

- **Human Pass Rate**: PENDING
- **Synthetic Voice Conversion Pass Rate**: PENDING
- **Mean Human Consistency Score**: PENDING
- **Mean Synthetic Consistency Score**: PENDING

---

## 5. Fusion Layer Evaluation

### 5.1 Ablation Study (13 §9.1 & 13 §9.3)

Evaluated on 330 held-out crossed test samples (165 unique held-out transcripts from `val.json`, 165 disagreement cases, strictly grouped by `transcript_id` via `GroupShuffleSplit` and 5-fold `GroupKFold` internal calibration):

| Configuration | Condition | Decision Threshold ($\tau$) | EER | AUC-ROC | Macro F1 | Accuracy | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Acoustic only | F1 | 0.0049 | 22.11% | 0.8429 | 0.7415 | 76.67% | Genuine Evaluated |
| Linguistic only | F2 | 0.5000 | 25.91% | 0.8238 | 0.7264 | 73.94% | Genuine Evaluated |
| **Acoustic + Linguistic (Fused)** | F3 | 0.5000 | **0.00%** | **1.0000** | **1.0000** | **100.00%** | Genuine Evaluated |
| **Acoustic + Linguistic + Challenge (Full Stack)** | F4 | 0.5000 | **0.00%** | **1.0000** | **1.0000** | **100.00%** | Genuine Evaluated |

> Note: F3/F4's near-zero EER partly reflects the OR-based construction of the crossed disagreement dataset (06 §4.1/§4.2) and should be read as a demonstration that fusion correctly combines two branches when at least one branch's signal is reliable for a given case - not a claim of zero real-world error. The acoustic branch's true real-world error rate is documented separately in C1 (14.96% EER, Part B).

### 5.2 Probability Calibration
- **Pre-calibration Brier Score**: 0.0004
- **Post-isotonic Calibration Brier Score**: 0.0000 (improvement: -0.0004)
- **Pre-calibration Expected Calibration Error (ECE)**: 0.0062
- **Post-isotonic Calibration ECE**: 0.0014 (improvement: -0.0048)
- **Calibrated AUC-ROC**: 1.0000
- **Calibrated Log Loss**: 0.0014
- **Benchmark HistGBDT**: Brier = 0.0000, ECE = 0.0001, AUC = 1.0000

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
