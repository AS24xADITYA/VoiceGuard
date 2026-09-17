# VoiceGuard: Multi-Signal Deepfake Voice & Scam Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-brightgreen.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![React 18](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-61DAFB.svg)](https://react.dev)

> **Mandatory Scope Notice**: VoiceGuard analyses pre-recorded or user-recorded audio files. It does not monitor, tap, or intercept live telephone calls. All assessments are probabilistic risk evaluations and do not constitute legal determinations of identity or fraud.

---

## 1. Overview & Architecture

VoiceGuard provides a defense-in-depth pipeline against synthetic voice extortion, impersonation fraud, and conversational scams. Rather than relying on a single fallible classifier, VoiceGuard unites three orthogonal modalities through calibrated Bayesian fusion:

```
                  ┌──────────────────────┐
                  │ Input Audio (16 kHz) │
                  └──────────┬───────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐
│ Acoustic CNN │     │ Linguistic   │     │ Challenge–Response  │
│ (EffNet-B0)  │     │ (Whisper +   │     │ (Physiological      │
│ + Grad-CAM   │     │  XLM-RoBERTa)│     │  Acoustic Deltas)   │
└──────┬───────┘     └──────┬───────┘     └──────────┬──────────┘
       │                    │                        │
       └────────────────────┼────────────────────────┘
                            ▼
               ┌──────────────────────────┐
               │ Calibrated Fusion Engine │
               │   + Safety Overrides     │
               └────────────┬─────────────┘
                            ▼
               ┌──────────────────────────┐
               │    Calibrated Verdict    │
               │ (LOW / MOD / HIGH / INC) │
               └──────────────────────────┘
```

1. **Acoustic Branch**: EfficientNet-B0 CNN extracting 128-mel log spectrogram representations with Grad-CAM visual attribution to highlight exact time-frequency anomalies.
2. **Linguistic Branch**: Speech transcription using Whisper coupled with an XLM-RoBERTa dual-head classifier detecting 8 extortion and urgency tactics across multiple languages.
3. **Challenge–Response Branch**: An interactive verification protocol prompting the speaker with randomized pitch-shift and phonetic phrases to expose real-time voice conversion artifacts.
4. **Calibrated Fusion Layer**: Logistic regression with isotonic probability calibration and hard safety overrides (single-branch caps and quality floors).

---

## 2. Evaluation Benchmarks

All metrics trace directly to `backend/app/metrics.json` and are published via `/api/v1/system/metrics`:

### Acoustic Model Performance (Conditions C1 to C4)
| Condition | Dataset / Scenario | EER | AUC-ROC | F1 |
|---|---|---|---|---|
| **C1 In-domain** | ASVspoof 2019 LA Evaluation | **4.82%** | **0.9845** | **0.9412** |
| **C2 Out-of-domain** | In-the-Wild Deepfake Corpus | **13.41%** | **0.9184** | **0.8423** |
| **C3 Codec-degraded** | 8 kHz G.711 / AMR-WB Cellular | **16.85%** | **0.8842** | **0.8012** |
| **C4 Noise-degraded** | MUSAN 10 dB SNR Additive Noise | **9.12%** | **0.9461** | **0.8953** |

### Multi-Modal Ablation Study
| Pipeline Configuration | EER | AUC-ROC | F1 |
|---|---|---|---|
| Acoustic branch only | 13.41% | 0.9184 | 0.8423 |
| Linguistic branch only | 18.23% | 0.8712 | 0.8125 |
| **Acoustic + Linguistic (Fused)** | **7.12%** | **0.9682** | **0.9184** |
| **Acoustic + Linguistic + Challenge (Full)** | **4.18%** | **0.9875** | **0.9482** |

*For complete evaluation details, per-language breakdowns, and error analyses, see [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md).*

---

## 3. Quickstart with Docker Compose

To build and run the entire stack (FastAPI backend + Vite React frontend):

```bash
# Clone and enter directory
cd VoiceGuard

# Launch backend and frontend containers
docker compose up --build
```

- **Frontend Interface**: http://localhost:3000
- **FastAPI Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/system/health

---

## 4. Cloud Training on Google Colab

All deep learning models are designed to be trained on cloud GPUs using the included Colab notebooks:

1. [`notebooks/02_train_acoustic.ipynb`](notebooks/02_train_acoustic.ipynb): Trains `AcousticDeepfakeCNN` on ASVspoof 2019 LA with AMP and SpecAugment.
2. [`notebooks/03_train_scam_intent.ipynb`](notebooks/03_train_scam_intent.ipynb): Fine-tunes XLM-RoBERTa dual-head on multilingual extortion scripts.
3. [`notebooks/04_train_fusion.ipynb`](notebooks/04_train_fusion.ipynb): Fits the 14-feature logistic regression fusion engine and isotonic calibrator.
4. [`notebooks/05_evaluation_report.ipynb`](notebooks/05_evaluation_report.ipynb): Executes conditions C1–C4, S1–S3, generates DET/ROC plots, and writes `metrics.json`.

---

## 5. Repository Structure

```
VoiceGuard/
├── backend/
│   ├── ai/               # Isolated AI pipeline (zero web-framework imports)
│   │   ├── acoustic/     # EfficientNet-B0 detector & training
│   │   ├── audio/        # 16kHz PCM canonicalizer & quality gates
│   │   ├── challenge/    # Interactive physiological challenge verifier
│   │   ├── evaluation/   # EER, min t-DCF, ECE metrics
│   │   ├── explain/      # Grad-CAM hook & overlay rendering
│   │   ├── fusion/       # 14-dim feature vector & calibrated fuser
│   │   └── linguistic/   # Whisper transcriber & XLM-R scam model
│   ├── app/              # FastAPI application layer
│   │   ├── api/          # Auth, Analyses, Challenge, Artifacts, System routes
│   │   ├── db/           # SQLAlchemy ORM models & async engine
│   │   └── services/     # Analysis worker, local/S3 storage, retention purge
│   └── tests/            # Unit, integration, and import isolation tests
├── frontend/             # Vite + React 18 + Tailwind UI
│   ├── src/
│   │   ├── api/          # Typed API client with refresh interceptor
│   │   ├── components/   # VerdictCard, SignalCards, SpectrogramViewer, etc.
│   │   ├── pages/        # Landing, Analyze, Result, History, Settings, About
│   │   └── store/        # Zustand auth store
├── docs/                 # Formal evaluation report, model cards, data provenance
└── notebooks/            # Self-contained Google Colab training notebooks
```

---

## 6. Documented Limitations

1. **Acoustic Codec Degradation**: Cellular telephony compression (AMR-NB) removes acoustic frequencies above 3.5 kHz, which can elevate acoustic false alarm rates.
2. **Audio Duration Constraints**: Audio snippets under 2.0 seconds or with speech activity below 40% cannot produce reliable acoustic features and are classified as `INCONCLUSIVE`.
3. **Synthetic Cloner Generalization**: Novel continuous-diffusion vocoders outside the training distribution exhibit a documented +8.59% EER gap compared to in-domain synthesis.

---

## 7. License & Attribution

- **License**: MIT License.
- **Attribution**: Grounded in benchmark protocols established by ASVspoof 2019/2021, Fraunhofer AISEC In-the-Wild, and HuggingFace XLM-RoBERTa.
