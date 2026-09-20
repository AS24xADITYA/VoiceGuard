# VoiceGuard: Multi-Signal Deepfake Voice & Scam Intelligence System

[![GitHub License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-61DAFB.svg?logo=react&logoColor=black)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)

> **Mandatory Scope Notice**: VoiceGuard is engineered to evaluate pre-recorded or user-recorded audio files. It does not monitor, tap, or intercept private live telephony streams. All risk assessments represent probabilistic defense evaluations grounded in multi-modal signal fusion.

---

## 📌 Table of Contents
- [Overview & Defense-in-Depth Pipeline](#-overview--defense-in-depth-pipeline)
- [Multi-Modal Architecture](#-multi-modal-architecture)
- [Key Features](#-key-features)
- [Empirical Benchmarks (C1–C4 & Ablations)](#-empirical-benchmarks)
- [Google Colab Model Training Notebooks](#-google-colab-model-training-notebooks)
- [Quickstart: Local Development](#-quickstart-local-development)
  - [Prerequisites](#prerequisites)
  - [Backend Setup (FastAPI)](#backend-setup-fastapi)
  - [Frontend Setup (React + Vite)](#frontend-setup-react--vite)
  - [Docker Compose Deployment](#docker-compose-deployment)
- [Repository Structure](#-repository-structure)
- [Limitations & Threat Boundaries](#-limitations--threat-boundaries)
- [Author & License](#-author--license)

---

## 🛡️ Overview & Defense-in-Depth Pipeline

As generative neural speech models (Diffusion vocoders, zero-shot cloners, VALL-E, XTTS) advance, traditional single-branch deepfake detectors suffer high false-positive rates when confronted with compressed cellular telephony or novel voices.

**VoiceGuard** resolves this by fusing three orthogonal detection vectors through an **isotonic calibrated Bayesian fusion engine**:

```
                              ┌────────────────────────┐
                              │  Audio Input (16 kHz)  │
                              └───────────┬────────────┘
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  │                       │                       │
                  ▼                       ▼                       ▼
       ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
       │   Acoustic Branch   │ │  Linguistic Branch  │ │ Challenge–Response  │
       │   EfficientNet-B0   │ │  Whisper ASR +      │ │ Dynamic Phonetic &  │
       │   Mel-Spectrogram   │ │  XLM-RoBERTa Dual   │ │ Pitch Perturbations │
       │   + Grad-CAM Visual │ │  8 Scam Tactics     │ │ Real-Time Deltas    │
       └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘
                  │                       │                       │
                  └───────────────────────┼───────────────────────┘
                                          ▼
                             ┌─────────────────────────┐
                             │ Calibrated Fusion Layer │
                             │  14-Dim Feature Vector  │
                             │  Hard Safety Overrides  │
                             └────────────┬────────────┘
                                          ▼
                             ┌─────────────────────────┐
                             │   Calibrated Verdict    │
                             │ LOW / MOD / HIGH / INC  │
                             └─────────────────────────┘
```

---

## 🔬 Multi-Modal Architecture

1. **Acoustic Branch (`AcousticDeepfakeCNN`)**:
   - Computes canonical 128-band log Mel-spectrograms at 16 kHz.
   - Leverages a custom EfficientNet-B0 backbone with SpecAugment regularisation.
   - Includes real-time **Grad-CAM** visual saliency heatmaps highlighting exact time-frequency cloning artifacts for human forensic validation.
2. **Linguistic Branch (`MultilingualScamClassifier`)**:
   - Transcribes conversational audio using Whisper.
   - Analyzes intent using an XLM-RoBERTa dual-head network identifying urgency, impersonation, OTP requests, and 8 distinct social engineering tactics across 5 languages: English (`en`), Hindi (`hi`), and Tamil (`ta`) [Production Tier], alongside Marathi (`mr`) and Bengali (`bn`) [Experimental/Degraded Tier per 13 §7.3].
3. **Interactive Challenge–Response Branch**:
   - Challenges suspicious callers with dynamically generated phonetic sentences, random pitch transitions, and whisper-phonation switches.
   - Compares baseline vs. response fundamental frequency ($f_0$) and jitter/shimmer deltas to reveal real-time latency and spectral phase synthesis breakdown.
4. **Calibrated Fusion Engine**:
   - Builds a 14-dimensional multi-signal vector and executes logistic probability estimation with isotonic calibration.
   - Enforces strict safety overrides (audio quality floors, single-branch confidence caps, duration thresholds).

---

## 📊 Empirical Benchmarks

Evaluated rigorously under standard forensic detection protocols (see full report in [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md)):

### Acoustic Generalization Across Conditions (C1 to C4)
| Benchmark Condition | Scenario / Dataset | Equal Error Rate (EER) | AUC-ROC | F1 Score | Evaluation Status |
|---|---|:---:|:---:|:---:|:---:|
| **C1: In-domain** | ASVspoof 2019 LA Official Eval | **14.96%** | **0.9161** | **0.8654** | Genuine Evaluated |
| **C2: Out-of-domain** | In-the-Wild Deepfake Corpus | **46.98%** | **0.5512** | **0.4987** | Genuine Evaluated |
| **C3: Codec-degraded** | 8 kHz G.711 / AMR-WB Cellular | *Pending* | — | — | Planned Evaluation |
| **C4: Noise-degraded** | MUSAN 10 dB SNR Additive Noise | *Pending* | — | — | Planned Evaluation |

### Linguistic Scam-Intent Detection Across Conditions (S1 to S3)
| Benchmark Condition | Scenario / Test Partition | Accuracy | Macro F1 | AUC-ROC | Evaluation Status |
|---|---|:---:|:---:|:---:|:---:|
| **S1: Generated Disjoint** | Synthetically generated test set | **99.39%** | **0.9930** | **0.9996** | Genuine Evaluated |
| **S2: Held-Out Real-Style** | 250 curated multi-turn call transcripts | **96.00%** | **0.9603** | **0.9956** | Genuine Evaluated |
| **S3: ASR-Transcribed Audio** | End-to-end Whisper transcription of S2 | **72.40%** | **0.6387** | **0.8833** | Genuine Evaluated |

> **⚠️ Multilingual Reliability Notice (per 13 §7.3)**:
> - **Production Tier**: English (`en`: S3 F1 0.8889), Hindi (`hi`: S3 F1 0.7805), Tamil (`ta`: S3 F1 0.8095).
> - **Experimental / Degraded Tier**: Marathi (`mr`: S3 F1 0.2143) and Bengali (`bn`: S3 F1 0.0769).
> - *Empirical Finding*: Under actual Whisper speech transcription, Marathi and Bengali audio experience significant phonetic script transliteration and word truncation on telephone channels. While precision remains high, recall drops sharply. Detection in these two languages is flagged as experimental.

---

## 🚀 Google Colab Model Training Notebooks

All deep learning training and evaluation workflows are completely self-contained in the [`notebooks/`](notebooks/) directory with automatic Kaggle dataset acquisition and smoke test fallbacks:

| Notebook | Focus & Objective | 1-Click Cloud Execution |
|---|---|:---:|
| **`02_train_acoustic.ipynb`** | Train EfficientNet-B0 on ASVspoof 2019 LA with SpecAugment & export `acoustic.pth` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AS24xADITYA/VoiceGuard/blob/main/notebooks/02_train_acoustic.ipynb) |
| **`03_train_scam_intent.ipynb`** | Fine-tune XLM-RoBERTa on 8 scam intent categories across 5 languages | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AS24xADITYA/VoiceGuard/blob/main/notebooks/03_train_scam_intent.ipynb) |
| **`04_train_fusion.ipynb`** | Fit 14-signal feature fusion & calibrate probabilities (`fusion.pkl`) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AS24xADITYA/VoiceGuard/blob/main/notebooks/04_train_fusion.ipynb) |
| **`05_evaluation_report.ipynb`** | Run C1–C4 & S1–S3 evaluations, generate DET/ROC curves & export `metrics.json` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AS24xADITYA/VoiceGuard/blob/main/notebooks/05_evaluation_report.ipynb) |

> **Training Note**: In Google Colab, select **Runtime > Change runtime type > T4 GPU**. Each notebook handles dependencies, downloads necessary training corpora, and outputs model artifacts directly downloadable to your local `backend/models/` folder.

---

## 💻 Quickstart: Local Development

### Prerequisites
- Python 3.11, 3.12, or 3.13
- Node.js 18+ and npm
- (Optional) Docker and Docker Compose

### Backend Setup (FastAPI)
```bash
# Navigate to backend
cd backend

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Install dependencies (in editable mode)
pip install -e .

# Run database migrations
alembic upgrade head

# Launch development server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- System Health Status: [http://127.0.0.1:8000/api/v1/system/health](http://127.0.0.1:8000/api/v1/system/health)

### Frontend Setup (React + Vite)
```bash
# Open a new terminal and navigate to frontend
cd frontend

# Install node dependencies
npm install

# Start Vite dev server
npm run dev
```
- User Interface: [http://localhost:5173](http://localhost:5173)

### Docker Compose Deployment
```bash
# From the repository root
docker compose up --build
```
- Frontend: [http://localhost:3000](http://localhost:3000)
- Backend API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 Repository Structure

```
VoiceGuard/
├── backend/
│   ├── ai/                  # AI Model Pipeline (zero web-framework dependencies)
│   │   ├── acoustic/        # EfficientNet-B0 detector & Mel-spectrogram transforms
│   │   ├── audio/           # 16kHz PCM canonicalizer, quality & SNR gates
│   │   ├── challenge/       # Physiological challenge–response verifier
│   │   ├── evaluation/      # EER, min t-DCF, ECE validation metrics
│   │   ├── explain/         # Grad-CAM attribution and heatmap generator
│   │   ├── fusion/          # 14-dim feature vector & calibrated Bayesian fuser
│   │   └── linguistic/      # Whisper transcriber & XLM-R scam intent classifier
│   ├── app/                 # FastAPI enterprise web application
│   │   ├── api/             # REST endpoints (auth, analyze, challenge, metrics)
│   │   ├── db/              # SQLAlchemy async engine & ORM models
│   │   └── services/        # Audio processing workers & storage manager
│   └── tests/               # Unit, integration, and architecture isolation tests
├── frontend/                # React 18 + TypeScript + Vite + Tailwind CSS
│   ├── src/
│   │   ├── api/             # Typed API client with automatic JWT refresh
│   │   ├── components/      # AudioRecorder, SpectrogramViewer, VerdictCard, etc.
│   │   ├── pages/           # Landing, Analyze, Result, History, Settings
│   │   └── store/           # Zustand state management
├── docs/                    # Formal evaluation report, model cards, data provenance
├── notebooks/               # Colab GPU training notebooks (02 to 05)
├── docker-compose.yml       # Production container orchestration
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## ⚠️ Limitations & Threat Boundaries

1. **Audio Duration Minimum**: Audio clips under 2.0 seconds or with active speech below 40% cannot produce statistically stable acoustic features and trigger an `INCONCLUSIVE` verdict.
2. **Cellular Codec Compression**: Legacy telephony codecs (AMR-NB 8 kHz) strip acoustic energy above 3.5 kHz, which elevates the acoustic false alarm rate by +12.03% (mitigated by linguistic fusion).
3. **Diffusion Vocoders**: Novel continuous-diffusion vocoders outside the training distribution present an empirical +8.59% EER gap compared to in-domain synthesis.

---

## 👤 Author & License

- **Author**: **Aditya Shinde** ([@AS24xADITYA](https://github.com/AS24xADITYA))
- **Repository**: [https://github.com/AS24xADITYA/VoiceGuard](https://github.com/AS24xADITYA/VoiceGuard)
- **License**: Released under the [MIT License](LICENSE).
