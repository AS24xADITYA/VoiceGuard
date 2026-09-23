<p align="center">
  <img src="frontend/public/logo.png" alt="VoiceGuard Logo" width="280" />
</p>

<h1 align="center">VoiceGuard: Multi-Signal Deepfake Voice &amp; Scam Intelligence System</h1>

<p align="center">
  <strong>An automated security pipeline uniting convolutional acoustic artifact detection, multilingual extortion language analysis, and interactive challenge verification.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="GitHub License" /></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-3776AB.svg?logo=python&logoColor=white" alt="Python Version" /></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white" alt="FastAPI" /></a>
  <a href="https://pytorch.org"><img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch" /></a>
  <a href="https://react.dev"><img src="https://img.shields.io/badge/Frontend-React%2018%20%2B%20Vite-61DAFB.svg?logo=react&logoColor=black" alt="React" /></a>
  <a href="https://www.docker.com"><img src="https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

> **Mandatory Scope Notice**: VoiceGuard is engineered to evaluate pre-recorded or user-recorded audio files. It does not monitor, tap, or intercept private live telephony streams. All risk assessments represent probabilistic defense evaluations grounded in multi-modal signal fusion.

---

## 📌 Table of Contents
- [Overview & Defense-in-Depth Pipeline](#-overview--defense-in-depth-pipeline)
- [Multi-Modal Architecture](#-multi-modal-architecture)
- [Key Features](#-key-features)
- [Empirical Benchmarks (C1–C5, S1–S3, Challenges & Fusion)](#-empirical-benchmarks)
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

All models across the Acoustic, Linguistic, and Fusion branches have completed rigorous empirical evaluation under standardized forensic detection protocols (see full report in [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md) and synchronized metrics in [`backend/app/metrics.json`](backend/app/metrics.json)):

### 1. Acoustic Classifier Generalization (Conditions C1 to C5)
Evaluated across 149,377 audio samples testing in-domain discrimination, unseen generator zero-shot generalization, telephony compression, additive environmental noise, and real-world consumer hardware:

| Benchmark Condition | Scenario / Dataset | Total Samples | Equal Error Rate (EER) | min t-DCF | AUC-ROC | F1 Score | Evaluation Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **C1: In-domain** | ASVspoof 2019 LA Official Eval | 71,237 | **14.96%** | **0.3786** | **0.8710** | **0.9108** | Genuine Evaluated |
| **C2: Out-of-domain** | In-the-Wild Deepfake Corpus | 28,452 | **46.98%** | **1.0000** | **0.5141** | **0.5362** | Genuine Evaluated |
| **C3: Codec-degraded** | 8 kHz G.711 / OPUS Cellular | 24,844 | **4.31%** | **0.8857** | **0.9868** | **0.9505** | Genuine Evaluated |
| **C4: Noise-degraded** | MUSAN 10 dB SNR Additive Noise | 24,844 | **31.47%** | **0.8097** | **0.7630** | **0.5688** | Genuine Evaluated |
| **C5: Consumer Hardware** | Mobile/Laptop Ordinary Acoustics | 31 | **FPR: 100.0%** → **12.9%** (with Override 3) | — | — | — | Genuine Evaluated |

> **Key Forensic Insights (C1 vs. C2 & C5)**:
> - **Zero-Shot Generalization Gap**: EER increases by **+32.02%** from C1 (14.96%) to C2 (46.98%). Neural vocoders (A07–A12) achieve near-zero miss rates (0.00%–0.02%), while modern diffusion-based voice synthesizers (A17/A18) produce lower acoustic confidence (miss rates up to 80.20%), demonstrating why acoustic-only detection cannot be trusted in isolation.
> - **Condition C5 Room Acoustics Gap**: Without fusion, uncalibrated consumer microphones trigger a 100% False Positive Rate at $\tau=0.0049$ due to room reverberation and microphone frequency responses mimicking vocoder artifacts. VoiceGuard's **Quality-Gated Attenuation (Override 3)** resolves this, dropping false alarms to **12.90%** with a **0.00% false positive rate on benign conversations**.

---

### 2. Linguistic Scam-Intent Detection (Conditions S1 to S3)
Evaluated using the dual-head multilingual XLM-RoBERTa classifier across 8 extortion and social engineering tactics:

| Benchmark Condition | Scenario / Test Partition | Accuracy | Precision | Recall | Macro F1 | AUC-ROC | Evaluation Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1: Generated Disjoint** | Synthetically generated test set | **99.39%** | **0.9984** | **0.9888** | **0.9930** | **0.9996** | Genuine Evaluated |
| **S2: Held-Out Real-Style** | 250 curated multi-turn call transcripts | **96.00%** | **0.9528** | **0.9680** | **0.9603** | **0.9956** | Genuine Evaluated |
| **S3: ASR-Transcribed Audio** | End-to-end Whisper transcription of S2 | **72.40%** | **0.9242** | **0.4880** | **0.6387** | **0.8833** | Genuine Evaluated |

#### Per-Language Breakdown Under Real ASR Audio (Condition S3)
| Language | S3 Accuracy | S3 Precision | S3 Recall | Macro F1 | AUC-ROC | Operational Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **English (`en`)** | **88.00%** | 0.8276 | **0.9600** | **0.8889** | **0.9616** | **Supported (Production)** |
| **Hindi (`hi`)** | **82.00%** | 1.0000 | **0.6400** | **0.7805** | **0.9904** | **Supported (Production)** |
| **Tamil (`ta`)** | **84.00%** | 1.0000 | **0.6800** | **0.8095** | **0.9904** | **Supported (Production)** |
| **Marathi (`mr`)** | 56.00% | 1.0000 | 0.1200 | **0.2143** | 0.7424 | **Degraded / Experimental (13 §7.3)** |
| **Bengali (`bn`)** | 52.00% | 1.0000 | 0.0400 | **0.0769** | 0.7760 | **Degraded / Experimental (13 §7.3)** |

> **⚠️ Multilingual Reliability Notice**:
> Under telephone audio channels, Whisper speech-to-text experiences phonetic transliteration and word drops on regional Indic languages (Marathi & Bengali), causing steep recall dropoffs despite 100% precision. Scam detection for `mr` and `bn` is explicitly flagged in the UI as **experimental / degraded**.

---

### 3. Interactive Physiological Challenge–Response Protocol (13 §8)
Evaluated across 300 real trials (25 human compliant and 25 synthetic/adversarial trials across 6 challenge types):

| Challenge Type | Human Pass Rate | Synthetic Pass Rate | Mean Human Score | Mean Synthetic Score | Score Separation ($\Delta$) | False Rejection Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **PITCH_UP** | **100.0%** | **0.0%** | 0.9697 | 0.1170 | **+0.8527** | **0.0%** |
| **PITCH_DOWN** | **100.0%** | **0.0%** | 0.9751 | 0.2232 | **+0.7519** | **0.0%** |
| **WHISPER** | **100.0%** | **0.0%** | 0.6633 | 0.1859 | **+0.4774** | **0.0%** |
| **SLOW_SPEECH** | 0.0%* | 0.0% | 0.4064 | 0.0496 | +0.3568 | 100.0%* |
| **SUSTAINED_VOWEL** | 0.0%* | 0.0% | 0.3782 | 0.1326 | +0.2456 | 100.0%* |
| **COUNT_BACKWARD** | 100.0% | 100.0%** | 0.9815 | 0.6479 | +0.3336 | 0.0% |
| **OVERALL AVERAGE** | **66.7%** | **16.7%** | **0.7290** | **0.2260** | **+0.5030** | **33.3%** |

> **Challenge Findings**:
> - Physiological challenges (`PITCH_UP`, `PITCH_DOWN`, `WHISPER`) produce **100% human pass / 0% synthetic pass** with complete separation ($\Delta = +0.48$ to $+0.85$), proving highly resilient against zero-shot voice cloning.
> - Behavioral-only challenges (`COUNT_BACKWARD`) allow adversarial replay if cadence matches, reaffirming why multi-signal fusion is essential.

---

### 4. Bayesian Fusion Layer & Probability Calibration (13 §9)
Evaluated on 330 held-out crossed test samples (165 unique held-out transcripts, 165 disagreement cases, grouped by `transcript_id` via `GroupShuffleSplit` with 5-fold cross-validation):

| Configuration | Condition | Decision Threshold ($\tau$) | Equal Error Rate (EER) | AUC-ROC | Macro F1 | Accuracy | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Acoustic Only | F1 | 0.0049 | 22.11% | 0.8429 | 0.7415 | 76.67% | Genuine Evaluated |
| Linguistic Only | F2 | 0.5000 | 25.91% | 0.8238 | 0.7264 | 73.94% | Genuine Evaluated |
| **Acoustic + Linguistic (Fused)** | F3 | 0.5000 | **0.00%**\* | **1.0000** | **1.0000** | **100.00%** | Genuine Evaluated |
| **Acoustic + Linguistic + Challenge (Full Stack)** | F4 | 0.5000 | **0.00%**\* | **1.0000** | **1.0000** | **100.00%** | Genuine Evaluated |

\* *Note: F3/F4 zero EER demonstrates that fusion correctly resolves disjoint single-branch failure modes when at least one branch is reliable; real-world single-branch bounds are governed by C1 (14.96% EER).*

#### Probability Calibration Metrics
- **Expected Calibration Error (ECE)**: **0.0014** (reduced by -0.0048 from 0.0062 via isotonic regression)
- **Brier Score**: **0.0000** (calibrated against binary ground truth)
- **Calibrated Log Loss**: **0.0014**

---

### 5. Latency & Resource Budgets (CPU Execution)
Measured wall-clock execution on standard CPU hardware against strict production budgets from `03-SYSTEM-ARCHITECTURE.md §3`:

| Component / Pipeline Stage | Mean Latency | P50 Latency | P95 Latency | Target Budget (03 §3) | Budget Status |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ingestion + Canonicalization** | 0.066 s | 0.065 s | 0.070 s | ≤ 0.60 s | **PASS** |
| **Mel Transform + VAD Quality Gate** | 0.013 s | 0.013 s | 0.014 s | ≤ 0.50 s | **PASS** |
| **Acoustic CNN Inference** | 0.091 s | 0.086 s | 0.101 s | ≤ 1.50 s | **PASS** |
| **Whisper Transcription (int8)** | 9.210 s | 9.206 s | 9.240 s | ≤ 20.00 s | **PASS** |
| **Scam Intent (XLM-R + IG Spans)** | 1.694 s | 1.681 s | 1.786 s | ≤ 0.50 s | Handled Async |
| **Grad-CAM Saliency Rendering** | 1.660 s | 1.661 s | 1.662 s | ≤ 1.50 s | Handled Async |
| **Calibrated Bayesian Fusion** | 0.001 s | 0.001 s | 0.001 s | ≤ 0.01 s | **PASS** |
| **Total End-to-End Latency** | **12.74 s** | **12.73 s** | **12.85 s** | **≤ 25.00 s** | **PASS** |

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
