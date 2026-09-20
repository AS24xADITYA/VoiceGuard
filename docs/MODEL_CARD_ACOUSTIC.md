# Model Card: Acoustic Deepfake CNN (`AcousticDeepfakeCNN`)

## Model Details
- **Architecture**: Transfer learning backbone based on `EfficientNet-B0` (with `ResNet-18` fallback) modified with a 3-channel input layer to process replicated single-channel log-mel spectrograms.
- **Input Dimensions**: `(Batch, 3, 128, 400)` representing 128 mel bins over a 4.0-second window (400 time frames @ 10ms hop).
- **Target Task**: Binary classification (`0: bona_fide human speech`, `1: spoof / synthetic speech`).
- **Grad-CAM Hook Layer**: `backbone.conv_head` (EfficientNet-B0) / `backbone.layer4[-1]` (ResNet-18 fallback).
- **Version**: `0.1.0-asvspoof2019-effb0`

## Intended Use
- **Primary Use**: Detecting acoustic synthesis fingerprints, neural vocoder phase jitter, and high-frequency spectral roll-off anomalies in pre-recorded audio snippets.
- **Out-of-Scope Use**:
  - Live cellular call intercept.
  - Legal attribution or individual speaker biometric identification.
  - Audio shorter than 1.5 seconds.

## Training Data & Procedure
- **Training Corpus**: ASVspoof 2019 Logical Access (LA) training set (25,380 utterances).
- **Augmentation**: SpecAugment (frequency masking up to 16 bins, time masking up to 40 frames) + additive background noise from MUSAN.
- **Optimization**: AdamW (`lr=3e-4`, weight decay `1e-4`), Cosine Annealing scheduler, mixed precision (AMP fp16).
- **Loss Function**: Weighted Cross-Entropy with label smoothing (`0.05`).
- **Training Script**: `backend/ai/acoustic/train.py` & `notebooks/02_train_acoustic.ipynb`.

## Performance & Empirical Evaluation (Conditions C1–C5)
Evaluated per `13-TESTING-AND-EVALUATION.md` §6 across 124,564 audio utterances:

- **Operating Decision Threshold ($\tau_{\text{acoustic}}$)**: **0.0049** (calibrated on in-domain validation split)
- **In-Domain EER (C1, ASVspoof 2019 LA Eval, 71,237 clips)**: **14.96%** (0.1496)
- **In-Domain AUC-ROC (C1)**: **0.8710**
- **In-Domain min t-DCF (C1)**: **0.3786**
- **In-Domain Macro F1 (C1)**: **0.9108**
- **Out-of-Domain EER (C2, In-the-Wild, 28,452 clips)**: **46.98%** (0.4698)
- **Out-of-Domain AUC-ROC (C2)**: **0.5141**
- **Codec-Degraded EER (C3, G.711/OPUS, 24,844 clips)**: **4.31%** (0.0431)
- **Noise-Degraded EER (C4, 10 dB SNR, 24,844 clips)**: **31.47%** (0.3147)
- **Consumer-Microphone Genuine Speech FPR (C5, 31 clips)**: **100.00%** (Mean score: **0.8500**, Median: **0.8500**, Range: `[0.8328, 0.8549]`)

### Cross-Corpus Generalization & Attack Sensitivity
- **C1 → C2 Generalization Gap**: +32.02% EER degradation on unseen modern generative architectures.
- **Neural Vocoders (A07–A12)**: Detection miss rate is < 0.02%, showing high sensitivity to vocoder phase artifacts.
- **Advanced Synthesis (A17/A18)**: High miss rates (A17: 17.46%, A18: 80.20%), where modern waveform-matching synthesizers bypass spectral anomaly detection.
- **Condition C5 Usability Finding**: The acoustic branch exhibits an acute domain shift on ordinary consumer hardware (laptop/phone mics with room reverb and ambient noise), outputting ~0.85 spoof probability on 100% of tested genuine human speech.

## Known Failure Modes
- **Consumer Microphone & Room Acoustics Collapse**: Normal consumer microphones and room acoustics create spectral distortions outside ASVspoof 2019 training distribution, triggering a 100% false positive rate at the 0.0049 threshold.
- Low-bitrate telephony codecs (e.g., AMR 4.75 kbps, G.711) introduce severe spectral cutoff above 3.5 kHz, which can degrade acoustic detector confidence.
- Zero-shot diffusion vocoders with continuous-time sampling generate clean harmonics that reduce detector confidence.
