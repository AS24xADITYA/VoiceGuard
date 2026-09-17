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

## Performance & Evaluation Status
> **PENDING** — No trained model artifact exists yet. Model training will be executed on Google Colab using `notebooks/02_train_acoustic.ipynb`. Per `13-TESTING-AND-EVALUATION.md` §12, rule 1, no performance metric is recorded until produced by an empirical evaluation run.

- **In-Domain EER (ASVspoof 2019 Eval)**: PENDING
- **In-Domain AUC-ROC**: PENDING
- **In-Domain min t-DCF**: PENDING
- **Out-of-Domain EER (In-the-Wild)**: PENDING
- **Out-of-Domain AUC-ROC**: PENDING

## Known Failure Modes
- Low-bitrate telephony codecs (e.g., AMR 4.75 kbps, G.711) introduce severe spectral cutoff above 3.5 kHz, which can degrade acoustic detector confidence.
- Zero-shot diffusion vocoders with continuous-time sampling generate clean harmonics that reduce detector confidence.
