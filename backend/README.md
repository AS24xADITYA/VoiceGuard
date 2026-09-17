# VoiceGuard Backend

FastAPI application and AI inference engine for the VoiceGuard Multi-Signal Deepfake Voice & Scam Intelligence platform.

## Modules
- `app`: FastAPI web application, database models, schemas, and API routes.
- `ai`: Core AI pipeline including acoustic detection (CNN + Grad-CAM), linguistic scam classification (Whisper + XLM-R), challenge-response verification, and calibrated fusion.

## Installation
```bash
pip install -e .
```
