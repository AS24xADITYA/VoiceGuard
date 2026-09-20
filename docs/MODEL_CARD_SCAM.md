# Model Card: Multilingual Scam-Intent Classifier (`ScamIntentModel`)

## Model Details
- **Architecture**: `XLM-RoBERTa-base` with dual-head multi-task architecture:
  - `binary_head`: Linear projection (`768 -> 2`) for overall scam vs benign intent.
  - `category_head`: Linear projection (`768 -> 8`) with sigmoid activations for multi-label extortion tactic classification.
- **Input**: Tokenized text sequences (up to 256 tokens) with sliding-window aggregation for long dialogues.
- **Attribution Method**: Integrated Gradients attribution mapped to character token spans.
- **Version**: `0.1.0-xlmr-scam-intent`

## Intended Use
- **Primary Use**: Analyzing transcripts produced by Whisper to identify coercion, impersonation of authority, artificial urgency, and financial extraction tactics.
- **Supported Languages**:
  - **Production Tier**: English (`en`), Hindi (`hi`), Tamil (`ta`).
  - **Experimental / Degraded Tier**: Marathi (`mr`), Bengali (`bn`) per `13 §7.3`.
- **Out-of-Scope Use**:
  - General conversational sentiment analysis.
  - Legal determination of criminal fraud.

## Training Data & Procedure
- **Corpus**: Multi-turn synthetic and curated extortion dialogues combined with Enron spam and SMS spam, balanced across English, Hindi, and regional Indic languages.
- **Loss Function**: Multi-task joint loss: `Loss = 0.6 * Loss_binary + 0.4 * Loss_category`.
- **Optimization**: AdamW (`lr=2e-5`, weight decay `0.01`), layer-freezing warm-up.
- **Training Script**: `backend/ai/linguistic/train_scam.py` & `notebooks/03_train_scam_intent.ipynb`.

## Performance & Evaluation Status

Empirically evaluated against Conditions S1 (synthetic generated disjoint), S2 (curated real-style human transcripts), and S3 (end-to-end ASR transcribed speech audio):

| Condition | Accuracy | Macro F1 | AUC-ROC | Precision | Recall | Evaluation Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **S1 (Held-out generated)** | 0.9939 | 0.9930 | 0.9996 | 0.9984 | 0.9888 | Genuine Evaluated |
| **S2 (Held-out real-style)** | 0.9600 | 0.9603 | 0.9956 | 0.9528 | 0.9680 | Genuine Evaluated |
| **S3 (ASR transcribed audio)** | 0.7240 | 0.6387 | 0.8833 | 0.9242 | 0.4880 | Genuine Evaluated |

### Per-Language Breakdown Under Condition S3 (End-to-End ASR)
- **English (`en`)**: Acc = 0.8800, F1 = 0.8889, AUC = 0.9616 [Production]
- **Hindi (`hi`)**: Acc = 0.8200, F1 = 0.7805, AUC = 0.9904 [Production]
- **Tamil (`ta`)**: Acc = 0.8400, F1 = 0.8095, AUC = 0.9904 [Production]
- **Marathi (`mr`)**: Acc = 0.5600, F1 = 0.2143, AUC = 0.7424 [**Experimental / Degraded**]
- **Bengali (`bn`)**: Acc = 0.5200, F1 = 0.0769, AUC = 0.7760 [**Experimental / Degraded**]

## Known Failure Modes
- **Bengali & Marathi ASR Transcription Breakdown (per 13 §7.3)**:
  Under real speech-to-text processing using Whisper, Bengali (F1 0.0769) and Marathi (F1 0.2143) experience heavy phonetic transliteration into Latin characters and word drops. This causes severe recall suppression for scam tactics in these two languages. They must be treated as experimental and monitored.
- **Ambiguous Written Credential Inquiries**:
  Passive credential requests (e.g. newsletter verification links) without urgent coercion often score below the strict 0.50 decision threshold, requiring explicit verbal demand cues.
- **Legitimate Urgent Alerts**:
  Urgent hospital or emergency notifications may occasionally trigger false-positive urgency tactic warnings if language mirrors high-pressure extortion tactics.
