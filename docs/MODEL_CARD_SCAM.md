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
- **Supported Languages**: English (`en`), Hindi (`hi`), Marathi (`mr`), Bengali (`bn`), Tamil (`ta`).
- **Out-of-Scope Use**:
  - General conversational sentiment analysis.
  - Legal determination of criminal fraud.

## Training Data & Procedure
- **Corpus**: Multi-turn synthetic and curated extortion dialogues across English, Hindi, and regional languages.
- **Loss Function**: Multi-task joint loss: `Loss = 0.6 * Loss_binary + 0.4 * Loss_category`.
- **Optimization**: AdamW (`lr=2e-5`, weight decay `0.01`), layer-freezing warm-up.
- **Training Script**: `backend/ai/linguistic/train_scam.py` & `notebooks/03_train_scam_intent.ipynb`.

## Performance & Evaluation Status
> **PENDING** — No fine-tuned model artifact exists yet. Model fine-tuning will be executed on Google Colab using `notebooks/03_train_scam_intent.ipynb`. Per `13-TESTING-AND-EVALUATION.md` §12, rule 1, no performance metric is recorded until produced by an empirical evaluation run.

- **Macro F1 (Overall Intent)**: PENDING
- **Held-Out Test Accuracy**: PENDING
- **Category F1 Breakdown**: PENDING

## Known Failure Modes
- Legitimate urgent messages (e.g., automated hospital billing reminders or banking fraud alerts) can trigger false positive urgency detections.
- ASR transcript errors (e.g., phonetically misspelled banking terms in non-English languages) can degrade category detection.
