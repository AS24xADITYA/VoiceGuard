"""Unit tests for ai/linguistic/transcriber.py and scam_classifier.py.

Verifies Phase 5 exit criteria:
  - Hallucination guards suppress output on pure silence
  - 5-gram repetition detection truncates infinite repetition loops
  - ScamIntentModel architecture outputs (B, 2) binary logits and (B, 8) category logits
  - 8 Tactic categories match specification exactly
  - Salient spans map to valid character offsets in the transcript
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from ai.base import TranscriptResult
from ai.linguistic.scam_classifier import (
    TACTIC_CATEGORIES,
    ScamIntentClassifier,
    ScamIntentModel,
)
from ai.linguistic.transcriber import Transcriber, check_5gram_repetition
from tests.fixtures.audio_fixtures import create_silence_audio, write_wav_file


def test_5gram_repetition_truncation():
    """Verify 5-gram loop detection truncates repetitive text."""
    repeated_phrase = "please verify your account immediately today "
    text = "Hello sir, " + (repeated_phrase * 5) + " thank you."

    cleaned, was_truncated = check_5gram_repetition(text, max_repeats=3)
    assert was_truncated
    assert len(cleaned) < len(text)


def test_transcriber_silence_suppression(tmp_path: Path):
    """Verify transcriber suppresses output on pure silence (hallucination guard)."""
    silence_wav = tmp_path / "silence.wav"
    y = create_silence_audio(duration_s=4.0, sr=16000)
    write_wav_file(silence_wav, y)

    transcriber = Transcriber()
    result = transcriber.transcribe(silence_wav)

    assert isinstance(result, TranscriptResult)
    assert result.text == ""
    assert not result.is_reliable
    assert "SILENCE_SUPPRESSED" in result.hallucination_flags


def test_scam_model_architecture():
    """Verify dual-head model output shapes: binary (B, 2), categories (B, 8)."""
    model = ScamIntentModel(n_categories=len(TACTIC_CATEGORIES), pretrained=False)
    model.eval()

    input_ids = torch.randint(0, 1000, (2, 32))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        b_logits, c_logits = model(input_ids=input_ids, attention_mask=attention_mask)

    assert b_logits.shape == (2, 2)
    assert c_logits.shape == (2, len(TACTIC_CATEGORIES))
    assert len(TACTIC_CATEGORIES) == 8


def test_scam_classifier_scoring_and_spans():
    """Verify ScamIntentClassifier returns scores and valid character offset spans."""
    classifier = ScamIntentClassifier(category_threshold=0.40)
    classifier.load()

    text = "We are calling from your bank security department. Your account will be frozen unless you provide the OTP immediately."
    result = classifier.score(text)

    assert 0.0 <= result.scam_probability <= 1.0
    assert len(result.category_scores) == 8
    for cat in TACTIC_CATEGORIES:
        assert cat in result.category_scores

    # Verify salient span character boundaries
    for span in result.salient_spans:
        assert 0 <= span.start < span.end <= len(text)
        assert span.text == text[span.start : span.end]
