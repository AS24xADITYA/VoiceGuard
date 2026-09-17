"""Unit tests for ai/challenge/catalog.py and verifier.py.

Verifies Phase 6 exit criteria:
  - Each challenge type issues a valid prompt with a random phrase
  - Comparative features extract correctly from synthetic audio
  - A genuine response to PITCH_UP scores >= 0.5; a baseline-identical response scores < 0.5
  - Non-compliant response returns NON_COMPLIANT with zero consistency score
"""

from __future__ import annotations

from pathlib import Path
import pytest

from ai.base import Challenge, ChallengeResult
from ai.challenge.catalog import CATALOG, issue_challenge
from ai.challenge.verifier import ChallengeVerifier, extract_acoustic_features
from tests.fixtures.audio_fixtures import create_synthetic_audio, write_wav_file


def test_issue_challenge_all_types():
    """Verify each challenge type issues a valid Challenge with prompt and instructions."""
    for c_type in CATALOG.keys():
        challenge = issue_challenge(c_type)
        assert isinstance(challenge, Challenge)
        assert challenge.challenge_type == c_type
        assert len(challenge.prompt_text) > 5
        assert len(challenge.instructions) >= 1
        assert challenge.id.startswith("chl_")


def test_extract_acoustic_features():
    """Verify feature extraction extracts required acoustic keys."""
    y = create_synthetic_audio(duration_s=3.0, sr=16000)
    features = extract_acoustic_features(y, sr=16000)

    for key in ["f0_median", "f0_std", "hnr", "spectral_flatness", "spectral_centroid", "rms_energy", "speaking_rate"]:
        assert key in features
        assert not isinstance(features[key], complex)


def test_pitch_up_response_scores_higher_than_identical(tmp_path: Path):
    """Verify genuine pitch rise passes PITCH_UP, while identical baseline fails."""
    # Baseline: 200 Hz tone
    base_wav = tmp_path / "base.wav"
    y_base = create_synthetic_audio(duration_s=3.0, sr=16000, frequencies=(200.0,))
    write_wav_file(base_wav, y_base)

    # Identical response: 200 Hz tone (no pitch change)
    same_wav = tmp_path / "same.wav"
    write_wav_file(same_wav, y_base)

    # Compliant pitch-up response: 280 Hz tone (+40% pitch rise)
    up_wav = tmp_path / "up.wav"
    y_up = create_synthetic_audio(duration_s=3.0, sr=16000, frequencies=(280.0,))
    write_wav_file(up_wav, y_up)

    verifier = ChallengeVerifier()
    challenge = issue_challenge("PITCH_UP")

    res_same = verifier.verify(base_wav, same_wav, challenge)
    res_up = verifier.verify(base_wav, up_wav, challenge)

    # Identical pitch did not pitch up -> scores below 0.5
    assert res_same.consistency_score < 0.50
    # Higher pitch -> scores above 0.5
    assert res_up.consistency_score >= 0.50


def test_non_compliant_phrase_returns_zero(tmp_path: Path):
    """Verify non-compliant transcript returns consistency_score = 0.0 and notes."""
    base_wav = tmp_path / "base.wav"
    y = create_synthetic_audio(3.0)
    write_wav_file(base_wav, y)

    verifier = ChallengeVerifier()
    challenge = Challenge(
        id="chl_test",
        challenge_type="WHISPER",
        prompt_text="Please whisper: 'Bright blue skies'",
        expected_phrase="Bright blue skies",
        expected_duration_s=5.0,
        instructions=[],
    )

    # Mismatched response transcript
    result = verifier.verify(
        base_wav,
        base_wav,
        challenge,
        response_transcript="Totally different unrelated response phrase",
    )

    assert not result.passed
    assert result.consistency_score == 0.0
    assert not result.compliance["matched"]
    assert any("NON_COMPLIANT" in note for note in result.notes)
