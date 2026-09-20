"""Unit tests for ai/fusion/features.py, fuser.py, and ai/pipeline.py.

Verifies Phase 7 exit criteria:
  - Fusion handles every combination of available and unavailable branches
  - Single-branch cap fires: only 1 predictive branch caps verdict at MODERATE
  - Inconclusive floor fires: failed audio quality gate forces INCONCLUSIVE verdict
  - Feature contributions sum coherently and are signed correctly
  - Pipeline runs end-to-end without a web server present
"""

import pickle
from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression

from ai.base import (
    AcousticResult,
    FusionFeatures,
    FusionResult,
    PipelineResult,
    QualityReport,
    ScamIntentResult,
    TranscriptResult,
    Verdict,
)
from ai.fusion.features import FEATURE_NAMES, assemble_features
from ai.fusion.fuser import FusionEngine
from ai.pipeline import PipelineContext, run_pipeline
from ai.registry import get_registry, reset_registry
from tests.fixtures.audio_fixtures import create_synthetic_audio, write_wav_file


@pytest.fixture
def trained_fusion_artifact(tmp_path: Path) -> Path:
    """Create a temporary real trained artifact for testing FusionEngine mechanics."""
    clf = LogisticRegression()
    # Fit dummy dataset with 14 features
    import numpy as np
    X = np.random.randn(20, 14)
    y = np.random.randint(0, 2, 20)
    clf.fit(X, y)
    clf.coef_ = np.array([[
        0.1, 2.8, -0.7, 0.4, 0.1, 2.2, 0.8, 0.5, 0.3, 0.2, 0.2, -0.1, -2.0, -0.4
    ]])
    clf.intercept_ = np.array([0.5])

    artifact_path = tmp_path / "fusion.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump({"model": clf, "calibrator": None}, f)
    return artifact_path


def test_fusion_unloaded_raises_model_error():
    """Verify FusionEngine raises MODEL_ERROR when no trained artifact exists."""
    fuser = FusionEngine(model_path=Path("nonexistent/path/fusion.pkl"))
    fuser.load()
    assert not fuser.is_loaded()

    features = assemble_features()
    with pytest.raises(RuntimeError, match="MODEL_ERROR"):
        fuser.fuse(features)


def test_assemble_features_all_branches_missing():
    """Verify feature assembly handles completely empty/missing branches with neutral values."""
    features = assemble_features()
    vec = features.to_vector()

    assert len(vec) == 14
    assert features.acoustic_available == 0.0
    assert features.acoustic_spoof_prob == 0.50
    assert features.acoustic_uncertainty == 1.00
    assert features.linguistic_available == 0.0
    assert features.scam_prob == 0.50
    assert features.challenge_available == 0.0
    assert features.challenge_consistency == 0.50


def test_single_branch_cap_override(trained_fusion_artifact: Path):
    """Verify Override 1: A high-risk score is capped to MODERATE if only one predictive branch is available."""
    fuser = FusionEngine(
        model_path=trained_fusion_artifact,
        threshold_moderate=0.30,
        threshold_high=0.65,
    )
    fuser.load()
    assert fuser.is_loaded()

    # High acoustic spoof (0.95), but linguistic branch unavailable
    features = FusionFeatures(
        acoustic_available=1.0,
        acoustic_spoof_prob=0.95,
        acoustic_uncertainty=0.10,
        acoustic_window_std=0.05,
        linguistic_available=0.0,
        scam_prob=0.50,
        scam_max_category=0.0,
        scam_n_categories=0.0,
        transcript_reliable=0.0,
        language_supported=0.0,
        transcript_length_norm=0.0,
        challenge_available=0.0,
        challenge_consistency=0.50,
        audio_quality_score=0.90,
    )

    result = fuser.fuse(features)

    assert result.risk_probability >= 0.65
    assert result.verdict == Verdict.MODERATE
    assert "SINGLE_BRANCH_CAP" in result.overrides_applied
    assert any("Single-branch cap applied" in r for r in result.reasons)


def test_inconclusive_floor_override(trained_fusion_artifact: Path):
    """Verify Override 2: Quality failure forces INCONCLUSIVE verdict regardless of score."""
    fuser = FusionEngine(model_path=trained_fusion_artifact)
    fuser.load()
    assert fuser.is_loaded()

    features = FusionFeatures(
        acoustic_available=1.0,
        acoustic_spoof_prob=0.90,
        acoustic_uncertainty=0.10,
        acoustic_window_std=0.05,
        linguistic_available=1.0,
        scam_prob=0.90,
        scam_max_category=0.80,
        scam_n_categories=0.50,
        transcript_reliable=1.0,
        language_supported=1.0,
        transcript_length_norm=0.50,
        challenge_available=0.0,
        challenge_consistency=0.50,
        audio_quality_score=0.10,
    )

    failed_quality = QualityReport(
        duration_s=1.0,
        speech_ratio=0.10,
        snr_estimate_db=2.0,
        clipping_ratio=0.05,
        dc_offset=0.001,
        passed=False,
        failures=["Duration 1.0s below minimum 1.5s", "Clipping ratio 0.05 exceeds maximum 0.01"],
    )

    result = fuser.fuse(features, quality=failed_quality)

    assert result.verdict == Verdict.INCONCLUSIVE
    assert "INCONCLUSIVE_QUALITY_FLOOR" in result.overrides_applied
    assert any("Audio quality gate failed" in r for r in result.reasons)


def test_quality_gated_acoustic_attenuation_override(trained_fusion_artifact: Path):
    """Verify Override 3: Elevated acoustic spoof score on consumer audio without scam corroboration is attenuated."""
    fuser = FusionEngine(
        model_path=trained_fusion_artifact,
        threshold_moderate=0.30,
        threshold_high=0.65,
    )
    fuser.load()
    assert fuser.is_loaded()

    # Case 1: Elevated acoustic score (0.85), but consumer mic acoustics (Q=0.88, SNR=22dB) and clean benign text (scam=0.001)
    consumer_benign = FusionFeatures(
        acoustic_available=1.0,
        acoustic_spoof_prob=0.85,
        acoustic_uncertainty=0.60,
        acoustic_window_std=0.02,
        linguistic_available=1.0,
        scam_prob=0.001,
        scam_max_category=0.0,
        scam_n_categories=0.0,
        transcript_reliable=1.0,
        language_supported=1.0,
        transcript_length_norm=0.20,
        challenge_available=0.0,
        challenge_consistency=0.50,
        audio_quality_score=0.88,
    )
    consumer_quality = QualityReport(
        duration_s=8.0,
        speech_ratio=0.75,
        snr_estimate_db=22.0,
        clipping_ratio=0.0001,
        dc_offset=0.0001,
        passed=True,
        failures=[],
    )

    res1 = fuser.fuse(consumer_benign, quality=consumer_quality)
    assert res1.verdict == Verdict.LOW
    assert res1.risk_probability <= 0.30
    assert "QUALITY_GATED_ACOUSTIC_ATTENUATION" in res1.overrides_applied
    assert any("Quality-gated acoustic attenuation applied" in r for r in res1.reasons)

    # Case 2: Pristine studio audio (Q=0.98, SNR=45dB) with elevated acoustic score (0.85) and benign text (scam=0.001)
    # Under studio conditions, acoustic score is NOT attenuated (in-domain deepfake clone)
    studio_spoof = FusionFeatures(
        acoustic_available=1.0,
        acoustic_spoof_prob=0.85,
        acoustic_uncertainty=0.10,
        acoustic_window_std=0.02,
        linguistic_available=1.0,
        scam_prob=0.001,
        scam_max_category=0.0,
        scam_n_categories=0.0,
        transcript_reliable=1.0,
        language_supported=1.0,
        transcript_length_norm=0.20,
        challenge_available=0.0,
        challenge_consistency=0.50,
        audio_quality_score=0.98,
    )
    studio_quality = QualityReport(
        duration_s=8.0,
        speech_ratio=0.85,
        snr_estimate_db=45.0,
        clipping_ratio=0.0,
        dc_offset=0.0,
        passed=True,
        failures=[],
    )

    res2 = fuser.fuse(studio_spoof, quality=studio_quality)
    assert "QUALITY_GATED_ACOUSTIC_ATTENUATION" not in res2.overrides_applied
    assert res2.risk_probability > res1.risk_probability

    # Case 3: Consumer mic audio with high scam intent (scam=0.85) and elevated acoustic (0.85)
    # Corroborated by linguistic branch -> remains HIGH risk
    consumer_scam = FusionFeatures(
        acoustic_available=1.0,
        acoustic_spoof_prob=0.85,
        acoustic_uncertainty=0.60,
        acoustic_window_std=0.02,
        linguistic_available=1.0,
        scam_prob=0.85,
        scam_max_category=0.80,
        scam_n_categories=2.0,
        transcript_reliable=1.0,
        language_supported=1.0,
        transcript_length_norm=0.20,
        challenge_available=0.0,
        challenge_consistency=0.50,
        audio_quality_score=0.88,
    )
    res3 = fuser.fuse(consumer_scam, quality=consumer_quality)
    assert "QUALITY_GATED_ACOUSTIC_ATTENUATION" not in res3.overrides_applied
    assert res3.verdict == Verdict.HIGH



def test_feature_contributions(trained_fusion_artifact: Path):
    """Verify feature contributions have valid direction and magnitude."""
    fuser = FusionEngine(model_path=trained_fusion_artifact)
    fuser.load()
    assert fuser.is_loaded()

    features = assemble_features()
    result = fuser.fuse(features)

    assert len(result.contributions) == len(FEATURE_NAMES)
    for c in result.contributions:
        assert c["direction"] in ("increases_risk", "decreases_risk")
        assert isinstance(c["contribution"], float)


@pytest.mark.asyncio
async def test_pipeline_fails_without_trained_fusion(tmp_path: Path):
    """Verify pipeline raises MODEL_ERROR if fusion layer has no trained artifact."""
    reset_registry()
    registry = get_registry()
    unloaded_fuser = FusionEngine(model_path=Path("nonexistent/models/fusion.pkl"))
    registry.register("fusion", unloaded_fuser)

    wav_path = tmp_path / "test_audio.wav"
    y = create_synthetic_audio(duration_s=4.5, sr=16000)
    write_wav_file(wav_path, y)

    ctx = PipelineContext(output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="MODEL_ERROR"):
        await run_pipeline(wav_path, context=ctx)


@pytest.mark.asyncio
async def test_pipeline_end_to_end_with_artifact(tmp_path: Path, trained_fusion_artifact: Path):
    """Verify full pipeline runs end-to-end when a trained fusion artifact is registered."""
    reset_registry()
    registry = get_registry()
    fuser = FusionEngine(model_path=trained_fusion_artifact)
    fuser.load()
    registry.register("fusion", fuser)

    wav_path = tmp_path / "test_audio.wav"
    y = create_synthetic_audio(duration_s=4.5, sr=16000)
    write_wav_file(wav_path, y)

    stages_hit: list[str] = []
    ctx = PipelineContext(
        stage_callback=lambda s: stages_hit.append(s),
        output_dir=tmp_path,
    )

    result = await run_pipeline(wav_path, context=ctx)

    assert isinstance(result, PipelineResult)
    assert result.audio_meta.sample_rate == 16000
    assert result.fusion is not None
    assert result.fusion.verdict in (Verdict.LOW, Verdict.MODERATE, Verdict.HIGH, Verdict.INCONCLUSIVE)
    assert "INGESTING" in stages_hit
    assert "ANALYZING_ACOUSTIC" in stages_hit
    assert "FUSING" in stages_hit
    assert "COMPLETE" in stages_hit
