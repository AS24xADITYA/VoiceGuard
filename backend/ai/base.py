"""AI component protocol and shared dataclasses.

This module defines the Component protocol that every AI module implements,
and the shared result dataclasses used across the pipeline.

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable


class Verdict(str, Enum):
    """Categorical verdict output from the fusion layer."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    INCONCLUSIVE = "INCONCLUSIVE"


@runtime_checkable
class Component(Protocol):
    """Protocol for all AI components.

    Every AI module exposes this interface. The backend loads components
    through the ComponentRegistry and never calls model internals directly.
    """

    name: str
    version: str

    def load(self) -> None:
        """Load model artefacts from disk/network."""
        ...

    def is_loaded(self) -> bool:
        """Return True if the component is ready for inference."""
        ...

    def warmup(self) -> None:
        """Run a dummy inference to warm up the component."""
        ...


@dataclass(frozen=True)
class AudioMeta:
    """Metadata from audio ingestion."""

    canonical_path: Path
    original_path: Path
    sha256: str
    duration_seconds: float
    sample_rate: int
    n_samples: int
    file_size_bytes: int


@dataclass(frozen=True)
class QualityReport:
    """Audio quality gate report."""

    duration_s: float
    speech_ratio: float
    snr_estimate_db: float
    clipping_ratio: float
    dc_offset: float
    passed: bool
    failures: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AcousticResult:
    """Output from the acoustic deepfake detector."""

    spoof_probability: float  # [0,1], aggregated
    predicted_class: int  # 0=bona fide, 1=spoof
    window_scores: list[float]  # per-window spoof probability
    window_times: list[tuple[float, float]]
    aggregation: dict  # method, weights, k
    uncertainty: float  # predictive entropy, normalised [0,1]
    window_std: float
    is_borderline: bool
    n_windows_scored: int
    n_windows_discarded: int
    spectrogram_path: Path | None
    model_version: str
    inference_ms: int


@dataclass(frozen=True)
class Segment:
    """A transcript segment with timing."""

    start: float
    end: float
    text: str
    avg_logprob: float
    confidence: float = 0.0


@dataclass(frozen=True)
class TranscriptResult:
    """Output from the Whisper transcriber."""

    text: str
    language: str  # ISO 639-1
    language_probability: float
    language_supported: bool
    segments: list[Segment]
    mean_confidence: float  # [0,1]
    is_reliable: bool
    word_count: int
    no_speech_ratio: float
    hallucination_flags: list[str]
    model_version: str
    inference_ms: int


@dataclass(frozen=True)
class SalientSpan:
    """A span in the transcript salient for scam classification."""

    start: int  # character offset
    end: int  # character offset
    weight: float
    text: str = ""


@dataclass(frozen=True)
class ScamIntentResult:
    """Output from the scam-intent classifier."""

    scam_probability: float  # [0,1]
    category_scores: dict[str, float]  # tactic taxonomy
    triggered_categories: list[str]
    category_threshold: float
    salient_spans: list[SalientSpan]
    n_windows: int
    model_version: str
    inference_ms: int


@dataclass(frozen=True)
class Challenge:
    """A challenge issued to the user."""

    id: str
    challenge_type: str
    prompt_text: str
    expected_phrase: str | None
    expected_duration_s: float
    instructions: list[str]


@dataclass(frozen=True)
class ChallengeResult:
    """Output from the challenge verifier."""

    consistency_score: float  # [0,1], higher = more human-consistent
    feature_deltas: dict[str, float]
    expected_deltas: dict[str, tuple[float, float]]
    passed: bool
    compliance: dict  # phrase_matched, similarity
    notes: list[str]


@dataclass(frozen=True)
class FusionFeatures:
    """The 14-element feature vector consumed by the fusion layer."""

    acoustic_available: float  # {0, 1}
    acoustic_spoof_prob: float  # [0,1]
    acoustic_uncertainty: float  # [0,1]
    acoustic_window_std: float  # [0,1]
    linguistic_available: float  # {0, 1}
    scam_prob: float  # [0,1]
    scam_max_category: float  # [0,1]
    scam_n_categories: float  # [0,1]
    transcript_reliable: float  # {0, 1}
    language_supported: float  # {0, 1}
    transcript_length_norm: float  # [0,1]
    challenge_available: float  # {0, 1}
    challenge_consistency: float  # [0,1]
    audio_quality_score: float  # [0,1]

    def to_vector(self) -> list[float]:
        """Return the feature vector in the documented order."""
        return [
            self.acoustic_available,
            self.acoustic_spoof_prob,
            self.acoustic_uncertainty,
            self.acoustic_window_std,
            self.linguistic_available,
            self.scam_prob,
            self.scam_max_category,
            self.scam_n_categories,
            self.transcript_reliable,
            self.language_supported,
            self.transcript_length_norm,
            self.challenge_available,
            self.challenge_consistency,
            self.audio_quality_score,
        ]


@dataclass(frozen=True)
class FusionResult:
    """Output from the fusion layer."""

    risk_probability: float  # [0,1], calibrated
    verdict: Verdict
    confidence: float
    feature_vector: dict[str, float]
    contributions: list[dict]  # {feature, value, contribution, direction}
    reasons: list[str]  # human-readable
    overrides_applied: list[str]
    thresholds: dict[str, float]
    model_version: str


@dataclass(frozen=True)
class ExplanationResult:
    """Output from the Grad-CAM explainer."""

    method: str  # "grad-cam"
    target_layer: str
    target_class: int
    windows: list[dict]  # per-window CAM metadata
    axis_extents: dict[str, float]
    disclaimer: str


@dataclass(frozen=True)
class PipelineResult:
    """Assembled result from the full pipeline."""

    audio_meta: AudioMeta
    quality: QualityReport
    acoustic: AcousticResult | None
    transcript: TranscriptResult | None
    scam: ScamIntentResult | None
    challenge: ChallengeResult | None
    fusion: FusionResult | None
    explanation: ExplanationResult | None
    stage_timings_ms: dict[str, int]
    model_versions: dict[str, str]
    degraded_branches: list[str]
