"""Feature vector assembly for the fusion layer.

Per 05 §5.2:
  Exactly 14 features in fixed order.
  Missing branches handled with explicit availability flags and neutral imputation values,
  never zero-imputation of probabilities.

  0: acoustic_available {0, 1}
  1: acoustic_spoof_prob [0, 1]
  2: acoustic_uncertainty [0, 1]
  3: acoustic_window_std [0, 1]
  4: linguistic_available {0, 1}
  5: scam_prob [0, 1]
  6: scam_max_category [0, 1]
  7: scam_n_categories [0, 1]
  8: transcript_reliable {0, 1}
  9: language_supported {0, 1}
  10: transcript_length_norm [0, 1]
  11: challenge_available {0, 1}
  12: challenge_consistency [0, 1]
  13: audio_quality_score [0, 1]

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ai.audio.quality import compute_audio_quality_score
from ai.base import (
    AcousticResult,
    ChallengeResult,
    FusionFeatures,
    QualityReport,
    ScamIntentResult,
    TranscriptResult,
)

FEATURE_NAMES: Final[list[str]] = [
    "acoustic_available",
    "acoustic_spoof_prob",
    "acoustic_uncertainty",
    "acoustic_window_std",
    "linguistic_available",
    "scam_prob",
    "scam_max_category",
    "scam_n_categories",
    "transcript_reliable",
    "language_supported",
    "transcript_length_norm",
    "challenge_available",
    "challenge_consistency",
    "audio_quality_score",
]


def assemble_features(
    quality: QualityReport | None = None,
    acoustic: AcousticResult | None = None,
    transcript: TranscriptResult | None = None,
    scam: ScamIntentResult | None = None,
    challenge: ChallengeResult | None = None,
) -> FusionFeatures:
    """Assemble the exact 14-dimensional feature vector.

    Missing branches are assigned explicit availability flags and principled neutral values:
      - Unavailable acoustic: spoof_prob=0.5, uncertainty=1.0, window_std=0.0
      - Unavailable linguistic: scam_prob=0.5, max_cat=0.0, n_cat=0.0, word_count=0
      - Unavailable challenge: consistency=0.5
    """
    # ── 1. Acoustic branch ──
    if acoustic is not None:
        acoustic_avail = 1.0
        acoustic_prob = float(np.clip(acoustic.spoof_probability, 0.0, 1.0))
        acoustic_unc = float(np.clip(acoustic.uncertainty, 0.0, 1.0))
        acoustic_std = float(np.std(acoustic.window_scores)) if len(acoustic.window_scores) > 1 else 0.0
    else:
        acoustic_avail = 0.0
        acoustic_prob = 0.50  # neutral probability
        acoustic_unc = 1.00   # maximum uncertainty
        acoustic_std = 0.00

    # ── 2. Linguistic branch ──
    if transcript is not None and transcript.text.strip():
        trans_reliable = 1.0 if transcript.is_reliable else 0.0
        lang_supp = 1.0 if transcript.language_supported else 0.0
        text_norm = min(1.0, transcript.word_count / 100.0)
    else:
        trans_reliable = 0.0
        lang_supp = 0.0
        text_norm = 0.0

    if scam is not None:
        ling_avail = 1.0
        scam_p = float(np.clip(scam.scam_probability, 0.0, 1.0))
        max_cat = max(scam.category_scores.values()) if scam.category_scores else 0.0
        n_cat = len(scam.triggered_categories) / 8.0
    else:
        ling_avail = 0.0
        scam_p = 0.50  # neutral probability
        max_cat = 0.0
        n_cat = 0.0

    # ── 3. Challenge branch ──
    if challenge is not None:
        chal_avail = 1.0
        chal_cons = float(np.clip(challenge.consistency_score, 0.0, 1.0))
    else:
        chal_avail = 0.0
        chal_cons = 0.50  # neutral

    # ── 4. Quality gate score ──
    if quality is not None:
        q_score = compute_audio_quality_score(quality)
    else:
        q_score = 0.50

    return FusionFeatures(
        acoustic_available=acoustic_avail,
        acoustic_spoof_prob=round(acoustic_prob, 4),
        acoustic_uncertainty=round(acoustic_unc, 4),
        acoustic_window_std=round(acoustic_std, 4),
        linguistic_available=ling_avail,
        scam_prob=round(scam_p, 4),
        scam_max_category=round(max_cat, 4),
        scam_n_categories=round(n_cat, 4),
        transcript_reliable=trans_reliable,
        language_supported=lang_supp,
        transcript_length_norm=round(text_norm, 4),
        challenge_available=chal_avail,
        challenge_consistency=round(chal_cons, 4),
        audio_quality_score=round(q_score, 4),
    )
