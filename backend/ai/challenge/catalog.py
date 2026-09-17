"""Challenge catalogue, randomized phrase pools, and feature expectations.

Per 05 §4.2:
  Six challenge types:
    - WHISPER: Expects collapse of harmonic energy, HNR drop, spectral flatness rise
    - PITCH_UP: Expects median F0 rise >= 20%
    - PITCH_DOWN: Expects median F0 fall >= 15%
    - SLOW_SPEECH: Expects speaking rate drop >= 35%
    - SUSTAINED_VOWEL: Expects stable F0 with human micro-jitter and shimmer
    - COUNT_BACKWARD: Tests live interactive agent and semantic compliance

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Final

from ai.base import Challenge

# Randomized phrase pool for interactive repetition challenges
PHRASE_POOL: Final[list[str]] = [
    "The silver fox jumped quickly over the fence",
    "Bright blue skies brought warm gentle rain",
    "Nine quick zebras walked near the quiet river",
    "Four happy children played outside in the garden",
    "Cold autumn winds blew orange leaves across the road",
    "Seven yellow boats sailed across the calm bay",
    "Early morning sunlight broke through the thick fog",
    "Five energetic puppies ran through the tall grass",
]

# Baseline expected ranges (provisional empirical ranges per 05 §4.3 & 06 §5)
DEFAULT_EXPECTED_RANGES: Final[dict[str, dict[str, tuple[float, float]]]] = {
    "WHISPER": {
        "hnr": (-0.95, -0.30),  # HNR drops significantly
        "spectral_flatness": (0.30, 2.50),  # Flatness increases
        "rms_energy": (-0.80, -0.10),
    },
    "PITCH_UP": {
        "f0_median": (0.20, 0.80),  # Median F0 increases >= 20%
        "spectral_centroid": (0.10, 0.60),
    },
    "PITCH_DOWN": {
        "f0_median": (-0.60, -0.15),  # Median F0 falls >= 15%
        "spectral_centroid": (-0.50, -0.05),
    },
    "SLOW_SPEECH": {
        "speaking_rate": (-0.80, -0.35),  # Speaking rate drops >= 35%
    },
    "SUSTAINED_VOWEL": {
        "f0_std": (-0.80, -0.20),  # F0 is steady
        "jitter": (0.005, 0.05),   # Natural micro-jitter in human range
        "shimmer": (0.01, 0.10),   # Natural shimmer in human range
    },
    "COUNT_BACKWARD": {
        "speaking_rate": (-0.50, 0.20),
    },
}

FEATURE_WEIGHTS: Final[dict[str, dict[str, float]]] = {
    "WHISPER": {"hnr": 0.50, "spectral_flatness": 0.35, "rms_energy": 0.15},
    "PITCH_UP": {"f0_median": 0.70, "spectral_centroid": 0.30},
    "PITCH_DOWN": {"f0_median": 0.70, "spectral_centroid": 0.30},
    "SLOW_SPEECH": {"speaking_rate": 1.0},
    "SUSTAINED_VOWEL": {"f0_std": 0.50, "jitter": 0.25, "shimmer": 0.25},
    "COUNT_BACKWARD": {"speaking_rate": 1.0},
}


@dataclass(frozen=True)
class ChallengeDefinition:
    challenge_type: str
    prompt_template: str
    expected_duration_s: float
    requires_phrase: bool
    relevant_features: list[str]
    expected_range: dict[str, tuple[float, float]]
    feature_weights: dict[str, float]
    instructions: list[str]


CATALOG: Final[dict[str, ChallengeDefinition]] = {
    "WHISPER": ChallengeDefinition(
        challenge_type="WHISPER",
        prompt_template="Please repeat the following phrase in a whisper: '{phrase}'",
        expected_duration_s=5.0,
        requires_phrase=True,
        relevant_features=["hnr", "spectral_flatness", "rms_energy"],
        expected_range=DEFAULT_EXPECTED_RANGES["WHISPER"],
        feature_weights=FEATURE_WEIGHTS["WHISPER"],
        instructions=[
            "Speak in a natural whisper without vocal cord phonation.",
            "Repeat the exact phrase clearly.",
        ],
    ),
    "PITCH_UP": ChallengeDefinition(
        challenge_type="PITCH_UP",
        prompt_template="Please say the following phrase in a noticeably higher pitch: '{phrase}'",
        expected_duration_s=5.0,
        requires_phrase=True,
        relevant_features=["f0_median", "spectral_centroid"],
        expected_range=DEFAULT_EXPECTED_RANGES["PITCH_UP"],
        feature_weights=FEATURE_WEIGHTS["PITCH_UP"],
        instructions=[
            "Raise your pitch higher than your natural speaking voice.",
            "Speak clearly at normal volume.",
        ],
    ),
    "PITCH_DOWN": ChallengeDefinition(
        challenge_type="PITCH_DOWN",
        prompt_template="Please say the following phrase in a noticeably lower pitch: '{phrase}'",
        expected_duration_s=5.0,
        requires_phrase=True,
        relevant_features=["f0_median", "spectral_centroid"],
        expected_range=DEFAULT_EXPECTED_RANGES["PITCH_DOWN"],
        feature_weights=FEATURE_WEIGHTS["PITCH_DOWN"],
        instructions=[
            "Drop your pitch to a deeper, lower register.",
            "Speak clearly at normal volume.",
        ],
    ),
    "SLOW_SPEECH": ChallengeDefinition(
        challenge_type="SLOW_SPEECH",
        prompt_template="Please say the following phrase very slowly: '{phrase}'",
        expected_duration_s=7.0,
        requires_phrase=True,
        relevant_features=["speaking_rate"],
        expected_range=DEFAULT_EXPECTED_RANGES["SLOW_SPEECH"],
        feature_weights=FEATURE_WEIGHTS["SLOW_SPEECH"],
        instructions=[
            "Elongate each word and pause between words.",
            "Take the full 7 seconds to complete the phrase.",
        ],
    ),
    "SUSTAINED_VOWEL": ChallengeDefinition(
        challenge_type="SUSTAINED_VOWEL",
        prompt_template="Please hold the steady vowel sound 'aaaa' continuously for three seconds.",
        expected_duration_s=5.0,
        requires_phrase=False,
        relevant_features=["f0_std", "jitter", "shimmer"],
        expected_range=DEFAULT_EXPECTED_RANGES["SUSTAINED_VOWEL"],
        feature_weights=FEATURE_WEIGHTS["SUSTAINED_VOWEL"],
        instructions=[
            "Hold a continuous 'aaah' at a steady comfortable pitch.",
            "Do not vary your pitch or stop breathing midway.",
        ],
    ),
    "COUNT_BACKWARD": ChallengeDefinition(
        challenge_type="COUNT_BACKWARD",
        prompt_template="Please count backwards out loud from seven to one: 7, 6, 5, 4, 3, 2, 1.",
        expected_duration_s=6.0,
        requires_phrase=False,
        relevant_features=["speaking_rate"],
        expected_range=DEFAULT_EXPECTED_RANGES["COUNT_BACKWARD"],
        feature_weights=FEATURE_WEIGHTS["COUNT_BACKWARD"],
        instructions=[
            "Count down clearly: seven, six, five, four, three, two, one.",
            "Maintain a steady natural cadence.",
        ],
    ),
}


def issue_challenge(challenge_type: str | None = None) -> Challenge:
    """Issue a randomized challenge with a unique ID and prompt."""
    if challenge_type is None or challenge_type not in CATALOG:
        challenge_type = random.choice(list(CATALOG.keys()))

    defn = CATALOG[challenge_type]
    phrase = random.choice(PHRASE_POOL) if defn.requires_phrase else None

    prompt_text = (
        defn.prompt_template.format(phrase=phrase)
        if phrase
        else defn.prompt_template
    )

    challenge_id = f"chl_{uuid.uuid4().hex[:12]}"

    return Challenge(
        id=challenge_id,
        challenge_type=defn.challenge_type,
        prompt_text=prompt_text,
        expected_phrase=phrase,
        expected_duration_s=defn.expected_duration_s,
        instructions=defn.instructions,
    )
