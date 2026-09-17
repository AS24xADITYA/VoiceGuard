"""Unit tests for scripts/prepare_datasets.py.

Verifies Phase 3 exit criteria:
  - Speaker-disjointness assertion passes when partitions have disjoint speakers
  - Speaker-disjointness assertion aborts and raises ValueError on any speaker leakage
"""

from __future__ import annotations

import pytest
import sys
from pathlib import Path

# Add scripts directory
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_datasets import verify_speaker_disjointness


def test_speaker_disjointness_clean_split():
    """Verify clean speaker-disjoint splits pass assertion."""
    manifest = [
        {"split": "train", "speaker_id": "SPK_01"},
        {"split": "train", "speaker_id": "SPK_02"},
        {"split": "dev", "speaker_id": "SPK_03"},
        {"split": "dev", "speaker_id": "SPK_04"},
        {"split": "eval", "speaker_id": "SPK_05"},
    ]
    # Should not raise
    verify_speaker_disjointness(manifest)


def test_speaker_disjointness_leakage_aborts():
    """Verify speaker overlap between train and dev/eval raises ValueError."""
    manifest_with_leakage = [
        {"split": "train", "speaker_id": "SPK_01"},
        {"split": "train", "speaker_id": "SPK_02"},
        {"split": "dev", "speaker_id": "SPK_02"},  # Leakage!
        {"split": "eval", "speaker_id": "SPK_03"},
    ]
    with pytest.raises(ValueError) as exc_info:
        verify_speaker_disjointness(manifest_with_leakage)

    assert "FATAL: Speaker disjointness violation detected" in str(exc_info.value)
    assert "SPK_02" in str(exc_info.value)
