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


def test_wavefake_protocol_generation(tmp_path: Path):
    """Verify WaveFake protocol builder assigns disjoint chapters."""
    from prepare_datasets import build_wavefake_protocol
    import soundfile as sf
    import numpy as np

    fake_dir = tmp_path / "wavefake" / "ljspeech_melgan"
    real_dir = tmp_path / "wavefake" / "bona_fide"
    fake_dir.mkdir(parents=True)
    real_dir.mkdir(parents=True)

    dummy = np.zeros(16000, dtype=np.float32)
    # LJ001 -> train, LJ045 -> dev
    sf.write(str(real_dir / "LJ001-0001.wav"), dummy, 16000)
    sf.write(str(fake_dir / "LJ045-0002.wav"), dummy, 16000)

    proto_out = tmp_path / "proto.csv"
    build_wavefake_protocol(tmp_path / "wavefake", proto_out)
    assert proto_out.is_file()

    import csv
    with open(proto_out, "r") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    splits = {r["split"] for r in rows}
    assert "train" in splits
    assert "dev" in splits


def test_in_the_wild_protocol_generation(tmp_path: Path):
    """Verify In-the-Wild protocol builder assigns all files to eval."""
    from prepare_datasets import build_in_the_wild_protocol
    import soundfile as sf
    import numpy as np

    itw_dir = tmp_path / "itw"
    itw_dir.mkdir()
    dummy = np.zeros(16000, dtype=np.float32)
    sf.write(str(itw_dir / "sample1.wav"), dummy, 16000)

    proto_out = tmp_path / "itw_proto.csv"
    build_in_the_wild_protocol(itw_dir, proto_out)
    assert proto_out.is_file()

    import csv
    with open(proto_out, "r") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["split"] == "eval"

