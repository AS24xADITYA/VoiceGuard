"""Linguistic branch: transcription engine using faster-whisper with hallucination guards.

Per 05 §3.1:
  - Runtime: faster-whisper (CTranslate2) with OpenAI weights
  - VAD filter enabled (min_silence_duration_ms=500)
  - condition_on_previous_text=False (prevents repetition loops)
  - Multi-tier temperature fallback ladder [0.0, 0.2, 0.4]
  - Language identification: supported set {en, hi, mr, bn, ta}
  - 4 Hallucination guards:
      1. VAD filter
      2. condition_on_previous_text=False
      3. 5-gram repetition detection and truncation
      4. Compression ratio filtering (> 2.4 dropped)
  - Pure silence suppression

IMPORTANT: This module must NOT import from app/, db/, or any web framework.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Final

import numpy as np
import structlog

from ai.audio.io import load_canonical
from ai.audio.quality import compute_speech_ratio
from ai.base import Component, Segment, TranscriptResult

log = structlog.get_logger()

SUPPORTED_LANGUAGES: Final[set[str]] = {"en", "hi", "mr", "bn", "ta"}


def check_5gram_repetition(text: str, max_repeats: int = 3) -> tuple[str, bool]:
    """Check for repetitive hallucination loops.

    If any 5-gram repeats more than max_repeats times, truncate at the first repetition.
    Returns (cleaned_text, was_truncated).
    """
    words = text.strip().split()
    if len(words) < 10:
        return text, False

    seen_5grams: Counter[tuple[str, ...]] = Counter()
    first_repeat_index = -1

    for i in range(len(words) - 4):
        gram = tuple(words[i : i + 5])
        seen_5grams[gram] += 1
        if seen_5grams[gram] > max_repeats and first_repeat_index == -1:
            first_repeat_index = i

    if first_repeat_index != -1:
        truncated_words = words[:first_repeat_index]
        return " ".join(truncated_words), True

    return text, False


class Transcriber(Component):
    """Whisper-based multilingual transcriber with anti-hallucination guards."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        version: str = "whisper-small-v0.1.0",
    ) -> None:
        self._name: Final[str] = "whisper"
        self._version = version
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Any = None
        self._is_loaded = False
        self._load_ms = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    def load(self) -> None:
        """Load faster-whisper model. Raises RuntimeError if loading fails."""
        start = time.perf_counter()
        try:
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            self._is_loaded = True
            self._load_ms = int((time.perf_counter() - start) * 1000)
        except Exception as e:
            self.model = None
            self._is_loaded = False
            log.warning("whisper_load_failed", error=str(e))
            raise RuntimeError(f"Failed to load faster-whisper model '{self.model_size}': {e}") from e

    def warmup(self) -> None:
        """Warm up transcriber with a short synthetic sample if loaded."""
        if not self._is_loaded or self.model is None:
            self.load()

    def transcribe(self, audio_path: Path | str) -> TranscriptResult:
        """Transcribe canonical WAV file with VAD and anti-hallucination guards.

        Args:
            audio_path: Path to canonical 16 kHz mono WAV.

        Returns:
            TranscriptResult with text, language, confidence, and reliability flags.

        Raises:
            RuntimeError: If the transcriber model is not loaded.
        """
        if not self.is_loaded() or self.model is None:
            self.load()

        if not self.is_loaded() or self.model is None:
            raise RuntimeError("faster-whisper model is not loaded or unavailable")

        path = Path(audio_path)
        start_t = time.perf_counter()

        y, sr = load_canonical(path)
        speech_ratio = compute_speech_ratio(y, sr=sr)

        # Pure silence guard: if speech ratio is negligible (<0.08), suppress output
        if speech_ratio < 0.08:
            return TranscriptResult(
                text="",
                language="unknown",
                language_probability=0.0,
                language_supported=False,
                segments=[],
                mean_confidence=0.0,
                is_reliable=False,
                word_count=0,
                no_speech_ratio=1.0 - speech_ratio,
                hallucination_flags=["SILENCE_SUPPRESSED"],
                model_version=self.version,
                inference_ms=int((time.perf_counter() - start_t) * 1000),
            )

        # Transcribe with faster-whisper per 05 §3.1
        segments_raw, info = self.model.transcribe(
            str(path),
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,
            condition_on_previous_text=False,  # prevents hallucination loops
            temperature=[0.0, 0.2, 0.4],
        )

        detected_lang = info.language
        lang_prob = float(info.language_probability)
        lang_supported = detected_lang in SUPPORTED_LANGUAGES

        extracted_segments: list[Segment] = []
        confidences: list[float] = []
        durations: list[float] = []
        hallucination_flags: list[str] = []

        full_text_pieces: list[str] = []

        for seg in segments_raw:
            # Guard 4: Drop segments with extreme compression ratio (> 2.4)
            if hasattr(seg, "compression_ratio") and seg.compression_ratio > 2.4:
                hallucination_flags.append(f"COMPRESSION_RATIO_EXCEEDED_{seg.id}")
                continue

            seg_text = seg.text.strip()
            if not seg_text:
                continue

            seg_dur = max(0.1, seg.end - seg.start)
            seg_conf = float(np.exp(getattr(seg, "avg_logprob", -0.5)))
            seg_conf = min(1.0, max(0.0, seg_conf))

            extracted_segments.append(
                Segment(
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=seg_text,
                    avg_logprob=round(getattr(seg, "avg_logprob", 0.0), 3),
                )
            )
            confidences.append(seg_conf)
            durations.append(seg_dur)
            full_text_pieces.append(seg_text)

        raw_full_text = " ".join(full_text_pieces).strip()

        # Guard 3: 5-gram repetition check
        cleaned_text, was_truncated = check_5gram_repetition(raw_full_text)
        if was_truncated:
            hallucination_flags.append("REPETITION_LOOP_TRUNCATED")

        # Duration-weighted confidence calculation
        if durations and sum(durations) > 0:
            mean_conf = float(np.average(confidences, weights=durations))
        elif confidences:
            mean_conf = float(np.mean(confidences))
        else:
            mean_conf = 0.0

        words = cleaned_text.split()
        word_count = len(words)
        no_speech_ratio = 1.0 - speech_ratio

        # Reliability thresholding per 05 §3.1:
        # (mean_conf >= 0.55) and (no_speech_ratio < 0.6) and (len(text.split()) >= 5)
        is_reliable = (
            (mean_conf >= 0.55)
            and (no_speech_ratio < 0.60)
            and (word_count >= 5)
            and (lang_prob >= 0.50)
            and ("REPETITION_LOOP_TRUNCATED" not in hallucination_flags)
        )

        return TranscriptResult(
            text=cleaned_text,
            language=detected_lang,
            language_probability=round(lang_prob, 3),
            language_supported=lang_supported,
            segments=extracted_segments,
            mean_confidence=round(mean_conf, 3),
            is_reliable=is_reliable,
            word_count=word_count,
            no_speech_ratio=round(no_speech_ratio, 3),
            hallucination_flags=hallucination_flags,
            model_version=self.version,
            inference_ms=int((time.perf_counter() - start_t) * 1000),
        )
