"""
audio_pipeline.py — Whisper transcription and three behavioural signal extractors.

Signals extracted per 30-second chunk:
  - Pace score    : words-per-second (slower speech → higher score)
  - Repetition    : Jaccard similarity of bigrams between adjacent chunks
  - Keyword score : density of exam-signal emphasis phrases
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import config
import utils


# ─────────────────────────────────────────────────────────────────────────────
# Transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe an audio file using OpenAI Whisper with word-level timestamps.

    Args:
        audio_path: Absolute or relative path to an audio file (.mp3, .wav, .m4a).

    Returns:
        The full Whisper result dict containing ``"segments"``, ``"text"``, and
        per-word ``"start"`` / ``"end"`` timestamps.

    Raises:
        FileNotFoundError: If the audio file does not exist at *audio_path*.
        RuntimeError:      If the Whisper model fails to load or transcribe.
    """
    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(
            f"[audio_pipeline] Audio file not found: {source.resolve()}\n"
            "Please check the path and try again."
        )

    try:
        import whisper  # lazy import — only needed when actually transcribing
    except ImportError:
        raise RuntimeError(
            "[audio_pipeline] openai-whisper is not installed.\n"
            "Run: pip install openai-whisper"
        )

    print(f"[audio_pipeline] Loading Whisper model '{config.WHISPER_MODEL}' …")
    try:
        model = whisper.load_model(config.WHISPER_MODEL)
    except Exception as exc:
        raise RuntimeError(
            f"[audio_pipeline] Failed to load Whisper model '{config.WHISPER_MODEL}': {exc}\n"
            "Check your internet connection — the model may need to be downloaded."
        ) from exc

    print(f"[audio_pipeline] Transcribing {source.name} …")
    try:
        result: dict = model.transcribe(
            str(source),
            word_timestamps=True,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[audio_pipeline] Transcription failed for '{source}': {exc}"
        ) from exc

    # Guard against silent / empty recordings
    transcript_text = result.get("text", "").strip()
    if not transcript_text:
        warnings.warn(
            "[audio_pipeline] Whisper returned an empty transcript — "
            "the audio may contain only silence.",
            UserWarning,
            stacklevel=2,
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Signal 1 — Speech pace (slower → more emphasis → higher score)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pace_scores(chunks: list[dict]) -> list[float]:
    """
    Compute a normalised pace score for each transcript chunk.

    A lower words-per-second rate signals deliberate, emphasised delivery.
    After normalisation (0 → 1), the scores are *inverted* so that slower
    speech maps to a higher importance score.

    Args:
        chunks: List of chunk dicts as produced by ``utils.chunk_transcript``.
                Each chunk must have ``"words"``, ``"start"``, and ``"end"`` keys.

    Returns:
        A list of floats in [0.0, 1.0] with the same length as *chunks*.
        Returns [0.0] for a single-chunk input (nothing to normalise against).
    """
    if not chunks:
        return []

    raw_rates: list[float] = []
    for chunk in chunks:
        duration = max(chunk["end"] - chunk["start"], 1e-6)
        word_count = len(chunk.get("words", []))
        raw_rates.append(word_count / duration)

    if len(raw_rates) == 1:
        return [0.0]

    min_rate = min(raw_rates)
    max_rate = max(raw_rates)

    # Normalise, then invert: fastest speaker → 0.0, slowest → 1.0
    pace_scores: list[float] = []
    for rate in raw_rates:
        normalised = utils.normalize_score(rate, min_rate, max_rate)
        pace_scores.append(round(1.0 - normalised, 4))

    return pace_scores


# ─────────────────────────────────────────────────────────────────────────────
# Signal 2 — Repetition (Jaccard similarity of bigrams with previous chunk)
# ─────────────────────────────────────────────────────────────────────────────

def compute_repetition_scores(chunks: list[dict]) -> list[float]:
    """
    Compute a normalised repetition score between each chunk and its predecessor.

    Uses Jaccard similarity on bigram sets:
        similarity = |A ∩ B| / |A ∪ B|

    A high score means similar vocabulary is reused, indicating emphasis through
    repetition.

    Args:
        chunks: List of chunk dicts each with at minimum a ``"text"`` key.

    Returns:
        A list of floats in [0.0, 1.0].  The first element is always 0.0
        (no predecessor to compare against).
    """
    if not chunks:
        return []

    raw_scores: list[float] = [0.0]  # first chunk has no predecessor

    for i in range(1, len(chunks)):
        prev_bigrams = set(utils.get_bigrams(chunks[i - 1]["text"]))
        curr_bigrams = set(utils.get_bigrams(chunks[i]["text"]))

        union = prev_bigrams | curr_bigrams
        if not union:
            raw_scores.append(0.0)
            continue

        intersection = prev_bigrams & curr_bigrams
        jaccard = len(intersection) / len(union)
        raw_scores.append(round(jaccard, 4))

    # Normalise to 0-1 (already bounded but ensures consistency)
    min_s = min(raw_scores)
    max_s = max(raw_scores)
    return [round(utils.normalize_score(s, min_s, max_s), 4) for s in raw_scores]


# ─────────────────────────────────────────────────────────────────────────────
# Signal 3 — Emphasis keyword density
# ─────────────────────────────────────────────────────────────────────────────

def compute_keyword_scores(chunks: list[dict], keywords: list[str]) -> list[float]:
    """
    Score each chunk by how many exam-signal emphasis phrases it contains.

    Scoring rule: ``min(keyword_count / 3, 1.0)`` — three or more keywords
    in a single chunk yields the maximum score of 1.0.

    Args:
        chunks:   List of chunk dicts each with a ``"text"`` key.
        keywords: List of emphasis keyword/phrase strings (from ``config.EMPHASIS_KEYWORDS``).

    Returns:
        A list of floats in [0.0, 1.0], one per chunk.
    """
    if not chunks:
        return []

    # Pre-lowercase all keywords for case-insensitive matching
    lc_keywords = [kw.lower() for kw in keywords]

    scores: list[float] = []
    for chunk in chunks:
        text_lower = chunk.get("text", "").lower()
        count = sum(1 for kw in lc_keywords if kw in text_lower)
        scores.append(round(min(count / 3.0, 1.0), 4))

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_audio_pipeline(audio_path: str) -> list[dict]:
    """
    End-to-end orchestration of audio transcription and signal extraction.

    Steps:
      1. Transcribe the audio with Whisper (word-level timestamps).
      2. Split the transcript into 30-second chunks.
      3. Compute pace, repetition, and keyword scores for each chunk.
      4. Return a list of enriched chunk dicts.

    Args:
        audio_path: Path to the audio file to analyse.

    Returns:
        A list of chunk dicts each containing:
          - ``start``             (float)
          - ``end``               (float)
          - ``text``              (str)
          - ``words``             (list)
          - ``pace_score``        (float, 0-1)
          - ``repetition_score``  (float, 0-1)
          - ``keyword_score``     (float, 0-1)

    Raises:
        FileNotFoundError: Propagated from ``transcribe_audio``.
        RuntimeError:      Propagated from ``transcribe_audio``.
    """
    whisper_result = transcribe_audio(audio_path)

    chunks = utils.chunk_transcript(whisper_result, chunk_duration=config.CHUNK_DURATION_SECONDS)

    if not chunks:
        warnings.warn(
            "[audio_pipeline] No transcript chunks produced — "
            "the recording may be silent or too short.",
            UserWarning,
            stacklevel=2,
        )
        return []

    pace_scores = compute_pace_scores(chunks)
    repetition_scores = compute_repetition_scores(chunks)
    keyword_scores = compute_keyword_scores(chunks, config.EMPHASIS_KEYWORDS)

    enriched: list[dict] = []
    for i, chunk in enumerate(chunks):
        enriched.append(
            {
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "words": chunk["words"],
                "pace_score": pace_scores[i],
                "repetition_score": repetition_scores[i],
                "keyword_score": keyword_scores[i],
            }
        )

    print(f"[audio_pipeline] Produced {len(enriched)} chunks from '{Path(audio_path).name}'.")
    return enriched
