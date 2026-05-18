"""
utils.py — Shared helper functions for Lecture Ghost.

Provides text cleaning, chunking, bigram extraction, score normalisation,
and JSON persistence utilities used across all pipeline modules.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


# ─── Filler words removed during cleaning ────────────────────────────────────
_FILLER_WORDS: set[str] = {"uh", "um", "like", "you", "know", "so"}


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise a raw text string for downstream NLP processing.

    Steps:
      1. Lowercase all characters.
      2. Remove all punctuation characters.
      3. Strip filler words ("uh", "um", "like", "you know", "so").
      4. Collapse consecutive whitespace into a single space and strip
         leading / trailing whitespace.

    Args:
        text: Raw string to clean.

    Returns:
        A cleaned, lowercase, punctuation-free string.
    """
    if not text:
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()
    tokens = [t for t in tokens if t not in _FILLER_WORDS]
    return " ".join(tokens)


def get_bigrams(text: str) -> list[tuple[str, str]]:
    """
    Extract all consecutive bigrams (pairs of adjacent words) from text.

    The input is cleaned before extraction so callers do not need to
    pre-process it themselves.

    Args:
        text: Raw or pre-cleaned string.

    Returns:
        A list of (word_a, word_b) tuples in document order.  Returns an
        empty list when the cleaned text has fewer than two tokens.
    """
    tokens = clean_text(text).split()
    if len(tokens) < 2:
        return []
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def normalize_score(value: float, min_val: float, max_val: float) -> float:
    """
    Apply min-max normalisation to map *value* into the range [0.0, 1.0].

    Args:
        value:   The raw score to normalise.
        min_val: The minimum observed value in the dataset.
        max_val: The maximum observed value in the dataset.

    Returns:
        A float in [0.0, 1.0].  Returns 0.0 when min_val == max_val to
        avoid division by zero.
    """
    if min_val == max_val:
        return 0.0
    normalised = (value - min_val) / (max_val - min_val)
    return float(max(0.0, min(1.0, normalised)))


# ─────────────────────────────────────────────────────────────────────────────
# Whisper transcript chunking
# ─────────────────────────────────────────────────────────────────────────────

def chunk_transcript(
    whisper_result: dict,
    chunk_duration: int = 30,
) -> list[dict]:
    """
    Split a Whisper transcription result into fixed-duration chunks.

    Whisper returns word-level timestamps when called with
    ``word_timestamps=True``.  This function groups those words into
    non-overlapping windows of *chunk_duration* seconds.

    Args:
        whisper_result: The raw dict returned by ``whisper.load_model(...).transcribe()``.
                        Must contain a ``"segments"`` key, each segment having a
                        ``"words"`` list with ``{"word", "start", "end"}`` dicts.
        chunk_duration: Target length of each chunk in seconds (default 30).

    Returns:
        A list of chunk dicts, each with keys:
          - ``start``  (float): chunk start time in seconds
          - ``end``    (float): chunk end time in seconds
          - ``text``   (str):   concatenated word text for the chunk
          - ``words``  (list):  original word-level dicts included in chunk

    Raises:
        KeyError: If ``whisper_result`` does not contain ``"segments"``.
    """
    # Flatten all word-level dicts from all segments
    all_words: list[dict] = []
    for segment in whisper_result.get("segments", []):
        words = segment.get("words", [])
        if words:
            all_words.extend(words)
        else:
            # Segment has no word timestamps — synthesise a single entry
            all_words.append(
                {
                    "word": segment.get("text", "").strip(),
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                }
            )

    if not all_words:
        return []

    chunks: list[dict] = []
    chunk_start = all_words[0].get("start", 0.0)
    current_words: list[dict] = []

    for word_info in all_words:
        word_start = word_info.get("start", 0.0)

        # Start a new chunk when we exceed the chunk window
        if word_start >= chunk_start + chunk_duration and current_words:
            chunk_end = current_words[-1].get("end", word_start)
            chunks.append(
                {
                    "start": chunk_start,
                    "end": chunk_end,
                    "text": " ".join(w.get("word", "").strip() for w in current_words),
                    "words": current_words,
                }
            )
            chunk_start = word_start
            current_words = []

        current_words.append(word_info)

    # Flush the final (potentially partial) chunk
    if current_words:
        chunk_end = current_words[-1].get("end", chunk_start + chunk_duration)
        chunks.append(
            {
                "start": chunk_start,
                "end": chunk_end,
                "text": " ".join(w.get("word", "").strip() for w in current_words),
                "words": current_words,
            }
        )

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# JSON persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_json(data: dict, path: str) -> None:
    """
    Serialise *data* to a pretty-printed JSON file at *path*.

    Parent directories are created automatically if they do not exist.

    Args:
        data: A JSON-serialisable dictionary.
        path: Destination file path (string or path-like).

    Raises:
        TypeError:    If *data* contains non-serialisable objects.
        OSError:      If the file cannot be written due to permissions.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    print(f"[utils] Report saved → {destination.resolve()}")


def load_json(path: str) -> dict:
    """
    Load a JSON file from *path* and return it as a dict.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed dictionary, or an empty dict if the file does not exist
        or cannot be decoded.
    """
    source = Path(path)
    if not source.exists():
        print(f"[utils] JSON file not found: {source} — returning empty dict.")
        return {}
    try:
        with source.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[utils] Failed to parse JSON at {source}: {exc} — returning empty dict.")
        return {}
