"""
audio_pipeline.py — Whisper transcription and three behavioural signal extractors.

Accuracy improvements over v1:
  - Repetition score uses TF-IDF cosine similarity (not raw Jaccard on bigrams)
  - Keyword score uses weighted phrase matching from config.EMPHASIS_KEYWORD_WEIGHTS

Signals extracted per 30-second chunk:
  - Pace score       : words-per-second (slower speech → higher score)
  - Repetition score : TF-IDF cosine similarity between adjacent chunks
  - Keyword score    : weighted sum of emphasis phrase matches
"""

from __future__ import annotations

import warnings
from pathlib import Path

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
        The full Whisper result dict with ``"segments"`` containing per-word
        ``"start"`` / ``"end"`` timestamps.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError:      If the Whisper model fails to load or transcribe.
    """
    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(
            f"[audio_pipeline] Audio file not found: {source.resolve()}\n"
            "Please check the path and try again."
        )

    try:
        import whisper
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
            f"[audio_pipeline] Failed to load Whisper '{config.WHISPER_MODEL}': {exc}\n"
            "Check your internet connection — model weights may need downloading."
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

    if not result.get("text", "").strip():
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

    Lower words-per-second (slower delivery) maps to a higher importance score
    after inversion.

    Args:
        chunks: List of chunk dicts from ``utils.chunk_transcript``.

    Returns:
        List of floats in [0.0, 1.0], length == len(chunks).
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

    min_rate, max_rate = min(raw_rates), max(raw_rates)

    # Invert: fastest speaker → 0.0, slowest → 1.0
    return [
        round(1.0 - utils.normalize_score(r, min_rate, max_rate), 4)
        for r in raw_rates
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Signal 2 — Repetition via TF-IDF cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

def _build_tfidf_matrix(texts: list[str]):
    """
    Build a TF-IDF matrix from a list of text strings.

    Returns (matrix, vectorizer) or (None, None) if sklearn is unavailable.

    Args:
        texts: List of document strings.

    Returns:
        Tuple of (sparse TF-IDF matrix, fitted TfidfVectorizer) or (None, None).
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
    except ImportError:
        return None, None

    if len(texts) < 2:
        return None, None

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        stop_words="english",
    )
    try:
        matrix = vectorizer.fit_transform(texts)
        matrix = normalize(matrix, norm="l2")
        return matrix, vectorizer
    except ValueError:
        return None, None


def compute_repetition_scores(chunks: list[dict]) -> list[float]:
    """
    Compute a normalised repetition score between each chunk and its predecessor
    using TF-IDF cosine similarity.

    Cosine similarity on TF-IDF vectors is far more sensitive to meaningful
    semantic repetition than raw Jaccard on bigrams, because stop-word bigrams
    are down-weighted by IDF while content-word bigrams are up-weighted.

    Falls back to Jaccard similarity if scikit-learn is not installed.

    Args:
        chunks: List of chunk dicts each with a ``"text"`` key.

    Returns:
        List of floats in [0.0, 1.0].  First element is always 0.0.
    """
    if not chunks:
        return []

    texts = [utils.clean_text(c.get("text", "")) for c in chunks]
    matrix, _ = _build_tfidf_matrix(texts)

    if matrix is not None:
        # Cosine similarity between consecutive rows (already L2-normalised)
        raw_scores: list[float] = [0.0]
        for i in range(1, len(chunks)):
            sim = float((matrix[i - 1] * matrix[i].T).toarray()[0][0])
            raw_scores.append(round(max(0.0, min(sim, 1.0)), 4))
    else:
        # Jaccard fallback
        raw_scores = [0.0]
        for i in range(1, len(chunks)):
            prev_bg = set(utils.get_bigrams(chunks[i - 1]["text"]))
            curr_bg = set(utils.get_bigrams(chunks[i]["text"]))
            union = prev_bg | curr_bg
            raw_scores.append(round(len(prev_bg & curr_bg) / len(union), 4) if union else 0.0)

    # Normalise to 0-1 for consistency with other signals
    min_s, max_s = min(raw_scores), max(raw_scores)
    return [round(utils.normalize_score(s, min_s, max_s), 4) for s in raw_scores]


# ─────────────────────────────────────────────────────────────────────────────
# Signal 3 — Weighted emphasis keyword density
# ─────────────────────────────────────────────────────────────────────────────

def compute_keyword_scores(
    chunks: list[dict],
    keyword_weights: dict[str, float] | None = None,
) -> list[float]:
    """
    Score each chunk by the weighted sum of emphasis phrases it contains.

    Uses ``config.EMPHASIS_KEYWORD_WEIGHTS`` by default.  The weighted sum is
    normalised by ``config.KEYWORD_SCORE_NORMALISER`` (default 2.0) so that
    two full-weight keywords in a chunk = max score 1.0.

    A strong keyword like "this will be on the exam" (weight 1.0) contributes
    more than a weak signal like "note this" (weight 0.4), giving a much more
    discriminating score than the old binary count method.

    Args:
        chunks:          List of chunk dicts each with a ``"text"`` key.
        keyword_weights: Optional override dict mapping keyword → weight.
                         Defaults to ``config.EMPHASIS_KEYWORD_WEIGHTS``.

    Returns:
        List of floats in [0.0, 1.0], one per chunk.
    """
    if not chunks:
        return []

    kw_weights = keyword_weights if keyword_weights is not None else config.EMPHASIS_KEYWORD_WEIGHTS
    # Pre-lowercase keys for case-insensitive matching
    lc_weights = {kw.lower(): w for kw, w in kw_weights.items()}
    normaliser = config.KEYWORD_SCORE_NORMALISER

    scores: list[float] = []
    for chunk in chunks:
        text_lower = chunk.get("text", "").lower()
        weighted_sum = sum(w for kw, w in lc_weights.items() if kw in text_lower)
        scores.append(round(min(weighted_sum / normaliser, 1.0), 4))

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_audio_pipeline(audio_path: str) -> list[dict]:
    """
    End-to-end orchestration: transcription → chunking → signal extraction.

    Args:
        audio_path: Path to the audio file to analyse.

    Returns:
        List of chunk dicts each containing:
          - ``start``, ``end``                  (float)
          - ``text``, ``words``                 (str, list)
          - ``pace_score``                      (float 0-1)
          - ``repetition_score``                (float 0-1, TF-IDF cosine)
          - ``keyword_score``                   (float 0-1, weighted)

    Raises:
        FileNotFoundError: Propagated from ``transcribe_audio``.
        RuntimeError:      Propagated from ``transcribe_audio``.
    """
    whisper_result = transcribe_audio(audio_path)
    chunks = utils.chunk_transcript(whisper_result, chunk_duration=config.CHUNK_DURATION_SECONDS)

    if not chunks:
        warnings.warn(
            "[audio_pipeline] No transcript chunks produced — "
            "recording may be silent or too short.",
            UserWarning,
            stacklevel=2,
        )
        return []

    pace_scores = compute_pace_scores(chunks)
    repetition_scores = compute_repetition_scores(chunks)
    keyword_scores = compute_keyword_scores(chunks)

    enriched: list[dict] = []
    for i, chunk in enumerate(chunks):
        enriched.append({
            "start": chunk["start"],
            "end": chunk["end"],
            "text": chunk["text"],
            "words": chunk["words"],
            "pace_score": pace_scores[i],
            "repetition_score": repetition_scores[i],
            "keyword_score": keyword_scores[i],
        })

    print(f"[audio_pipeline] Produced {len(enriched)} chunks from '{Path(audio_path).name}'.")
    return enriched
