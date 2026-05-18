"""
scorer.py — Cross-modal weighted importance scorer.

Accuracy improvements over v1:
  - Semantic overlap uses spaCy word-vector similarity instead of exact string match
  - Audio-only mode re-normalises weights when no OCR reference is available
  - Chunk topic extraction passes per-chunk context for better TF-IDF ranking
"""

from __future__ import annotations

import warnings
from typing import Optional

import config
import utils
import ocr_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk topic extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_chunk_topics(chunk_text: str) -> list[str]:
    """
    Extract key noun-chunk and named-entity topics from a single chunk's text.

    Delegates to ``ocr_pipeline.extract_topics_from_text`` so the spaCy model
    is loaded once and cached by spaCy internally.

    Args:
        chunk_text: Raw transcript text for one chunk.

    Returns:
        A list of lowercase topic strings (up to ``config.TFIDF_TOP_N_TOPICS``).
    """
    return ocr_pipeline.extract_topics_from_text(chunk_text)


# ─────────────────────────────────────────────────────────────────────────────
# Semantic topic similarity helper
# ─────────────────────────────────────────────────────────────────────────────

def _topics_match_semantically(
    topic_a: str,
    topic_b: str,
    nlp,
    threshold: float,
) -> bool:
    """
    Return True if two topic strings are semantically similar above *threshold*.

    Uses spaCy word vectors (requires en_core_web_md or larger).  Falls back to
    exact substring matching when vectors are unavailable (e.g. en_core_web_sm).

    Args:
        topic_a:   First topic string.
        topic_b:   Second topic string.
        nlp:       A loaded spaCy Language model.
        threshold: Minimum similarity score to count as a match.

    Returns:
        bool
    """
    doc_a = nlp(topic_a)
    doc_b = nlp(topic_b)

    # Check if vectors are available (en_core_web_md/lg)
    if doc_a.has_vector and doc_b.has_vector:
        return doc_a.similarity(doc_b) >= threshold

    # Fallback: exact substring containment
    return topic_a in topic_b or topic_b in topic_a


# ─────────────────────────────────────────────────────────────────────────────
# Cross-modal overlap score
# ─────────────────────────────────────────────────────────────────────────────

def compute_overlap_score(
    chunk_topics: list[str],
    exam_topics: list[str],
    assignment_topics: list[str],
    nlp=None,
) -> float:
    """
    Measure how many chunk topics semantically match past exam or assignment topics.

    Upgraded formula (v2):
        matches = count of chunk_topics where spaCy similarity ≥ SEMANTIC_SIMILARITY_THRESHOLD
                  against any topic in the reference set
        score   = matches / max(len(chunk_topics), 1)

    Catches near-matches like "neural net" ↔ "neural network" or
    "gradient descent" ↔ "gradient" that exact matching would miss.

    Args:
        chunk_topics:      Topics from the audio chunk transcript.
        exam_topics:       Topics from uploaded exam paper files.
        assignment_topics: Topics from uploaded assignment files.
        nlp:               Loaded spaCy model (optional — loaded lazily if None).

    Returns:
        Float in [0.0, 1.0].  Returns 0.0 when reference set is empty.
    """
    reference_topics = list(set(exam_topics) | set(assignment_topics))

    if not reference_topics or not chunk_topics:
        return 0.0

    # Lazy-load spaCy if not passed in
    if nlp is None:
        try:
            import spacy
            try:
                nlp = spacy.load(config.SPACY_MODEL)
            except OSError:
                nlp = spacy.load("en_core_web_sm")
        except Exception:
            # Fallback to exact match if spaCy entirely unavailable
            exact_matches = sum(1 for t in chunk_topics if t in set(reference_topics))
            return round(min(exact_matches / max(len(chunk_topics), 1), 1.0), 4)

    threshold = config.SEMANTIC_SIMILARITY_THRESHOLD
    reference_set = set(reference_topics)

    matches = 0
    for chunk_topic in chunk_topics:
        # Fast path: exact match
        if chunk_topic in reference_set:
            matches += 1
            continue
        # Semantic path: check against all reference topics
        for ref_topic in reference_topics:
            if _topics_match_semantically(chunk_topic, ref_topic, nlp, threshold):
                matches += 1
                break  # count each chunk_topic at most once

    raw_score = matches / max(len(chunk_topics), 1)
    return round(min(max(raw_score, 0.0), 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Weighted final score
# ─────────────────────────────────────────────────────────────────────────────

def compute_final_score(
    pace: float,
    repetition: float,
    keyword: float,
    overlap: float,
    audio_only: bool = False,
) -> float:
    """
    Compute the weighted composite importance score for a single chunk.

    In audio-only mode (no exam papers / assignments uploaded), the overlap
    weight is redistributed proportionally to the other three signals so
    the scoring remains meaningful.

    Args:
        pace:       Normalised pace score  (0-1).
        repetition: Normalised repetition score (0-1, TF-IDF cosine).
        keyword:    Normalised keyword score (0-1, weighted).
        overlap:    Cross-modal overlap score (0-1, semantic).
        audio_only: If True, redistribute overlap weight to other signals.

    Returns:
        Float in [0.0, 1.0].
    """
    weights = dict(config.SCORING_WEIGHTS)

    if audio_only:
        # Redistribute overlap weight proportionally among the other three
        removed = weights.pop("overlap_score")
        total_remaining = sum(weights.values())
        for k in weights:
            weights[k] += removed * (weights[k] / total_remaining)
        overlap = 0.0

    score = (
        weights.get("pace_score", 0.20) * pace
        + weights.get("repetition_score", 0.25) * repetition
        + weights.get("keyword_score", 0.25) * keyword
        + weights.get("overlap_score", 0.30) * overlap
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def score_all_chunks(
    audio_chunks: list[dict],
    ocr_results: list[dict],
) -> list[dict]:
    """
    Enrich every audio chunk with cross-modal semantic scores and an importance label.

    Args:
        audio_chunks: List of chunk dicts from ``audio_pipeline.run_audio_pipeline``.
        ocr_results:  List of processed source dicts from ``ocr_pipeline.run_ocr_pipeline``.

    Returns:
        Enriched chunk list with additional per-chunk keys:
          - ``chunk_topics``     (list[str])
          - ``overlap_score``    (float, semantic)
          - ``final_score``      (float)
          - ``importance_label`` (str: "high" | "medium" | "low")
    """
    if not audio_chunks:
        warnings.warn("[scorer] No audio chunks to score — returning empty list.", stacklevel=2)
        return []

    # ── Aggregate reference topics by source type ─────────────────────────
    exam_topics: list[str] = []
    assignment_topics: list[str] = []

    for item in ocr_results:
        src = item.get("source_type", "")
        topics = item.get("topics", [])
        if src == "exam_paper":
            exam_topics.extend(topics)
        elif src == "assignment":
            assignment_topics.extend(topics)

    exam_topics = list(dict.fromkeys(exam_topics))
    assignment_topics = list(dict.fromkeys(assignment_topics))
    audio_only = not exam_topics and not assignment_topics

    if audio_only:
        print("[scorer] No exam/assignment topics — audio-only mode (overlap weight redistributed).")

    # ── Pre-load spaCy once for semantic similarity ───────────────────────
    nlp = None
    if not audio_only:
        try:
            import spacy
            try:
                nlp = spacy.load(config.SPACY_MODEL)
            except OSError:
                nlp = spacy.load("en_core_web_sm")
                warnings.warn(
                    f"[scorer] '{config.SPACY_MODEL}' not found — using en_core_web_sm "
                    "(no word vectors; falling back to substring matching).",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception as exc:
            warnings.warn(f"[scorer] Could not load spaCy: {exc} — using exact topic match.", stacklevel=2)

    # ── Score each chunk ──────────────────────────────────────────────────
    scored: list[dict] = []
    for chunk in audio_chunks:
        chunk_topics = extract_chunk_topics(chunk.get("text", ""))

        overlap = compute_overlap_score(
            chunk_topics, exam_topics, assignment_topics, nlp=nlp
        ) if not audio_only else 0.0

        final = compute_final_score(
            pace=chunk.get("pace_score", 0.0),
            repetition=chunk.get("repetition_score", 0.0),
            keyword=chunk.get("keyword_score", 0.0),
            overlap=overlap,
            audio_only=audio_only,
        )

        label = (
            "high" if final > config.HIGH_THRESHOLD
            else "medium" if final > config.MEDIUM_THRESHOLD
            else "low"
        )

        enriched = dict(chunk)
        enriched.update({
            "chunk_topics": chunk_topics,
            "overlap_score": overlap,
            "final_score": final,
            "importance_label": label,
        })
        scored.append(enriched)

    counts = {lbl: sum(1 for c in scored if c["importance_label"] == lbl) for lbl in ("high", "medium", "low")}
    print(f"[scorer] Scored {len(scored)} chunks — high: {counts['high']}, medium: {counts['medium']}, low: {counts['low']}.")
    return scored
