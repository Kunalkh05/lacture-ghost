"""
scorer.py — Cross-modal weighted importance scorer.

Combines three audio signals (pace, repetition, keyword) with one cross-modal
signal (topic overlap with exam papers and assignments) to produce a final
importance score for each lecture chunk.
"""

from __future__ import annotations

import warnings

import config
import utils
import ocr_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Per-chunk topic extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_chunk_topics(chunk_text: str) -> list[str]:
    """
    Extract key noun-chunk topics from a single lecture chunk's transcript text.

    This is a thin wrapper around ``ocr_pipeline.extract_topics_from_text`` so
    that spaCy is only loaded once (cached internally by spaCy itself).

    Args:
        chunk_text: Raw transcript text for one chunk.

    Returns:
        A list of lowercase topic strings (up to 20).
    """
    return ocr_pipeline.extract_topics_from_text(chunk_text)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-modal overlap score
# ─────────────────────────────────────────────────────────────────────────────

def compute_overlap_score(
    chunk_topics: list[str],
    exam_topics: list[str],
    assignment_topics: list[str],
) -> float:
    """
    Measure how many topics in a chunk also appear in past exam or assignment text.

    Score formula:
        matches / max(len(chunk_topics), 1)

    Where *matches* is the count of chunk topics that appear in the combined
    reference set (exam + assignment topics).

    Args:
        chunk_topics:      Topics extracted from the audio chunk transcript.
        exam_topics:       Topics extracted from all uploaded exam paper files.
        assignment_topics: Topics extracted from all uploaded assignment files.

    Returns:
        A float in [0.0, 1.0].  Returns 0.0 when the reference set is empty
        (i.e. no exam papers or assignments were uploaded — audio-only mode).
    """
    reference_set = set(exam_topics) | set(assignment_topics)

    if not reference_set:
        return 0.0

    if not chunk_topics:
        return 0.0

    matches = sum(1 for topic in chunk_topics if topic in reference_set)
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
) -> float:
    """
    Compute the weighted composite importance score for a single chunk.

    Weights are sourced from ``config.SCORING_WEIGHTS`` and must sum to 1.0.

    Args:
        pace:       Normalised pace score  (0-1; higher = slower speech).
        repetition: Normalised repetition score (0-1; higher = more repetition).
        keyword:    Normalised keyword score (0-1; higher = more emphasis phrases).
        overlap:    Cross-modal overlap score (0-1; higher = more topic matches).

    Returns:
        A float in [0.0, 1.0] representing the chunk's predicted exam importance.
    """
    weights = config.SCORING_WEIGHTS
    score = (
        weights["pace_score"] * pace
        + weights["repetition_score"] * repetition
        + weights["keyword_score"] * keyword
        + weights["overlap_score"] * overlap
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
    Enrich every audio chunk with cross-modal scores and an importance label.

    Processing pipeline per chunk:
      1. Extract topics from the chunk's transcript text via spaCy.
      2. Compute overlap_score against aggregated exam + assignment topics.
      3. Compute final_score using all four signals.
      4. Assign an importance_label: ``"high"``, ``"medium"``, or ``"low"``.

    Args:
        audio_chunks: List of chunk dicts from ``audio_pipeline.run_audio_pipeline``.
                      Each dict must already contain ``pace_score``,
                      ``repetition_score``, ``keyword_score``, and ``text``.
        ocr_results:  List of processed source dicts from
                      ``ocr_pipeline.run_ocr_pipeline``.  May be empty if
                      no supplementary files were uploaded (audio-only mode).

    Returns:
        The same list of chunk dicts with these additional keys per chunk:
          - ``chunk_topics``   (list[str])
          - ``overlap_score``  (float)
          - ``final_score``    (float)
          - ``importance_label`` (str: "high" | "medium" | "low")
    """
    if not audio_chunks:
        warnings.warn(
            "[scorer] No audio chunks to score — returning empty list.",
            UserWarning,
            stacklevel=2,
        )
        return []

    # ── Aggregate topics from OCR sources by type ─────────────────────────
    exam_topics: list[str] = []
    assignment_topics: list[str] = []

    for ocr_item in ocr_results:
        src_type = ocr_item.get("source_type", "")
        topics = ocr_item.get("topics", [])
        if src_type == "exam_paper":
            exam_topics.extend(topics)
        elif src_type == "assignment":
            assignment_topics.extend(topics)
        # "slide" topics are not used for overlap scoring (audio already covers that)

    # Deduplicate
    exam_topics = list(dict.fromkeys(exam_topics))
    assignment_topics = list(dict.fromkeys(assignment_topics))

    if not exam_topics and not assignment_topics:
        print(
            "[scorer] No exam paper or assignment topics found — "
            "overlap_score will be 0.0 for all chunks (audio-only mode)."
        )

    # ── Score each chunk ──────────────────────────────────────────────────
    scored: list[dict] = []
    for chunk in audio_chunks:
        chunk_topics = extract_chunk_topics(chunk.get("text", ""))

        overlap = compute_overlap_score(chunk_topics, exam_topics, assignment_topics)
        final = compute_final_score(
            pace=chunk.get("pace_score", 0.0),
            repetition=chunk.get("repetition_score", 0.0),
            keyword=chunk.get("keyword_score", 0.0),
            overlap=overlap,
        )

        # Assign label
        if final > config.HIGH_THRESHOLD:
            label = "high"
        elif final > config.MEDIUM_THRESHOLD:
            label = "medium"
        else:
            label = "low"

        enriched_chunk = dict(chunk)
        enriched_chunk.update(
            {
                "chunk_topics": chunk_topics,
                "overlap_score": overlap,
                "final_score": final,
                "importance_label": label,
            }
        )
        scored.append(enriched_chunk)

    high_count = sum(1 for c in scored if c["importance_label"] == "high")
    print(
        f"[scorer] Scored {len(scored)} chunks — "
        f"{high_count} high-importance, "
        f"{sum(1 for c in scored if c['importance_label'] == 'medium')} medium, "
        f"{sum(1 for c in scored if c['importance_label'] == 'low')} low."
    )
    return scored
