"""
topic_ranker.py — Aggregates topics across all sources and ranks the top N by
predicted exam likelihood.

Accuracy improvements over v1:
  - IDF-weighted ranking: rare high-scoring topics rank above common mediocre ones
  - Boosts still applied (×1.5 exam, ×1.2 assignment) after IDF weighting
  - Ties broken by lexicographic order for determinism
"""

from __future__ import annotations

import math
from collections import defaultdict

import config
import utils


# ─────────────────────────────────────────────────────────────────────────────
# Score aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_topic_scores(scored_chunks: list[dict]) -> dict[str, float]:
    """
    Sum the final_score of every chunk that mentions a given topic.

    Args:
        scored_chunks: List of chunk dicts with ``"chunk_topics"`` and ``"final_score"``.

    Returns:
        Dict mapping topic → cumulative score.
    """
    topic_scores: dict[str, float] = defaultdict(float)
    for chunk in scored_chunks:
        score = chunk.get("final_score", 0.0)
        for topic in chunk.get("chunk_topics", []):
            topic_scores[topic] += score
    return dict(topic_scores)


# ─────────────────────────────────────────────────────────────────────────────
# IDF weighting
# ─────────────────────────────────────────────────────────────────────────────

def _compute_idf_weighted_scores(
    scored_chunks: list[dict],
    raw_scores: dict[str, float],
) -> dict[str, float]:
    """
    Re-weight cumulative topic scores by an Inverse Document Frequency (IDF)
    factor to penalise topics that appear in nearly every chunk.

    Formula:
        idf(topic) = log( total_chunks / (chunks_containing_topic + 1) ) + 1
        idf_score  = raw_score * idf(topic)

    A topic appearing in only 2 of 20 chunks but with high chunk scores will
    rank above a topic appearing in 18 of 20 chunks with mediocre scores.

    Args:
        scored_chunks: Full list of scored chunk dicts.
        raw_scores:    Dict of topic → cumulative raw score.

    Returns:
        Dict of topic → IDF-weighted score.
    """
    total_chunks = max(len(scored_chunks), 1)

    # Count how many chunks contain each topic
    doc_freq: dict[str, int] = defaultdict(int)
    for chunk in scored_chunks:
        seen_in_chunk = set(chunk.get("chunk_topics", []))
        for topic in seen_in_chunk:
            doc_freq[topic] += 1

    idf_weighted: dict[str, float] = {}
    for topic, score in raw_scores.items():
        df = doc_freq.get(topic, 0)
        idf = math.log(total_chunks / (df + 1)) + 1.0
        idf_weighted[topic] = score * idf

    return idf_weighted


# ─────────────────────────────────────────────────────────────────────────────
# Ranking with cross-source boosts
# ─────────────────────────────────────────────────────────────────────────────

def rank_topics(
    scored_chunks: list[dict],
    ocr_results: list[dict],
    top_n: int = 10,
) -> list[dict]:
    """
    Produce a ranked list of predicted exam topics with confidence scores.

    Ranking pipeline (v2 — IDF-weighted):
      1. Aggregate raw cumulative scores per topic.
      2. Apply IDF weighting to penalise overly common topics.
      3. Build reference sets from exam papers and assignments.
      4. Apply source boosts (exam ×1.5, assignment ×1.2; stack multiplicatively).
      5. Normalise all scores to [0.0, 1.0].
      6. Return top *top_n* sorted by descending confidence.

    Args:
        scored_chunks: Enriched chunk list from ``scorer.score_all_chunks``.
        ocr_results:   Processed OCR source dicts from ``ocr_pipeline.run_ocr_pipeline``.
        top_n:         Maximum topics to return (default 10).

    Returns:
        List of up to *top_n* dicts:
          - ``topic``                  (str)
          - ``confidence``             (float 0-1)
          - ``appeared_in_exam``       (bool)
          - ``appeared_in_assignment`` (bool)
        Sorted by descending confidence.
    """
    raw_scores = aggregate_topic_scores(scored_chunks)

    if not raw_scores:
        return []

    # ── IDF weighting ─────────────────────────────────────────────────────
    idf_scores = _compute_idf_weighted_scores(scored_chunks, raw_scores)

    # ── Build reference sets ──────────────────────────────────────────────
    exam_topic_set: set[str] = set()
    assignment_topic_set: set[str] = set()

    for item in ocr_results:
        src = item.get("source_type", "")
        topics = item.get("topics", [])
        if src == "exam_paper":
            exam_topic_set.update(topics)
        elif src == "assignment":
            assignment_topic_set.update(topics)

    # ── Apply source boosts ───────────────────────────────────────────────
    boosted: dict[str, float] = {}
    for topic, score in idf_scores.items():
        boosted_score = score
        if topic in exam_topic_set:
            boosted_score *= config.EXAM_PAPER_BOOST
        if topic in assignment_topic_set:
            boosted_score *= config.ASSIGNMENT_BOOST
        boosted[topic] = boosted_score

    # ── Normalise ─────────────────────────────────────────────────────────
    min_s = min(boosted.values()) if boosted else 0.0
    max_s = max(boosted.values()) if boosted else 0.0

    ranked: list[dict] = []
    for topic, score in boosted.items():
        confidence = utils.normalize_score(score, min_s, max_s)
        ranked.append({
            "topic": topic,
            "confidence": round(confidence, 4),
            "appeared_in_exam": topic in exam_topic_set,
            "appeared_in_assignment": topic in assignment_topic_set,
        })

    # Sort: descending confidence, then alphabetical for deterministic ties
    ranked.sort(key=lambda x: (-x["confidence"], x["topic"]))
    result = ranked[:top_n]

    print(f"[topic_ranker] Ranked {len(raw_scores)} unique topics → returning top {len(result)}.")
    return result
