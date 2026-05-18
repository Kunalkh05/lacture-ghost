"""
topic_ranker.py — Aggregates topics across all sources and ranks the top N by
predicted exam likelihood.

Applies source-specific boosts:
  - Topics appearing in past exam papers → ×1.5
  - Topics appearing in assignments       → ×1.2
"""

from __future__ import annotations

from collections import defaultdict

import config
import utils


# ─────────────────────────────────────────────────────────────────────────────
# Score aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_topic_scores(scored_chunks: list[dict]) -> dict[str, float]:
    """
    Sum the final_score of every chunk that mentions a given topic.

    A topic's cumulative score reflects both how frequently it appears across
    the lecture and how important (by combined signals) those appearances were.

    Args:
        scored_chunks: List of chunk dicts produced by ``scorer.score_all_chunks``.
                       Each dict must contain ``"chunk_topics"`` (list[str]) and
                       ``"final_score"`` (float).

    Returns:
        A dict mapping each unique topic string to its cumulative score.
        Returns an empty dict if *scored_chunks* is empty.
    """
    topic_scores: dict[str, float] = defaultdict(float)

    for chunk in scored_chunks:
        chunk_score = chunk.get("final_score", 0.0)
        for topic in chunk.get("chunk_topics", []):
            topic_scores[topic] += chunk_score

    return dict(topic_scores)


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

    Ranking pipeline:
      1. Aggregate raw cumulative scores per topic from ``scored_chunks``.
      2. Build reference sets of topics found in exam papers and assignments.
      3. Apply boosts:
           - Exam paper match  → raw_score × 1.5
           - Assignment match  → raw_score × 1.2
           - Both              → boosts stack multiplicatively
      4. Normalise all boosted scores to [0.0, 1.0].
      5. Return the top *top_n* topics sorted by descending confidence.

    Args:
        scored_chunks: Enriched chunk list from ``scorer.score_all_chunks``.
        ocr_results:   List of processed OCR source dicts from
                       ``ocr_pipeline.run_ocr_pipeline``.
        top_n:         Maximum number of topics to return (default 10).

    Returns:
        A list of up to *top_n* dicts each containing:
          - ``topic``                (str)
          - ``confidence``           (float, 0-1)
          - ``appeared_in_exam``     (bool)
          - ``appeared_in_assignment`` (bool)

        Sorted by ``confidence`` descending.
    """
    raw_scores = aggregate_topic_scores(scored_chunks)

    if not raw_scores:
        return []

    # ── Build exam / assignment reference sets from OCR results ──────────
    exam_topic_set: set[str] = set()
    assignment_topic_set: set[str] = set()

    for ocr_item in ocr_results:
        src_type = ocr_item.get("source_type", "")
        topics = ocr_item.get("topics", [])
        if src_type == "exam_paper":
            exam_topic_set.update(topics)
        elif src_type == "assignment":
            assignment_topic_set.update(topics)

    # ── Apply boosts ──────────────────────────────────────────────────────
    boosted: dict[str, float] = {}
    for topic, score in raw_scores.items():
        boosted_score = score
        if topic in exam_topic_set:
            boosted_score *= config.EXAM_PAPER_BOOST
        if topic in assignment_topic_set:
            boosted_score *= config.ASSIGNMENT_BOOST
        boosted[topic] = boosted_score

    # ── Normalise to [0, 1] ───────────────────────────────────────────────
    if boosted:
        min_s = min(boosted.values())
        max_s = max(boosted.values())
    else:
        min_s = max_s = 0.0

    ranked: list[dict] = []
    for topic, score in boosted.items():
        confidence = utils.normalize_score(score, min_s, max_s)
        ranked.append(
            {
                "topic": topic,
                "confidence": round(confidence, 4),
                "appeared_in_exam": topic in exam_topic_set,
                "appeared_in_assignment": topic in assignment_topic_set,
            }
        )

    # Sort descending by confidence, alphabetically for ties
    ranked.sort(key=lambda x: (-x["confidence"], x["topic"]))

    result = ranked[:top_n]
    print(f"[topic_ranker] Ranked {len(raw_scores)} unique topics → returning top {len(result)}.")
    return result
