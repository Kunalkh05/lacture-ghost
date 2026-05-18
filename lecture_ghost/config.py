"""
config.py — Central configuration for Lecture Ghost.

All tunable constants, weights, and model identifiers live here.
No magic numbers should appear anywhere else in the codebase.
"""

# ─── Audio chunking ───────────────────────────────────────────────────────────
CHUNK_DURATION_SECONDS: int = 30
SLIDING_WINDOW_SECONDS: int = 5

# ─── Model identifiers ────────────────────────────────────────────────────────
WHISPER_MODEL: str = "base"
SPACY_MODEL: str = "en_core_web_sm"

# ─── Emphasis / exam-signal keywords ─────────────────────────────────────────
EMPHASIS_KEYWORDS: list[str] = [
    "important",
    "remember this",
    "this will come",
    "note this",
    "recall",
    "key concept",
    "this is crucial",
    "pay attention",
    "you should know",
    "this is significant",
    "mark this",
    "expected in exam",
    "most likely",
    "focus on this",
    "this is the main",
    "don't forget",
    "make sure you know",
    "this appears every year",
    "classic question",
    "definition of",
]

# ─── Cross-modal scoring weights (must sum to 1.0) ────────────────────────────
SCORING_WEIGHTS: dict[str, float] = {
    "pace_score": 0.25,
    "repetition_score": 0.25,
    "keyword_score": 0.20,
    "overlap_score": 0.30,
}

# ─── Heatmap colour palette ───────────────────────────────────────────────────
HEATMAP_COLORS: dict[str, str] = {
    "low": "#3B8BD4",
    "medium": "#EF9F27",
    "high": "#E24B4A",
}

# ─── Importance thresholds ────────────────────────────────────────────────────
HIGH_THRESHOLD: float = 0.65
MEDIUM_THRESHOLD: float = 0.35

# ─── Topic ranking boosts ─────────────────────────────────────────────────────
EXAM_PAPER_BOOST: float = 1.5
ASSIGNMENT_BOOST: float = 1.2

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_JSON_PATH: str = "lecture_ghost_output.json"
