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

# en_core_web_md has word vectors (300-dim) — required for semantic similarity.
# Install: python -m spacy download en_core_web_md
SPACY_MODEL: str = "en_core_web_md"

# ─── OCR preprocessing ────────────────────────────────────────────────────────
# Upscale images narrower than this pixel width before running Tesseract
OCR_MIN_WIDTH_PX: int = 1000
OCR_UPSCALE_FACTOR: int = 2
# Contrast enhancement multiplier (1.0 = no change, 1.5 = moderate boost)
OCR_CONTRAST_FACTOR: float = 1.4

# ─── Topic extraction ─────────────────────────────────────────────────────────
TFIDF_TOP_N_TOPICS: int = 20
# Minimum character length for a noun chunk to be kept
TOPIC_MIN_CHARS: int = 3

# ─── Semantic similarity threshold for cross-modal overlap ───────────────────
# Two topics are considered a "match" when spaCy similarity >= this value.
# Range 0.0–1.0; 0.65 balances precision vs recall well.
SEMANTIC_SIMILARITY_THRESHOLD: float = 0.65

# ─── Weighted emphasis / exam-signal keywords ─────────────────────────────────
# Weight 1.0 = definitive exam signal; 0.4 = weak signal.
# Weighted sum is normalised in audio_pipeline.compute_keyword_scores.
EMPHASIS_KEYWORD_WEIGHTS: dict[str, float] = {
    # Strong signals (weight 1.0)
    "this will be on the exam": 1.0,
    "expected in exam": 1.0,
    "this appears every year": 1.0,
    "classic question": 1.0,
    "this is crucial": 1.0,
    "make sure you know": 1.0,
    # Medium signals (weight 0.7)
    "important": 0.7,
    "key concept": 0.7,
    "this is significant": 0.7,
    "focus on this": 0.7,
    "most likely": 0.7,
    "definition of": 0.7,
    "this is the main": 0.7,
    # Weak signals (weight 0.4)
    "remember this": 0.4,
    "note this": 0.4,
    "recall": 0.4,
    "pay attention": 0.4,
    "you should know": 0.4,
    "mark this": 0.4,
    "don't forget": 0.4,
}

# Flat list for backward-compatible uses (e.g., display in UI)
EMPHASIS_KEYWORDS: list[str] = list(EMPHASIS_KEYWORD_WEIGHTS.keys())

# Denominator used to normalise weighted keyword sum to 0-1.
# Value of 2.0 means "2 full-weight keywords = max score".
KEYWORD_SCORE_NORMALISER: float = 2.0

# ─── Cross-modal scoring weights (must sum to 1.0) ────────────────────────────
SCORING_WEIGHTS: dict[str, float] = {
    "pace_score": 0.20,
    "repetition_score": 0.25,
    "keyword_score": 0.25,
    "overlap_score": 0.30,
}

# ─── Importance thresholds ────────────────────────────────────────────────────
HIGH_THRESHOLD: float = 0.65
MEDIUM_THRESHOLD: float = 0.35

# ─── Topic ranking ────────────────────────────────────────────────────────────
EXAM_PAPER_BOOST: float = 1.5
ASSIGNMENT_BOOST: float = 1.2

# ─── Heatmap colour palette ───────────────────────────────────────────────────
HEATMAP_COLORS: dict[str, str] = {
    "low": "#3B8BD4",
    "medium": "#EF9F27",
    "high": "#E24B4A",
}

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_JSON_PATH: str = "lecture_ghost_output.json"
