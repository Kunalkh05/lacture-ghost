"""
ocr_pipeline.py — OCR extraction from images / PDFs + spaCy topic mining.

Accuracy improvements over v1:
  - 4-step adaptive image preprocessing (upscale → grayscale → contrast → denoise)
  - Dual Tesseract PSM modes — picks the result with more content
  - TF-IDF topic ranking (scikit-learn) instead of raw frequency counts
  - spaCy Named Entity Recognition merged with noun-chunk extraction

Handles three source types:
  - "slide"       : Lecture slide images / PDF pages
  - "exam_paper"  : Past exam paper photographs
  - "assignment"  : Assignment PDFs or scanned images
"""

from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Optional

import config
import utils


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_image(img: "PIL.Image.Image") -> "PIL.Image.Image":
    """
    Apply a 4-step pipeline to maximise Tesseract OCR accuracy.

    Steps:
      1. Upscale if the image is too narrow (Lanczos resampling).
      2. Convert to grayscale (L mode).
      3. Boost contrast via ImageEnhance.Contrast.
      4. Median-filter denoising to remove scanner noise / compression artifacts.

    Args:
        img: An open Pillow Image in any mode.

    Returns:
        A preprocessed grayscale Pillow Image ready for Tesseract.
    """
    from PIL import Image, ImageEnhance, ImageFilter

    # Step 1 — upscale narrow images
    w, h = img.size
    if w < config.OCR_MIN_WIDTH_PX:
        scale = config.OCR_UPSCALE_FACTOR
        img = img.resize((w * scale, h * scale), Image.LANCZOS)

    # Step 2 — grayscale
    img = img.convert("L")

    # Step 3 — contrast enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(config.OCR_CONTRAST_FACTOR)

    # Step 4 — median filter denoising
    img = img.filter(ImageFilter.MedianFilter(size=3))

    return img


def _run_tesseract(img: "PIL.Image.Image") -> str:
    """
    Run Tesseract with two PSM modes and return the longer (richer) result.

    PSM 3  = fully automatic page segmentation (best for mixed layouts).
    PSM 6  = assume a single uniform block of text (best for slides).

    Args:
        img: A preprocessed grayscale Pillow Image.

    Returns:
        The OCR text string with the most content across both PSM modes.
    """
    import pytesseract

    config_psm3 = "--oem 3 --psm 3"
    config_psm6 = "--oem 3 --psm 6"

    text_psm3 = pytesseract.image_to_string(img, config=config_psm3)
    text_psm6 = pytesseract.image_to_string(img, config=config_psm6)

    # Return whichever mode produced more text content
    return text_psm3 if len(text_psm3) >= len(text_psm6) else text_psm6


# ─────────────────────────────────────────────────────────────────────────────
# Image OCR
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from a single image file using adaptive preprocessing + Tesseract.

    Args:
        image_path: Absolute or relative path to an image file (.jpg, .png, etc.).

    Returns:
        The extracted text as a cleaned string.  Returns an empty string if
        the image is corrupt or unreadable — a warning is printed so batch
        processing can continue uninterrupted.
    """
    try:
        from PIL import Image
        import pytesseract  # noqa: F401 — validates install
    except ImportError as exc:
        raise ImportError(
            "[ocr_pipeline] Required library missing. "
            "Run: pip install Pillow pytesseract"
        ) from exc

    try:
        img = Image.open(image_path)
        img = _preprocess_image(img)
        raw_text = _run_tesseract(img)
        return utils.clean_text(raw_text)
    except Exception as exc:
        warnings.warn(
            f"[ocr_pipeline] Could not extract text from '{image_path}': {exc}",
            UserWarning,
            stacklevel=2,
        )
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# PDF OCR
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """
    Extract text from every page of a PDF by rendering pages as images first.

    Uses pdf2image (wraps poppler) to render at 200 DPI, then applies the
    full adaptive preprocessing pipeline to each page image.

    Args:
        pdf_path: Absolute or relative path to a PDF file.

    Returns:
        A list of cleaned strings — one entry per page.  Pages that fail OCR
        contribute an empty string so the list length equals the page count.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ImportError(
            "[ocr_pipeline] pdf2image is not installed. Run: pip install pdf2image\n"
            "Also ensure poppler is installed on your OS."
        ) from exc

    source = Path(pdf_path)
    if not source.exists():
        warnings.warn(
            f"[ocr_pipeline] PDF not found: {source} — skipping.",
            UserWarning,
            stacklevel=2,
        )
        return []

    try:
        pages = convert_from_path(str(source), dpi=200)
    except Exception as exc:
        warnings.warn(
            f"[ocr_pipeline] Failed to render PDF '{source}': {exc}",
            UserWarning,
            stacklevel=2,
        )
        return []

    page_texts: list[str] = []
    for page_idx, page_img in enumerate(pages):
        try:
            preprocessed = _preprocess_image(page_img)
            raw = _run_tesseract(preprocessed)
            page_texts.append(utils.clean_text(raw))
        except Exception as exc:
            warnings.warn(
                f"[ocr_pipeline] OCR failed on page {page_idx + 1} of '{source}': {exc}",
                UserWarning,
                stacklevel=2,
            )
            page_texts.append("")

    if not any(page_texts):
        warnings.warn(
            f"[ocr_pipeline] No text extracted from any page of '{source}'.",
            UserWarning,
            stacklevel=2,
        )
    return page_texts


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF + NER topic extraction
# ─────────────────────────────────────────────────────────────────────────────

def _tfidf_rank_phrases(phrases: list[str], corpus_texts: list[str]) -> list[str]:
    """
    Rank candidate phrases by TF-IDF score within a text corpus.

    A phrase's TF-IDF score reflects how discriminative it is within the
    corpus — frequent across all documents → low score; specific to a few
    documents → high score.

    Args:
        phrases:      Candidate phrase strings to rank.
        corpus_texts: List of document strings forming the TF-IDF corpus.
                      Usually the pages / chunks from a single source.

    Returns:
        The *phrases* list sorted by descending TF-IDF relevance.
        Falls back to the original order if sklearn is unavailable.
    """
    if not phrases or not corpus_texts:
        return phrases

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        warnings.warn(
            "[ocr_pipeline] scikit-learn not installed — falling back to frequency ranking.\n"
            "Run: pip install scikit-learn",
            UserWarning,
            stacklevel=3,
        )
        return phrases

    try:
        # Build vocabulary from candidate phrases only
        vocab = list(set(phrases))
        vectorizer = TfidfVectorizer(
            vocabulary=vocab,
            ngram_range=(1, 3),
            analyzer="word",
            lowercase=True,
            sublinear_tf=True,  # log(1+tf) smoothing
        )
        tfidf_matrix = vectorizer.fit_transform(corpus_texts)
        # Sum scores across all documents to get a single importance value per phrase
        scores = tfidf_matrix.sum(axis=0).A1  # shape: (vocab_size,)
        phrase_scores = dict(zip(vocab, scores))
        return sorted(phrases, key=lambda p: phrase_scores.get(p, 0.0), reverse=True)
    except Exception as exc:
        warnings.warn(f"[ocr_pipeline] TF-IDF ranking failed: {exc} — using frequency order.", stacklevel=3)
        return phrases


def extract_topics_from_text(
    text: str,
    corpus_texts: Optional[list[str]] = None,
) -> list[str]:
    """
    Extract key topics from text using spaCy NLP (noun chunks + NER) ranked by TF-IDF.

    Pipeline:
      1. spaCy noun-chunk extraction (filtered by length, stopwords, numerics).
      2. spaCy Named Entity Recognition for PERSON, ORG, EVENT, LAW, etc.
      3. Merge + deduplicate both sets.
      4. Re-rank by TF-IDF score within *corpus_texts* (defaults to [text]).
      5. Return top ``config.TFIDF_TOP_N_TOPICS`` topics.

    Args:
        text:          Raw or pre-cleaned text to analyse.
        corpus_texts:  Optional list of related documents to build the TF-IDF
                       corpus from.  Defaults to ``[text]`` when not provided.

    Returns:
        A list of up to ``TFIDF_TOP_N_TOPICS`` lowercase topic strings.
    """
    if not text.strip():
        return []

    try:
        import spacy
    except ImportError:
        warnings.warn(
            "[ocr_pipeline] spaCy is not installed — topic extraction skipped.\n"
            "Run: pip install spacy && python -m spacy download en_core_web_md",
            UserWarning,
            stacklevel=2,
        )
        return []

    try:
        nlp = spacy.load(config.SPACY_MODEL)
    except OSError:
        # Graceful fallback: try the smaller model before giving up entirely
        print(
            f"[ocr_pipeline] spaCy model '{config.SPACY_MODEL}' not found.\n"
            f"Install it with:  python -m spacy download {config.SPACY_MODEL}\n"
            "Trying en_core_web_sm as fallback…"
        )
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print(
                "[ocr_pipeline] en_core_web_sm also not found.\n"
                "Run: python -m spacy download en_core_web_sm"
            )
            return []

    doc = nlp(text[:50_000])

    # ── Noun chunks ───────────────────────────────────────────────────────
    candidates: list[str] = []
    freq: Counter = Counter()

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        if (
            len(phrase) > config.TOPIC_MIN_CHARS
            and not phrase.replace(" ", "").isnumeric()
            and not chunk.root.is_stop
        ):
            candidates.append(phrase)
            freq[phrase] += 1

    # ── Named entities (adds precision for proper nouns / concepts) ───────
    relevant_labels = {"ORG", "PERSON", "GPE", "EVENT", "LAW", "WORK_OF_ART", "PRODUCT", "NORP"}
    for ent in doc.ents:
        phrase = ent.text.strip().lower()
        if ent.label_ in relevant_labels and len(phrase) > config.TOPIC_MIN_CHARS:
            candidates.append(phrase)
            freq[phrase] += 1

    # Deduplicate while preserving frequency info
    unique_candidates = list(dict.fromkeys(candidates))

    if not unique_candidates:
        return []

    # ── TF-IDF re-ranking ─────────────────────────────────────────────────
    corpus = corpus_texts if corpus_texts else [text]
    ranked = _tfidf_rank_phrases(unique_candidates, corpus)

    return ranked[: config.TFIDF_TOP_N_TOPICS]


# ─────────────────────────────────────────────────────────────────────────────
# Single-source processor
# ─────────────────────────────────────────────────────────────────────────────

def process_ocr_source(file_path: str, source_type: str) -> dict:
    """
    Extract text and topics from a single file (image or PDF).

    Args:
        file_path:   Path to the file to process.
        source_type: One of ``"slide"``, ``"exam_paper"``, or ``"assignment"``.

    Returns:
        A dict with keys:
          - ``file_path``   (str)
          - ``source_type`` (str)
          - ``raw_text``    (str): concatenated extracted text
          - ``topics``      (list[str]): top TF-IDF ranked topics

    Never raises — errors are captured and logged so batch processing continues.
    """
    source = Path(file_path)
    suffix = source.suffix.lower()

    raw_text: str = ""
    page_texts: list[str] = []

    if suffix == ".pdf":
        page_texts = extract_text_from_pdf(file_path)
        raw_text = " ".join(page_texts)
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        raw_text = extract_text_from_image(file_path)
        page_texts = [raw_text]
    else:
        warnings.warn(
            f"[ocr_pipeline] Unsupported file type '{suffix}' for '{source}' — skipping.",
            UserWarning,
            stacklevel=2,
        )

    # Pass all pages as the TF-IDF corpus so topic ranking is cross-page aware
    topics = (
        extract_topics_from_text(raw_text, corpus_texts=page_texts if page_texts else [raw_text])
        if raw_text.strip()
        else []
    )

    return {
        "file_path": str(source),
        "source_type": source_type,
        "raw_text": raw_text,
        "topics": topics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_ocr_pipeline(file_list: list[dict]) -> list[dict]:
    """
    Run adaptive OCR and TF-IDF topic extraction over a list of files.

    Args:
        file_list: A list of dicts each with ``"path"`` and ``"source_type"`` keys.

    Returns:
        A list of processed source dicts, one per input file.
        Returns an empty list immediately if *file_list* is empty.
    """
    if not file_list:
        print("[ocr_pipeline] No OCR files provided — running in audio-only mode.")
        return []

    results: list[dict] = []
    for entry in file_list:
        path = entry.get("path", "")
        source_type = entry.get("source_type", "slide")

        if not path:
            warnings.warn("[ocr_pipeline] Empty path in file_list entry — skipping.", stacklevel=2)
            continue

        print(f"[ocr_pipeline] Processing {source_type}: {Path(path).name} …")
        result = process_ocr_source(path, source_type)
        results.append(result)

    print(f"[ocr_pipeline] Processed {len(results)} source file(s).")
    return results
