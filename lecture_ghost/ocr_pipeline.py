"""
ocr_pipeline.py — OCR extraction from images / PDFs + spaCy topic mining.

Handles three source types:
  - "slide"       : Lecture slide images / PDF pages
  - "exam_paper"  : Past exam paper photographs
  - "assignment"  : Assignment PDFs or scanned images
"""

from __future__ import annotations

import warnings
from pathlib import Path

import config
import utils


# ─────────────────────────────────────────────────────────────────────────────
# Image OCR
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from a single image file using Tesseract OCR.

    Pre-processing steps applied to improve OCR accuracy:
      1. Open with Pillow and convert to grayscale.
      2. Apply a binary threshold (point-function) to increase contrast.
      3. Pass the processed image to pytesseract.

    Args:
        image_path: Absolute or relative path to an image file (.jpg, .png, etc.).

    Returns:
        The extracted text as a cleaned string.  Returns an empty string if
        the image is corrupt or cannot be read — a warning is printed instead
        of raising an exception so batch processing can continue.
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError as exc:
        raise ImportError(
            "[ocr_pipeline] Required library missing. "
            "Run: pip install Pillow pytesseract"
        ) from exc

    try:
        img = Image.open(image_path).convert("L")          # grayscale
        img = img.point(lambda px: 0 if px < 140 else 255, "1")  # threshold
        raw_text: str = pytesseract.image_to_string(img)
        return utils.clean_text(raw_text)
    except Exception as exc:
        warnings.warn(
            f"[ocr_pipeline] Could not extract text from '{image_path}': {exc}",
            UserWarning,
            stacklevel=2,
        )
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# PDF OCR  (converts each page to image first)
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """
    Extract text from every page of a PDF by converting pages to images first.

    Uses pdf2image (which wraps poppler) to render each page, then passes
    each rendered image through ``extract_text_from_image``.

    Args:
        pdf_path: Absolute or relative path to a PDF file.

    Returns:
        A list of strings — one entry per page.  Pages that fail OCR contribute
        an empty string so the list length always equals the page count.

    Raises:
        ImportError: If pdf2image is not installed.
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
            f"[ocr_pipeline] Failed to convert PDF '{source}' to images: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return []

    page_texts: list[str] = []
    for page_idx, page_img in enumerate(pages):
        try:
            from PIL import Image
            import pytesseract

            gray = page_img.convert("L")
            gray = gray.point(lambda px: 0 if px < 140 else 255, "1")
            raw = pytesseract.image_to_string(gray)
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
            f"[ocr_pipeline] No text could be extracted from any page of '{source}'.",
            UserWarning,
            stacklevel=2,
        )

    return page_texts


# ─────────────────────────────────────────────────────────────────────────────
# Topic extraction (spaCy noun chunks)
# ─────────────────────────────────────────────────────────────────────────────

def extract_topics_from_text(text: str) -> list[str]:
    """
    Extract the most frequent noun-chunk topics from text using spaCy.

    Filter criteria for accepted noun chunks:
      - Longer than 2 characters.
      - Not purely numeric.
      - Root token is not a stop-word.

    Returns the top-20 unique topics sorted by descending frequency.

    Args:
        text: Raw or pre-cleaned text string.

    Returns:
        A list of up to 20 lowercase topic strings.

    Raises:
        OSError: If the spaCy model is not installed — prints the download
                 command and returns an empty list rather than crashing.
    """
    if not text.strip():
        return []

    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "[ocr_pipeline] spaCy is not installed. Run: pip install spacy"
        ) from exc

    try:
        nlp = spacy.load(config.SPACY_MODEL)
    except OSError:
        print(
            f"[ocr_pipeline] spaCy model '{config.SPACY_MODEL}' not found.\n"
            f"Install it with:  python -m spacy download {config.SPACY_MODEL}"
        )
        return []

    # spaCy works best on reasonably-sized text; truncate if needed
    doc = nlp(text[:50_000])

    from collections import Counter

    freq: Counter = Counter()
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        # Filter: length, not purely numeric, not a pure stop-word phrase
        if (
            len(phrase) > 2
            and not phrase.replace(" ", "").isnumeric()
            and not chunk.root.is_stop
        ):
            freq[phrase] += 1

    # Return top 20 by frequency, deduplicated
    top_topics = [topic for topic, _ in freq.most_common(20)]
    return top_topics


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
          - ``topics``      (list[str]): top topics found

    The function never raises — errors are captured and logged so that batch
    processing of many files can continue even if one is problematic.
    """
    source = Path(file_path)
    suffix = source.suffix.lower()

    raw_text: str = ""

    if suffix == ".pdf":
        page_texts = extract_text_from_pdf(file_path)
        raw_text = " ".join(page_texts)
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        raw_text = extract_text_from_image(file_path)
    else:
        warnings.warn(
            f"[ocr_pipeline] Unsupported file type '{suffix}' for '{source}' — skipping.",
            UserWarning,
            stacklevel=2,
        )

    topics = extract_topics_from_text(raw_text) if raw_text.strip() else []

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
    Run OCR and topic extraction over a list of files.

    Args:
        file_list: A list of dicts each with ``"path"`` and ``"source_type"`` keys.
                   ``source_type`` should be ``"slide"``, ``"exam_paper"``, or
                   ``"assignment"``.

    Returns:
        A list of processed source dicts (same structure as ``process_ocr_source``
        output), one per input file.

    Notes:
        If *file_list* is empty the function returns immediately with an empty list,
        and the audio-only mode in ``scorer.py`` handles the missing OCR data.
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
