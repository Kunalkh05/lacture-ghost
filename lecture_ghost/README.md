# Lecture Ghost

**Lecture Ghost** is a multimodal AI system that predicts which topics your professor is most likely to test in the next exam. It analyses lecture audio recordings, slide images, past exam papers, and assignment PDFs simultaneously, combines behavioural and content signals using a weighted scoring model, and delivers a ranked list of predicted exam topics with confidence scores and a visual heatmap timeline.

---

## Prerequisites

- **Python 3.10+**
- **Tesseract OCR** binary installed on your operating system
- **Poppler** (required by pdf2image for PDF → image conversion)

### Install Tesseract

**macOS (Homebrew)**
```bash
brew install tesseract
brew install poppler
```

**Windows**
1. Download the installer from https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer and note the installation path (e.g. `C:\Program Files\Tesseract-OCR`)
3. Add that path to your system `PATH` environment variable
4. Install poppler for Windows: https://github.com/oschwartz10612/poppler-windows/releases
5. Add poppler's `bin/` folder to `PATH`

---

## Installation

```bash
# 1. Clone / download the project
cd "lacture ghost/lecture_ghost"

# 2. (Recommended) Create a virtual environment
python -m venv .venv
# Mac/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download the spaCy English language model
python -m spacy download en_core_web_sm
```

---

## Running the App

```bash
streamlit run main.py
```

The app opens automatically at `http://localhost:8501`.

---

## How to Use

1. **Upload a lecture recording** (`.mp3`, `.wav`, or `.m4a`) in the sidebar — this is the only required file.
2. Optionally upload **lecture slides**, **past exam papers**, and **assignments** (`.jpg`, `.png`, or `.pdf`).
3. Click **Analyze**.
4. Watch the four-step progress bar as the system:
   - Transcribes the audio with Whisper
   - Runs OCR on all images and PDFs
   - Scores each 30-second lecture chunk
   - Ranks predicted exam topics
5. Explore the **heatmap timeline**, **topic table**, and **segment explorer**.
6. Download the full **JSON report**.

### Testing with a Sample File

```bash
# Generate a short silent MP3 for quick UI testing (requires ffmpeg)
ffmpeg -f lavfi -i anullsrc=r=44100:cl=mono -t 90 -q:a 9 -acodec libmp3lame sample.mp3
streamlit run main.py
# Upload sample.mp3 — you will see a "silent audio" warning (expected)
```

---

## Folder Structure

```
lecture_ghost/
├── main.py            # Streamlit UI — run this file
├── audio_pipeline.py  # Whisper transcription + pace/repetition/keyword signals
├── ocr_pipeline.py    # Tesseract OCR + spaCy topic extraction
├── scorer.py          # Cross-modal weighted importance scorer
├── topic_ranker.py    # Aggregates and ranks predicted exam topics
├── utils.py           # Shared helpers: chunking, cleaning, JSON I/O
├── config.py          # All constants and weights
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Known Limitations

| Limitation | Detail |
|---|---|
| Whisper speed | The `base` model is fast but less accurate on heavy accents. Switch to `small` or `medium` in `config.py` for better results. |
| OCR quality | Handwritten text and low-resolution scans may produce poor OCR results. Use 300 DPI+ scans. |
| spaCy topic quality | Noun-chunk extraction works best on typed, grammatically correct text. |
| Audio-only mode | Without exam papers or assignments, `overlap_score` is 0 for all chunks — the ranking still works but is less precise. |
| Tesseract on Windows | PATH must be correctly configured or pytesseract will raise an error. |
| PDF extraction | Scanned PDFs (image-only) are supported; text-layer PDFs are also converted via pdf2image for consistency. |
| Long recordings | Files over 2 hours may require significant RAM and processing time on the `base` Whisper model. |
