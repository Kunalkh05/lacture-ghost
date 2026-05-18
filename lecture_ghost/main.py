"""
main.py — Streamlit UI entry point for Lecture Ghost.

Page layout (in order):
  Sidebar  : File uploaders + Analyze button
  Section 1: Header + progress status
  Section 2: Heatmap timeline
  Section 3: Top predicted exam topics table
  Section 4: Chunk detail explorer
  Section 5: JSON download
"""

from __future__ import annotations

import datetime
import json
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import config
import audio_pipeline
import ocr_pipeline
import scorer
import topic_ranker
import utils

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Lecture Ghost — Exam Prediction System",
    page_icon="👻",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-high {
        background: #E24B4A22; color: #E24B4A;
        border: 1px solid #E24B4A55;
        border-radius: 6px; padding: 2px 10px;
        font-size: 0.82rem; font-weight: 600;
    }
    .badge-medium {
        background: #EF9F2722; color: #EF9F27;
        border: 1px solid #EF9F2755;
        border-radius: 6px; padding: 2px 10px;
        font-size: 0.82rem; font-weight: 600;
    }
    .badge-low {
        background: #3B8BD422; color: #3B8BD4;
        border: 1px solid #3B8BD455;
        border-radius: 6px; padding: 2px 10px;
        font-size: 0.82rem; font-weight: 600;
    }
    .stProgress > div > div { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_upload(uploaded_file, tmp_dir: str) -> str:
    """Write a Streamlit UploadedFile to *tmp_dir* and return the path."""
    dest = Path(tmp_dir) / uploaded_file.name
    dest.write_bytes(uploaded_file.read())
    return str(dest)


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _build_explanation(chunk: dict) -> str:
    """Build a plain-English reason string for why a chunk was flagged."""
    reasons = []
    if chunk.get("pace_score", 0) > 0.6:
        reasons.append("🐢 Speech slowed significantly")
    if chunk.get("repetition_score", 0) > 0.5:
        reasons.append("🔁 Vocabulary repeated from previous segment")
    if chunk.get("keyword_score", 0) > 0:
        text_lower = chunk.get("text", "").lower()
        matched = [kw for kw in config.EMPHASIS_KEYWORDS if kw in text_lower]
        if matched:
            reasons.append(f"🔑 Emphasis phrase detected: \"{matched[0]}\"")
    if chunk.get("overlap_score", 0) > 0:
        n = len(chunk.get("chunk_topics", []))
        reasons.append(f"📚 {n} topic(s) overlap with past exams / assignments")
    if not reasons:
        reasons.append("ℹ️ No strong individual signal — composite score is low")
    return " · ".join(reasons)


def _build_heatmap_html(chunks: list[dict]) -> str:
    """Render an HTML heatmap timeline from scored chunks."""
    if not chunks:
        return "<p style='color:#94a3b8'>No chunks to display.</p>"

    total_duration = max(c["end"] for c in chunks) or 1.0
    bars = []
    for c in chunks:
        width_pct = ((c["end"] - c["start"]) / total_duration) * 100
        color = config.HEATMAP_COLORS[c.get("importance_label", "low")]
        start_fmt = _fmt_time(c["start"])
        end_fmt = _fmt_time(c["end"])
        score = round(c.get("final_score", 0), 3)
        topics_preview = ", ".join(c.get("chunk_topics", [])[:3]) or "—"
        tooltip = f"{start_fmt}–{end_fmt} | Score: {score} | Topics: {topics_preview}"
        bars.append(
            f'<div title="{tooltip}" style="display:inline-block;width:{width_pct:.2f}%;'
            f'height:52px;background:{color};border-radius:4px;margin-right:1px;'
            f'cursor:pointer;transition:opacity 0.2s;" '
            f'onmouseover="this.style.opacity=\'0.75\'" '
            f'onmouseout="this.style.opacity=\'1\'"></div>'
        )

    legend = (
        '<div style="margin-top:12px;display:flex;gap:20px;font-size:0.82rem;color:#94a3b8">'
        f'<span><span style="display:inline-block;width:12px;height:12px;background:{config.HEATMAP_COLORS["low"]};border-radius:2px;margin-right:4px"></span>Low</span>'
        f'<span><span style="display:inline-block;width:12px;height:12px;background:{config.HEATMAP_COLORS["medium"]};border-radius:2px;margin-right:4px"></span>Medium</span>'
        f'<span><span style="display:inline-block;width:12px;height:12px;background:{config.HEATMAP_COLORS["high"]};border-radius:2px;margin-right:4px"></span>High importance</span>'
        "</div>"
    )
    return (
        '<div style="background:#0f172a;padding:16px;border-radius:10px;'
        'border:1px solid #1e293b;overflow-x:auto">'
        + "".join(bars)
        + legend
        + "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — file uploaders
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👻 Lecture Ghost")
    st.markdown("---")
    st.header("Upload your files")

    audio_file = st.file_uploader(
        "Lecture recording",
        type=["mp3", "wav", "m4a"],
        help="Your lecture audio (MP3, WAV, or M4A)",
    )
    slide_files = st.file_uploader(
        "Lecture slides",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        help="Slide images or PDF",
    )
    exam_files = st.file_uploader(
        "Past exam papers",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        help="Photographs or scans of past exams",
    )
    assignment_files = st.file_uploader(
        "Assignments",
        type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
        help="Assignment PDFs or images",
    )

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
    st.caption("Audio is required. Other sources are optional.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">👻 Lecture Ghost</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Multimodal AI that predicts which topics your professor '
    "will test — powered by Whisper, OCR, and cross-modal scoring.</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS TRIGGER
# ─────────────────────────────────────────────────────────────────────────────
if analyze_btn:
    if not audio_file:
        st.error("⚠️ Please upload a lecture recording before analyzing.")
        st.stop()

    tmp_dir = tempfile.mkdtemp()

    # Save all uploaded files to disk
    audio_path = _save_upload(audio_file, tmp_dir)

    ocr_file_list: list[dict] = []
    for f in (slide_files or []):
        ocr_file_list.append({"path": _save_upload(f, tmp_dir), "source_type": "slide"})
    for f in (exam_files or []):
        ocr_file_list.append({"path": _save_upload(f, tmp_dir), "source_type": "exam_paper"})
    for f in (assignment_files or []):
        ocr_file_list.append({"path": _save_upload(f, tmp_dir), "source_type": "assignment"})

    # ── Run pipelines with progress display ──────────────────────────────
    progress_bar = st.progress(0, text="Starting analysis…")

    with st.status("🔬 Analysing your lecture…", expanded=True) as status:
        # Step 1
        st.write("**Step 1/4:** Transcribing audio with Whisper…")
        progress_bar.progress(10, text="Step 1/4: Transcribing audio…")
        try:
            audio_chunks = audio_pipeline.run_audio_pipeline(audio_path)
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.stop()
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        if not audio_chunks:
            st.warning("The audio produced no usable transcript. Try a clearer recording.")
            st.stop()

        # Step 2
        st.write("**Step 2/4:** Extracting text from images and PDFs…")
        progress_bar.progress(40, text="Step 2/4: Running OCR…")
        ocr_results = ocr_pipeline.run_ocr_pipeline(ocr_file_list)

        # Step 3
        st.write("**Step 3/4:** Scoring lecture chunks…")
        progress_bar.progress(65, text="Step 3/4: Scoring chunks…")
        scored_chunks = scorer.score_all_chunks(audio_chunks, ocr_results)

        # Step 4
        st.write("**Step 4/4:** Ranking predicted exam topics…")
        progress_bar.progress(85, text="Step 4/4: Ranking topics…")
        ranked_topics = topic_ranker.rank_topics(scored_chunks, ocr_results, top_n=10)

        progress_bar.progress(100, text="✅ Analysis complete!")
        status.update(label="✅ Analysis complete!", state="complete", expanded=False)

    # ── Build report payload ──────────────────────────────────────────────
    report = {
        "metadata": {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "audio_file": audio_file.name,
            "slide_files": [f.name for f in (slide_files or [])],
            "exam_files": [f.name for f in (exam_files or [])],
            "assignment_files": [f.name for f in (assignment_files or [])],
            "whisper_model": config.WHISPER_MODEL,
            "spacy_model": config.SPACY_MODEL,
            "total_chunks": len(scored_chunks),
        },
        "ranked_topics": ranked_topics,
        "scored_chunks": [
            {k: v for k, v in c.items() if k != "words"} for c in scored_chunks
        ],
        "ocr_results": [
            {k: v for k, v in r.items() if k != "raw_text"} for r in ocr_results
        ],
    }

    st.session_state["results"] = report
    utils.save_json(report, config.OUTPUT_JSON_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS — rendered only when analysis is complete
# ─────────────────────────────────────────────────────────────────────────────
results = st.session_state.get("results")

if results:
    scored_chunks = results["scored_chunks"]
    ranked_topics = results["ranked_topics"]
    st.markdown("---")

    # ── Section 3: Heatmap ────────────────────────────────────────────────
    st.subheader("📊 Lecture importance timeline")
    heatmap_html = _build_heatmap_html(scored_chunks)
    components.html(heatmap_html, height=120, scrolling=False)

    # ── Section 4: Top predicted exam topics ─────────────────────────────
    st.subheader("🎯 Top predicted exam topics")
    if ranked_topics:
        col_headers = st.columns([0.5, 3, 2, 1.5, 1.5])
        col_headers[0].markdown("**Rank**")
        col_headers[1].markdown("**Topic**")
        col_headers[2].markdown("**Confidence**")
        col_headers[3].markdown("**In past exam?**")
        col_headers[4].markdown("**In assignment?**")
        st.markdown("---")

        for idx, topic_item in enumerate(ranked_topics, start=1):
            c0, c1, c2, c3, c4 = st.columns([0.5, 3, 2, 1.5, 1.5])
            c0.markdown(f"**#{idx}**")
            c1.markdown(f"**{topic_item['topic'].title()}**")
            c2.progress(float(topic_item["confidence"]), text=f"{topic_item['confidence']:.0%}")
            c3.markdown("✅" if topic_item["appeared_in_exam"] else "✗")
            c4.markdown("✅" if topic_item["appeared_in_assignment"] else "✗")
    else:
        st.info("No topics could be ranked — try uploading more source files.")

    # ── Section 5: Chunk detail explorer ─────────────────────────────────
    st.subheader("🔎 Explore lecture segments")
    if scored_chunks:
        chunk_labels = [
            f"{_fmt_time(c['start'])} – {_fmt_time(c['end'])}" for c in scored_chunks
        ]
        selected_label = st.selectbox("Select a segment", chunk_labels)
        selected_idx = chunk_labels.index(selected_label)
        chunk = scored_chunks[selected_idx]

        label = chunk.get("importance_label", "low")
        badge_cls = f"badge-{label}"
        st.markdown(
            f'<span class="{badge_cls}">{label.upper()} IMPORTANCE</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"> {chunk.get('text', '*(no transcript)*')}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pace score", f"{chunk.get('pace_score', 0):.2f}")
        m2.metric("Repetition", f"{chunk.get('repetition_score', 0):.2f}")
        m3.metric("Keyword", f"{chunk.get('keyword_score', 0):.2f}")
        m4.metric("Overlap", f"{chunk.get('overlap_score', 0):.2f}")

        st.markdown(f"**Final score:** `{chunk.get('final_score', 0):.3f}`")

        topics_in_chunk = chunk.get("chunk_topics", [])
        if topics_in_chunk:
            st.markdown("**Topics detected:** " + " · ".join(f"`{t}`" for t in topics_in_chunk[:10]))

        st.info("**Why was this flagged?** " + _build_explanation(chunk))

    # ── Section 6: Download ───────────────────────────────────────────────
    st.subheader("⬇️ Download full report")
    json_bytes = json.dumps(results, indent=2).encode("utf-8")
    st.download_button(
        label="📄 Download JSON report",
        data=json_bytes,
        file_name="lecture_ghost_report.json",
        mime="application/json",
        use_container_width=True,
    )

elif not analyze_btn:
    # Welcome state
    st.markdown(
        """
        <div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;
        padding:2rem;text-align:center;margin-top:2rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">👻</div>
            <h3 style="color:#e2e8f0">Ready to haunt your exams</h3>
            <p style="color:#64748b">Upload your lecture recording in the sidebar and click <strong>Analyze</strong>.<br>
            Optionally add lecture slides, past exam papers, and assignments for richer predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
