"""Streamlit view for keyword / parametric evidence search."""

import pandas as pd
import streamlit as st

from pathlib import Path

from services.search.evidence_search_engine import (
    search_evidence,
)


# ------------------------------------------------------------
# 6. EVIDENCE PREVIEW
# ------------------------------------------------------------

def _render_evidence_preview(row):
    evidence_path = str(row.get("Evidence Path", "")).strip()

    if not evidence_path:
        st.caption("No evidence file path is associated with this record.")
        return

    path = Path(evidence_path)

    if not path.is_file():
        st.caption("Evidence file is not currently accessible at its recorded path.")
        return

    extension = path.suffix.lower()

    try:
        if extension in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            st.image(str(path), caption=path.name, width="stretch")

        elif extension in [".mp4", ".mov", ".avi", ".mkv"]:
            st.video(str(path))

        elif extension in [".wav", ".mp3", ".m4a", ".ogg", ".flac"]:
            st.audio(str(path))

        else:
            st.caption(f"Evidence file: {path.name}")

    except Exception as error:
        st.warning(f"Preview unavailable: {error}")


# ------------------------------------------------------------
# 7. SEARCH RESULT DISPLAY
# ------------------------------------------------------------

def _render_search_result(row, index):
    object_name = row.get("Object", "Unknown object")
    source = row.get("Source", "Unknown source")
    modality = row.get("Modality", "Unknown")
    timestamp = row.get("Timestamp", "Not available")
    observation_id = row.get("Observation ID", "Unknown ID")
    review_status = row.get("Review Status", "Pending")

    heading = (
        f"{index + 1}. {object_name} | "
        f"{modality} | {timestamp}"
    )

    with st.expander(heading):
        left, right = st.columns(2)

        with left:
            st.write("**Observation ID:**", observation_id)
            st.write("**Source:**", source)
            st.write("**Modality:**", modality)
            st.write("**Timestamp:**", timestamp)
            st.write("**Track ID:**", row.get("Track ID", ""))
            st.write("**Review Status:**", review_status)

        with right:
            confidence = row.get("Confidence")

            if pd.notna(confidence):
                st.write("**Detection Confidence:**", f"{confidence:.3f}")

            st.write(
                "**Timestamp (seconds):**",
                row.get("Timestamp (seconds)", "Not available"),
            )

        observation = row.get("Observation", "")

        if observation:
            st.write("**Observation details:**")
            st.write(observation)

        original = row.get("Original Transcript", "")
        translation = row.get("English Translation", "")

        if original:
            st.write("**Original transcript:**")
            st.write(original)

        if translation:
            st.write("**English translation:**")
            st.write(translation)

        notes = row.get("Review Notes", "")

        if notes:
            st.write("**Investigator review notes:**")
            st.write(notes)

        st.divider()
        st.write("**Evidence preview**")
        _render_evidence_preview(row)


# ------------------------------------------------------------
# 8. MAIN STREAMLIT INTERFACE
# ------------------------------------------------------------

def render_evidence_search(observations):

    st.caption(
        "Keyword search across detections, transcripts, tracks and review notes. Every term must match. "
        "For questions that need time windows, colours or connected evidence, use Incident Reconstruction."
    )

    query = st.text_input(
        "Enter your investigation query",
        placeholder=(
            "Example: Find backpack between 00:01:30 and 00:03:00"
        ),
        key="evidence_search_query",
    )

    search_clicked = st.button(
        "Search Evidence",
        type="primary",
        key="run_evidence_search",
    )

    if search_clicked:

        if not query.strip():
            st.warning("Enter a search query first.")

        else:
            results, interpretation = search_evidence(
                observations,
                query,
            )

            st.session_state["evidence_search_results"] = results
            st.session_state["evidence_search_interpretation"] = (
                interpretation
            )
            st.session_state["evidence_search_last_query"] = query

    results = st.session_state.get(
        "evidence_search_results",
        pd.DataFrame(),
    )

    interpretation = st.session_state.get(
        "evidence_search_interpretation",
        {},
    )

    if results.empty:
        st.caption(
            "Search results will appear here after you submit a query."
        )
        return

    st.divider()
    st.subheader("Search Results")

    total = len(results)
    cctv_count = int((results["Modality"] == "CCTV").sum())
    photo_count = int((results["Modality"] == "Photo").sum())
    audio_count = int((results["Modality"] == "Audio").sum())

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Matching records", total)
    metric2.metric("CCTV", cctv_count)
    metric3.metric("Photos", photo_count)
    metric4.metric("Audio", audio_count)

    if interpretation.get("modality"):
        st.caption(
            f"Detected modality filter: {interpretation['modality']}"
        )

    if interpretation.get("time_range"):
        start, end = interpretation["time_range"]

        st.caption(
            f"Time filter: {start:.1f}s to {end:.1f}s "
            "(relative to each source recording)"
        )

    if total == 0:
        st.warning("No matching evidence records were found.")
        return

    st.write("**Matching evidence overview**")

    overview_columns = [
        "Observation ID",
        "Modality",
        "Source",
        "Timestamp",
        "Object",
        "Track ID",
        "Review Status",
    ]

    st.dataframe(
        results[overview_columns],
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.subheader("Inspect Matching Evidence")

    for index, (_, row) in enumerate(results.iterrows()):
        _render_search_result(row, index)

    st.caption(
        "Search results are retrieved from existing observations. "
        "A match does not establish identity, causation or a verified event."
    )