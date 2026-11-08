# ============================================================
# SECTION 26: EVIDENCE SEARCH & NATURAL LANGUAGE QUERY ENGINE
# ============================================================

import re
from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

SEARCH_COLUMNS = [
    "Observation ID",
    "Modality",
    "Source",
    "Source ID",
    "Camera ID",
    "Timestamp",
    "Timestamp (seconds)",
    "Segment End (seconds)",
    "Object",
    "Track ID",
    "Raw Track ID",
    "Confidence",
    "Observation",
    "Original Transcript",
    "English Translation",
    "Review Status",
    "Review Notes",
    "Evidence Path",
]

STOP_WORDS = {
    "show", "find", "search", "everything", "related", "about",
    "what", "happened", "between", "from", "to", "the", "a",
    "an", "me", "all", "evidence", "records", "observations",
    "please", "give", "list", "display", "during", "in", "on",
    "at", "and", "or", "is", "was", "were", "with", "of",
    "this", "that", "there", "which", "who", "where",
    "detections", "detection", "camera", "cctv", "footage", "video",
    "photo", "audio", "transcripts", "transcript", "recordings",
    "recording", "images", "image", "sound", "seen", "appeared"
}

MODALITY_ALIASES = {
    "video": "CCTV",
    "camera": "CCTV",
    "cctv": "CCTV",
    "image": "Photo",
    "picture": "Photo",
    "photo": "Photo",
    "audio": "Audio",
    "sound": "Audio",
    "speech": "Audio",
    "voice": "Audio",
}


# ------------------------------------------------------------
# 2. DATA NORMALISATION
# ------------------------------------------------------------

def _normalise_observations(observations):
    if observations is None:
        return pd.DataFrame(columns=SEARCH_COLUMNS)

    df = observations.copy()

    for column in SEARCH_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df.copy()

    for column in [
        "Observation ID", "Modality", "Source", "Source ID",
        "Timestamp", "Object", "Track ID", "Raw Track ID",
        "Observation", "Original Transcript",
        "English Translation", "Review Status",
        "Review Notes", "Evidence Path",
    ]:
        df[column] = df[column].fillna("").astype(str)

    df["Timestamp (seconds)"] = pd.to_numeric(
        df["Timestamp (seconds)"], errors="coerce"
    )

    df["Segment End (seconds)"] = pd.to_numeric(
        df["Segment End (seconds)"], errors="coerce"
    )

    df["Confidence"] = pd.to_numeric(
        df["Confidence"], errors="coerce"
    )

    return df


# ------------------------------------------------------------
# 3. TIME PARSING
# ------------------------------------------------------------

def _time_to_seconds(value):
    """
    Converts:
    10:30       -> 630 seconds (MM:SS)
    01:10:30    -> 4230 seconds (HH:MM:SS)
    90          -> 90 seconds
    """

    value = str(value).strip()

    if not value:
        return None

    try:
        if ":" not in value:
            return float(value)

        parts = [float(part) for part in value.split(":")]

        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds

        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds

    except (ValueError, TypeError):
        return None

    return None


def _extract_time_range(query):
    """
    Recognises queries such as:
    between 10:30 and 10:45
    from 00:02:10 to 00:05:00
    """

    patterns = [
        r"(?:between|from)\s+(\d{1,2}:\d{2}(?::\d{2})?)"
        r"\s+(?:and|to)\s+(\d{1,2}:\d{2}(?::\d{2})?)"
    ]

    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)

        if match:
            start = _time_to_seconds(match.group(1))
            end = _time_to_seconds(match.group(2))

            if start is not None and end is not None:
                return min(start, end), max(start, end)

    return None


# ------------------------------------------------------------
# 4. QUERY INTERPRETATION
# ------------------------------------------------------------

def _interpret_query(query):
    lowered = query.lower()

    interpreted = {
        "modality": None,
        "time_range": _extract_time_range(lowered),
        "terms": [],
    }

    for alias, modality in MODALITY_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", lowered):
            interpreted["modality"] = modality
            break

    cleaned = re.sub(
        r"(?:between|from)\s+\d{1,2}:\d{2}(?::\d{2})?"
        r"\s+(?:and|to)\s+\d{1,2}:\d{2}(?::\d{2})?",
        " ",
        lowered,
        flags=re.IGNORECASE,
    )

    words = re.findall(r"[a-zA-Z0-9_-]+", cleaned)

    interpreted["terms"] = [
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    ]

    return interpreted


# ------------------------------------------------------------
# 5. SEARCH ENGINE
# ------------------------------------------------------------

def search_evidence(observations, query):
    df = _normalise_observations(observations)

    if df.empty:
        return df, _interpret_query(query)

    interpretation = _interpret_query(query)
    results = df.copy()

    # Modality filter
    if interpretation["modality"]:
        results = results[
            results["Modality"].str.lower()
            == interpretation["modality"].lower()
        ]

    # Time range filter.
    # Timestamps are evaluated relative to each source video.
    time_range = interpretation["time_range"]

    if time_range:
        start_seconds, end_seconds = time_range

        results = results[
            results["Timestamp (seconds)"].notna()
            & (
                results["Timestamp (seconds)"].between(
                    start_seconds, end_seconds
                )
                |
                results["Segment End (seconds)"].between(
                    start_seconds, end_seconds
                )
            )
        ]

    # Text matching
    searchable_columns = [
        "Observation ID",
        "Camera ID",
        "Source",
        "Source ID",
        "Timestamp",
        "Object",
        "Track ID",
        "Raw Track ID",
        "Observation",
        "Original Transcript",
        "English Translation",
        "Review Status",
        "Review Notes",
    ]

    if interpretation["terms"]:
        searchable_text = (
            results[searchable_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )

        # Require every meaningful search term to be present.
        mask = pd.Series(True, index=results.index)

        for term in interpretation["terms"]:
            mask &= searchable_text.str.contains(
                re.escape(term), regex=True, na=False
            )

        results = results[mask]

    results = results.sort_values(
        by=["Source", "Timestamp (seconds)"],
        na_position="last",
    )

    return results.reset_index(drop=True), interpretation


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
            st.image(str(path), caption=path.name, use_container_width=True)

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

    st.header("🔎 Section 26: Evidence Search & Natural Language Queries")

    st.write(
        "Search CCTV detections, photographs, audio transcripts, "
        "tracking records and investigator review notes using "
        "ordinary language."
    )

    st.info(
        "Example: Show me everything related to the backpack"
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
        use_container_width=True,
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