
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# SECTION 25 MODULE: UNIFIED EVIDENCE TIMELINE
# ============================================================

def _normalise_timeline_data(observations):
    """Convert observations into a consistent DataFrame."""

    if observations is None:
        return pd.DataFrame()

    if isinstance(observations, pd.DataFrame):
        df = observations.copy()
    else:
        try:
            df = pd.DataFrame(observations)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return df

    required_defaults = {
        "Observation ID": "",
        "Modality": "Unknown",
        "Source": "Unknown source",
        "Source ID": "",
        "Timestamp": "Not available",
        "Timestamp (seconds)": None,
        "Segment End (seconds)": None,
        "Object": "Unknown",
        "Track ID": "Not applicable",
        "Confidence": None,
        "Observation": "",
        "Original Transcript": "",
        "English Translation": "",
        "Evidence Path": "",
        "Review Status": "Pending",
        "Review Notes": "",
        "Reviewed At": "",
    }

    for column, default in required_defaults.items():
        if column not in df.columns:
            df[column] = default

    df["Modality"] = (
        df["Modality"]
        .fillna("Unknown")
        .astype(str)
    )

    df["Source"] = (
        df["Source"]
        .fillna("Unknown source")
        .astype(str)
    )

    df["Review Status"] = (
        df["Review Status"]
        .fillna("Pending")
        .astype(str)
    )

    df["Timestamp (seconds)"] = pd.to_numeric(
        df["Timestamp (seconds)"],
        errors="coerce"
    )

    df["Segment End (seconds)"] = pd.to_numeric(
        df["Segment End (seconds)"],
        errors="coerce"
    )

    return df


def _safe_text(value, fallback="Not available"):
    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    return value if value else fallback


def _format_seconds(value):
    """Format seconds as MM:SS.ss or HH:MM:SS.ss."""

    try:
        total_seconds = float(value)

        if total_seconds < 0:
            return "Invalid timestamp"

        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60

        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"

        return f"{minutes:02d}:{seconds:05.2f}"

    except (TypeError, ValueError):
        return "Time unavailable"


def _get_observation_description(row):
    modality = str(row.get("Modality", "")).lower()

    if modality == "audio":
        original = _safe_text(
            row.get("Original Transcript"),
            ""
        )

        translation = _safe_text(
            row.get("English Translation"),
            ""
        )

        if original and translation:
            return (
                f"Original speech: {original}\n\n"
                f"English translation: {translation}"
            )

        if original:
            return f"Original speech: {original}"

        if translation:
            return f"English translation: {translation}"

    return _safe_text(
        row.get("Observation"),
        "No observation description available."
    )


def _display_evidence_preview(row):
    evidence_path = _safe_text(
        row.get("Evidence Path"),
        ""
    )

    if not evidence_path:
        return

    path = Path(evidence_path)

    if not path.is_file():
        st.caption("Evidence preview file is not currently available.")
        return

    modality = str(row.get("Modality", "")).lower()

    if modality in ("cctv", "photo"):
        try:
            st.image(
                str(path),
                caption="Linked visual evidence",
                width="stretch"
            )
        except Exception:
            st.caption("Could not display the linked image.")

    elif modality == "audio":
        try:
            if path.suffix.lower() in {
                ".wav", ".mp3", ".m4a", ".aac",
                ".flac", ".ogg"
            }:
                st.audio(str(path))
        except Exception:
            st.caption("Could not play the linked audio evidence.")


def _render_observation(row, index):
    """Render one observation in the timeline."""

    modality = _safe_text(row.get("Modality"), "Unknown")
    source = _safe_text(row.get("Source"), "Unknown source")
    timestamp = _safe_text(
        row.get("Timestamp"),
        "Time unavailable"
    )

    observation_id = _safe_text(
        row.get("Observation ID"),
        f"Observation {index + 1}"
    )

    review_status = _safe_text(
        row.get("Review Status"),
        "Pending"
    )

    object_name = _safe_text(
        row.get("Object"),
        "Unknown"
    )

    title = (
        f"{timestamp} | {modality} | "
        f"{object_name} | {review_status}"
    )

    with st.expander(title):

        left, right = st.columns(2)

        with left:
            st.write("**Observation ID:**", observation_id)
            st.write("**Evidence source:**", source)
            st.write("**Modality:**", modality)
            st.write("**Object / content type:**", object_name)
            st.write("**Review status:**", review_status)

        with right:
            st.write(
                "**Track ID:**",
                _safe_text(row.get("Track ID"))
            )

            st.write(
                "**Confidence:**",
                _safe_text(row.get("Confidence"))
            )

            st.write(
                "**Reviewed at:**",
                _safe_text(row.get("Reviewed At"))
            )

            st.write(
                "**Timestamp:**",
                timestamp
            )

        st.markdown("**Observation details**")

        description = _get_observation_description(row)

        st.write(description)

        original = _safe_text(
            row.get("Original Transcript"),
            ""
        )

        translation = _safe_text(
            row.get("English Translation"),
            ""
        )

        if original:
            st.markdown("**Original-language transcript**")
            st.write(original)

        if translation:
            st.markdown("**English translation**")
            st.write(translation)

        notes = _safe_text(
            row.get("Review Notes"),
            ""
        )

        if notes:
            st.markdown("**Investigator review notes**")
            st.write(notes)

        _display_evidence_preview(row)


def render_unified_evidence_timeline(observations):
    """
    Render a source-aware unified evidence timeline.

    Important:
    Timestamps are relative to each individual recording.
    This module does not assume that separate recordings are
    synchronized to the same clock.
    """

    st.divider()

    st.subheader("Unified evidence timeline")

    st.write(
        "Explore CCTV detections, image observations and "
        "timestamped audio transcripts in one evidence interface."
    )

    df = _normalise_timeline_data(observations)

    if df.empty:
        st.info(
            "No evidence observations are available yet. "
            "Upload and analyse CCTV, photographs or audio "
            "to populate the unified timeline."
        )
        return

    # --------------------------------------------------------
    # OVERVIEW METRICS
    # --------------------------------------------------------

    total = len(df)

    cctv_count = int(
        (df["Modality"].str.lower() == "cctv").sum()
    )

    photo_count = int(
        (df["Modality"].str.lower() == "photo").sum()
    )

    audio_count = int(
        (df["Modality"].str.lower() == "audio").sum()
    )

    pending_count = int(
        (df["Review Status"].str.lower() == "pending").sum()
    )

    approved_count = int(
        (df["Review Status"].str.lower() == "approved").sum()
    )

    metrics = st.columns(5)

    metrics[0].metric("Total observations", total)
    metrics[1].metric("CCTV", cctv_count)
    metrics[2].metric("Photos", photo_count)
    metrics[3].metric("Audio segments", audio_count)
    metrics[4].metric("Pending review", pending_count)

    st.caption(
        f"{approved_count} observation(s) currently marked Approved."
    )

    st.divider()

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.subheader("Timeline filters")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    modality_options = sorted(
        df["Modality"].dropna().astype(str).unique().tolist()
    )

    with filter_col1:
        selected_modalities = st.multiselect(
            "Evidence modality",
            options=modality_options,
            default=modality_options,
            key="unified_timeline_modalities"
        )

    status_options = sorted(
        df["Review Status"].dropna().astype(str).unique().tolist()
    )

    with filter_col2:
        selected_statuses = st.multiselect(
            "Investigator review status",
            options=status_options,
            default=status_options,
            key="unified_timeline_statuses"
        )

    source_options = sorted(
        df["Source"].dropna().astype(str).unique().tolist()
    )

    with filter_col3:
        selected_sources = st.multiselect(
            "Evidence source",
            options=source_options,
            default=source_options,
            key="unified_timeline_sources"
        )

    search_text = st.text_input(
        "Search objects, transcripts, observations or source names",
        key="unified_timeline_search"
    )

    filtered = df[
        df["Modality"].isin(selected_modalities)
        & df["Review Status"].isin(selected_statuses)
        & df["Source"].isin(selected_sources)
    ].copy()

    if search_text.strip():
        search_columns = [
            "Source",
            "Object",
            "Observation",
            "Original Transcript",
            "English Translation",
            "Track ID",
            "Observation ID"
        ]

        search_mask = pd.Series(
            False,
            index=filtered.index
        )

        for column in search_columns:
            search_mask |= (
                filtered[column]
                .fillna("")
                .astype(str)
                .str.contains(
                    search_text.strip(),
                    case=False,
                    regex=False
                )
            )

        filtered = filtered[search_mask].copy()

    st.caption(
        f"Showing {len(filtered)} of {total} observations."
    )

    if filtered.empty:
        st.warning("No observations match the selected filters.")
        return

    # --------------------------------------------------------
    # SOURCE-AWARE TIMELINE
    # --------------------------------------------------------

    st.divider()

    st.subheader("Chronological evidence")

    st.warning(
        "Timestamps are relative to the beginning of their "
        "respective recordings. Separate CCTV and audio files "
        "are not merged into a single real-world clock timeline "
        "unless verified recording-time anchors are available."
    )

    # Photos with no verified timestamp are kept separate.
    timed = filtered[
        filtered["Timestamp (seconds)"].notna()
    ].copy()

    untimed = filtered[
        filtered["Timestamp (seconds)"].isna()
    ].copy()

    # Each source gets its own timeline. This avoids falsely
    # implying synchronization between separate recordings.
    source_groups = timed.groupby(
        ["Modality", "Source"],
        sort=True,
        dropna=False
    )

    for (modality, source), group in source_groups:

        group = group.sort_values(
            by=["Timestamp (seconds)", "Observation ID"],
            kind="stable"
        )

        first_time = _format_seconds(
            group["Timestamp (seconds)"].min()
        )

        last_time = _format_seconds(
            group["Timestamp (seconds)"].max()
        )

        heading = (
            f"{modality} | {source} "
            f"({len(group)} observations)"
        )

        with st.expander(
            heading,
            expanded=True
        ):

            st.caption(
                f"Recording-relative range: {first_time} to {last_time}"
            )

            for index, (_, row) in enumerate(group.iterrows()):
                _render_observation(row, index)

    # --------------------------------------------------------
    # TIME UNKNOWN GROUP
    # --------------------------------------------------------

    if not untimed.empty:

        st.divider()

        st.subheader("Evidence without verified timestamps")

        st.info(
            "These observations cannot be placed accurately "
            "on a chronological timeline because a verified "
            "capture or recording timestamp is unavailable."
        )

        for index, (_, row) in enumerate(
            untimed.iterrows()
        ):
            _render_observation(row, index)

    # --------------------------------------------------------
    # SUMMARY TABLE
    # --------------------------------------------------------

    st.divider()

    with st.expander("View timeline as a data table"):

        display_columns = [
            "Observation ID",
            "Modality",
            "Source",
            "Timestamp",
            "Object",
            "Track ID",
            "Review Status",
            "Observation"
        ]

        available_columns = [
            column
            for column in display_columns
            if column in filtered.columns
        ]

        table = filtered[available_columns].copy()

        st.dataframe(
            table,
            width="stretch",
            hide_index=True
        )

    st.caption(
        "Unified Evidence Timeline | AI-generated observations "
        "remain separate from investigator-verified findings. "
        "Preserve original evidence files and verify all findings."
    )
