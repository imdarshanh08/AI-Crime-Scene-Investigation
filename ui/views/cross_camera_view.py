"""Streamlit view for cross-camera link proposals and their review."""

import pandas as pd
import streamlit as st

from datetime import datetime

from services.correlation.cross_camera_linking import (
    DATE_FORMAT,
    REVIEW_STATUSES,
    generate_cross_camera_links,
    get_anchors,
    get_link_history,
    get_links,
    get_reconstructed_events,
    initialise_database,
    review_link,
    save_anchor,
)


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

def render_cross_camera_linking():

    initialise_database()

    st.subheader("Cross-camera link proposals")

    st.caption(
        "Find potentially related events across CCTV sources "
        "using registered recording times and detected categories."
    )

    st.warning(
        "Enter verified CCTV recording start times. "
        "Do not use WhatsApp receipt times, file creation times, "
        "or assumed timestamps as actual recording times."
    )

    events = get_reconstructed_events()

    if events is None or events.empty:

        st.info(
            "Generate candidate events in the Event Reconstruction "
            "Engine before using cross-camera linking."
        )

        return

    sources = (
        events[["source_id", "source"]]
        .drop_duplicates()
        .dropna(subset=["source_id"])
    )

    # --------------------------------------------------------
    # CAMERA REGISTRATION
    # --------------------------------------------------------

    st.subheader("1. Register CCTV recording times")

    source_options = {
        f"{row['source']} ({row['source_id']})":
        (str(row["source_id"]), str(row["source"]))
        for _, row in sources.iterrows()
    }

    selected_source = st.selectbox(
        "Select CCTV source",
        list(source_options.keys()),
        key="cross_camera_source"
    )

    source_id, source_name = source_options[selected_source]

    anchors = get_anchors()

    existing = (
        anchors[
            anchors["source_id"].astype(str) == source_id
        ]
        if not anchors.empty
        else pd.DataFrame()
    )

    existing_value = ""

    if not existing.empty:

        stored_time = datetime.fromisoformat(
            str(existing.iloc[0]["recording_start"])
        )

        existing_value = stored_time.strftime(DATE_FORMAT)

    st.caption(
        "Enter the actual date and time when this CCTV recording began."
    )

    recording_start = st.text_input(
        "Recording start time (DD-MM-YYYY HH:MM:SS)",
        value=existing_value,
        placeholder="06-11-2026 14:30:00",
        key=f"anchor_input_{source_id}"
    )

    st.caption(
        "Example: 06-11-2026 14:30:00 means "
        "6 November 2026 at 2:30 PM."
    )

    if st.button(
        "Save camera time anchor",
        width="stretch"
    ):

        try:

            save_anchor(
                source_id,
                source_name,
                recording_start.strip()
            )

            st.success("Camera recording time saved.")

            st.rerun()

        except ValueError:

            st.error(
                "Invalid date/time. "
                "Use DD-MM-YYYY HH:MM:SS."
            )

    # --------------------------------------------------------
    # REGISTERED CAMERAS
    # --------------------------------------------------------

    st.markdown("**Registered camera time anchors**")

    anchors = get_anchors()

    if anchors.empty:

        st.info("No camera recording times registered yet.")

    else:

        display_anchors = anchors.copy()

        display_anchors["recording_start"] = (
            display_anchors["recording_start"].apply(
                lambda value: datetime.fromisoformat(
                    str(value)
                ).strftime(DATE_FORMAT)
            )
        )

        st.dataframe(
            display_anchors[
                [
                    "source_name",
                    "source_id",
                    "recording_start"
                ]
            ],
            width="stretch",
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # GENERATE PROPOSALS
    # --------------------------------------------------------

    st.subheader("2. Generate cross-camera proposals")

    max_gap = st.slider(
        "Maximum separation between event intervals (seconds)",
        min_value=5,
        max_value=600,
        value=60,
        step=5,
        key="cross_camera_max_gap"
    )

    if st.button(
        "Find potentially related events",
        type="primary",
        width="stretch"
    ):

        with st.spinner(
            "Comparing events across camera sources..."
        ):

            count = generate_cross_camera_links(max_gap)

        st.success(
            f"Cross-camera comparison completed. "
            f"{count} candidate links generated or refreshed."
        )

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # LINK DASHBOARD
    # --------------------------------------------------------

    st.subheader("3. Cross-camera link dashboard")

    links = get_links()

    if links.empty:

        st.info(
            "No cross-camera links yet. Register recording times "
            "for at least two sources, then generate proposals."
        )

        return

    total = len(links)

    pending = int(
        (links["review_status"] == "Pending").sum()
    )

    approved = int(
        (links["review_status"] == "Approved").sum()
    )

    rejected = int(
        (links["review_status"] == "Rejected").sum()
    )

    flagged = int(
        (links["review_status"] == "Flagged").sum()
    )

    metrics = st.columns(5)

    metrics[0].metric("Total links", total)
    metrics[1].metric("Pending", pending)
    metrics[2].metric("Approved", approved)
    metrics[3].metric("Rejected", rejected)
    metrics[4].metric("Flagged", flagged)

    st.dataframe(
        links[
            [
                "link_id",
                "source_a",
                "source_b",
                "category",
                "time_gap_seconds",
                "review_status"
            ]
        ],
        width="stretch",
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # LINK INSPECTION
    # --------------------------------------------------------

    st.subheader("4. Inspect and verify a proposed link")

    link_options = {
        f"{row['link_id']} | "
        f"{row['source_a']} <-> {row['source_b']}":
        row["link_id"]

        for _, row in links.iterrows()
    }

    selected_label = st.selectbox(
        "Select a proposed link",
        list(link_options.keys()),
        key="selected_cross_camera_link"
    )

    selected_id = link_options[selected_label]

    selected = links[
        links["link_id"] == selected_id
    ].iloc[0]

    st.markdown("**Proposed relationship**")

    st.write(selected["explanation"])

    st.write({
        "Link ID": selected["link_id"],
        "Event A": selected["event_a"],
        "Event B": selected["event_b"],
        "Camera A": selected["source_a"],
        "Camera B": selected["source_b"],
        "Detected category": selected["category"],
        "Event A absolute start": selected["event_a_time"],
        "Event B absolute start": selected["event_b_time"],
        "Interval separation (seconds)": selected["time_gap_seconds"]
    })

    st.info(
        "A matching category and nearby time do not establish "
        "that the same physical person or object appears in both videos."
    )

    current_status = selected["review_status"]

    status = st.selectbox(
        "Investigator decision",
        REVIEW_STATUSES,
        index=REVIEW_STATUSES.index(current_status),
        key=f"cross_review_status_{selected_id}"
    )

    notes = st.text_area(
        "Investigator notes",
        value=selected["reviewer_notes"] or "",
        key=f"cross_review_notes_{selected_id}"
    )

    if st.button(
        "Save link verification decision",
        type="primary",
        width="stretch",
        key=f"save_cross_review_{selected_id}"
    ):

        try:

            review_link(
                selected_id,
                status,
                notes
            )

            st.success(
                "Link decision and review history saved."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Could not save decision: {error}"
            )

    # --------------------------------------------------------
    # REVIEW HISTORY
    # --------------------------------------------------------

    st.markdown("**Link review history**")

    history = get_link_history(selected_id)

    if history.empty:

        st.info("No review history for this link yet.")

    else:

        st.dataframe(
            history,
            width="stretch",
            hide_index=True
        )

    st.caption(
        "Only investigator-reviewed links should be treated "
        "as verified relationships. Original evidence remains unchanged."
    )
