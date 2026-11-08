
import streamlit as st
import pandas as pd

from event_reconstruction import (
    reconstruct_events,
    get_reconstructed_events,
    get_event_observations,
    get_event_review_history,
    review_event,
    get_event_summary,
    normalise_observations,
)


def format_event_time(seconds):
    if pd.isna(seconds):
        return "Unknown"

    seconds = max(0, int(float(seconds)))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{remaining:02d}"


def render_event_reconstruction(observations):

    st.header("Event Reconstruction Engine")
    st.caption(
        "Generate and review possible events from existing "
        "CCTV observations."
    )

    st.info(
        "Events are AI-generated investigative candidates. "
        "They are not confirmed incidents or proof of identity."
    )

    df = normalise_observations(observations)

    if df.empty:
        st.warning(
            "No usable observations are available. "
            "Upload and analyse CCTV footage first."
        )
        return

    cctv_df = df[
        (df["Modality"].str.lower() == "cctv")
        & (~df["Track ID"].isin(["", "None", "nan"]))
    ]

    st.subheader("Observation overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total observations", len(df))
    col2.metric("CCTV observations", len(cctv_df))
    col3.metric(
        "Unique tracking IDs",
        cctv_df["Track ID"].nunique()
    )

    st.divider()

    st.subheader("1. Generate candidate events")

    st.write(
        "The engine groups consecutive observations that share "
        "a tracking ID within the same CCTV source."
    )

    col1, col2 = st.columns(2)

    with col1:
        max_gap = st.slider(
            "Maximum gap between observations (seconds)",
            min_value=5,
            max_value=120,
            value=30,
            step=5,
            key="event_max_gap"
        )

    with col2:
        min_observations = st.slider(
            "Minimum observations per event",
            min_value=2,
            max_value=10,
            value=2,
            key="event_min_observations"
        )

    if st.button(
        "Generate candidate events",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Reconstructing candidate events..."):

            generated = reconstruct_events(
                observations,
                max_gap_seconds=max_gap,
                min_observations=min_observations
            )

        st.success(
            f"Reconstruction complete. "
            f"{len(generated)} candidate events generated."
        )

        st.rerun()

    st.divider()

    st.subheader("2. Event dashboard")

    events = get_reconstructed_events()

    summary = get_event_summary()

    metric_cols = st.columns(5)

    metric_cols[0].metric(
        "Total events", summary["total_events"]
    )

    metric_cols[1].metric(
        "Pending", summary["pending_events"]
    )

    metric_cols[2].metric(
        "Approved", summary["approved_events"]
    )

    metric_cols[3].metric(
        "Rejected", summary["rejected_events"]
    )

    metric_cols[4].metric(
        "Flagged", summary["flagged_events"]
    )

    if events.empty:
        st.info(
            "No candidate events have been generated yet. "
            "Use the Generate candidate events button above."
        )
        return

    display_events = events.copy()

    display_events["Start time"] = (
        display_events["start_seconds"].apply(format_event_time)
    )

    display_events["End time"] = (
        display_events["end_seconds"].apply(format_event_time)
    )

    display_events["Confidence"] = (
        display_events["average_confidence"]
        .apply(
            lambda value:
            f"{float(value) * 100:.1f}%"
            if pd.notna(value)
            else "Unavailable"
        )
    )

    st.dataframe(
        display_events[
            [
                "event_id",
                "event_title",
                "source",
                "object_category",
                "Start time",
                "End time",
                "observation_count",
                "Confidence",
                "review_status"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("3. Inspect an event")

    event_options = {
        f"{row['event_id']} | {row['event_title']} | "
        f"{row['source']}": row["event_id"]
        for _, row in events.iterrows()
    }

    selected_label = st.selectbox(
        "Select a candidate event",
        list(event_options.keys()),
        key="selected_reconstructed_event"
    )

    selected_event_id = event_options[selected_label]

    selected_event = events[
        events["event_id"] == selected_event_id
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Observations",
        int(selected_event["observation_count"])
    )

    col2.metric(
        "Review status",
        selected_event["review_status"]
    )

    col3.metric(
        "Duration",
        f"{max(0, selected_event['end_seconds'] - selected_event['start_seconds']):.2f} sec"
    )

    st.markdown("**Event explanation**")

    st.write(selected_event["event_explanation"])

    st.markdown("**Event information**")

    st.write({
        "Event ID": selected_event["event_id"],
        "Source": selected_event["source"],
        "Source ID": selected_event["source_id"],
        "Modality": selected_event["modality"],
        "Object category": selected_event["object_category"],
        "Start (source-relative)": format_event_time(
            selected_event["start_seconds"]
        ),
        "End (source-relative)": format_event_time(
            selected_event["end_seconds"]
        ),
    })

    st.markdown("**Supporting observations**")

    event_observation_ids = get_event_observations(
        selected_event_id
    )

    if not event_observation_ids.empty:

        observation_ids = set(
            event_observation_ids["observation_id"].astype(str)
        )

        supporting = df[
            df["Observation ID"].astype(str).isin(observation_ids)
        ].copy()

        if not supporting.empty:

            st.dataframe(
                supporting,
                use_container_width=True,
                hide_index=True
            )

            for _, observation in supporting.iterrows():

                evidence_path = observation.get("Evidence Path")

                if (
                    isinstance(evidence_path, str)
                    and evidence_path.strip()
                ):

                    from pathlib import Path

                    evidence_file = Path(evidence_path)

                    if evidence_file.is_file():
                        with st.expander(
                            f"Evidence preview: "
                            f"{observation['Observation ID']}"
                        ):

                            if evidence_file.suffix.lower() in [
                                ".jpg", ".jpeg", ".png", ".webp"
                            ]:
                                st.image(
                                    str(evidence_file),
                                    use_container_width=True
                                )

                            elif evidence_file.suffix.lower() in [
                                ".mp4", ".avi", ".mov", ".mkv"
                            ]:
                                st.video(str(evidence_file))

        else:
            st.warning(
                "The event references observations that are not "
                "currently available in the application data."
            )

    else:
        st.warning("No supporting observation records found.")

    st.divider()

    st.subheader("4. Investigator verification")

    st.write(
        "Review the supporting evidence before changing the "
        "event's verification status."
    )

    current_status = selected_event["review_status"]

    status_options = [
        "Pending",
        "Approved",
        "Rejected",
        "Flagged"
    ]

    selected_status = st.selectbox(
        "Investigation decision",
        status_options,
        index=status_options.index(current_status),
        key=f"event_status_{selected_event_id}"
    )

    existing_notes = selected_event["reviewer_notes"] or ""

    reviewer_notes = st.text_area(
        "Investigator notes",
        value=existing_notes,
        key=f"event_notes_{selected_event_id}"
    )

    if st.button(
        "Save investigator decision",
        type="primary",
        key=f"save_event_review_{selected_event_id}"
    ):

        try:

            review_event(
                selected_event_id,
                selected_status,
                reviewer_notes
            )

            st.success(
                "Investigator decision and audit history saved."
            )

            st.rerun()

        except Exception as error:
            st.error(f"Could not save review: {error}")

    st.divider()

    st.subheader("5. Event review history")

    history = get_event_review_history(selected_event_id)

    if history.empty:

        st.info("No investigator decisions have been recorded yet.")

    else:

        st.dataframe(
            history[
                [
                    "previous_status",
                    "new_status",
                    "reviewer_notes",
                    "reviewed_at"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Event timestamps are source-relative. "
        "Cross-camera synchronization is not assumed. "
        "An event generated from tracking continuity is a "
        "candidate association, not proof of identity."
    )
