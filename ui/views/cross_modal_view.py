"""Streamlit view for cross-modal (audio / photo / CCTV) correlation proposals and their review."""

import sqlite3

import pandas as pd
import streamlit as st

from services.correlation.cross_modal_correlation import (
    MAX_PROPOSALS_PER_RUN,
    RELATIONSHIP_TYPES,
    REVIEW_STATUSES,
    generate_cross_modal_candidates,
    get_cross_modal_relationships,
    get_cross_modal_review_history,
    initialise_correlation_database,
    normalise_observations,
    review_cross_modal_relationship,
    save_cross_modal_candidates,
)


# ------------------------------------------------------------
# 12. STREAMLIT INTERFACE
# ------------------------------------------------------------

def render_cross_modal_correlation(observations):

    initialise_correlation_database()

    st.subheader(
        "Cross-modal correlation"
    )

    st.write(
        "Find explainable relationships between CCTV "
        "detections, audio transcripts and photographs."
    )

    st.warning(
        "These are AI-generated candidate relationships, "
        "not verified facts. Matching categories or words "
        "does not establish physical identity or prove "
        "that two observations belong to the same incident."
    )

    df = normalise_observations(observations)

    if df.empty:
        st.info(
            "No observations are available. Upload and analyse "
            "CCTV, audio or photographs first."
        )
        return

    counts = (
        df["Modality"]
        .value_counts()
        .to_dict()
    )

    cctv_count = counts.get("cctv", 0)
    audio_count = counts.get("audio", 0)
    photo_count = counts.get("photo", 0)

    col1, col2, col3 = st.columns(3)

    col1.metric("CCTV observations", cctv_count)
    col2.metric("Audio observations", audio_count)
    col3.metric("Photo observations", photo_count)

    st.divider()

    # --------------------------------------------------------
    # GENERATE CANDIDATES
    # --------------------------------------------------------

    st.subheader("Generate Cross-Modal Proposals")

    st.write(
        "The engine checks transcript mentions and matching "
        "object categories. It does not assume that separate "
        "recordings are synchronized."
    )

    include_photo_context = st.checkbox(
        "Include same-category CCTV and photo context proposals",
        value=True,
        help=(
            "These proposals have low confidence because "
            "same-category detections do not prove object identity."
        ),
    )

    if st.button(
        "Generate Cross-Modal Relationships",
        type="primary",
        key="generate_cross_modal_relationships",
    ):

        with st.spinner(
            "Analysing cross-modal evidence..."
        ):

            candidates = generate_cross_modal_candidates(
                df,
                include_photo_context=include_photo_context,
            )

            inserted = save_cross_modal_candidates(
                candidates
            )

        st.success(
            f"Generated {len(candidates)} candidate(s). "
            f"Added {inserted} new relationship(s) to the "
            f"existing Evidence Graph."
        )

        if len(candidates) >= MAX_PROPOSALS_PER_RUN:
            st.warning(
                "The proposal limit was reached. Review the "
                "current results before generating more."
            )

    # --------------------------------------------------------
    # RELATIONSHIP DASHBOARD
    # --------------------------------------------------------

    st.divider()
    st.subheader("Cross-Modal Relationship Dashboard")

    relationships = get_cross_modal_relationships()

    if not relationships:
        st.info(
            "No cross-modal relationships have been generated yet."
        )
        return

    relationship_df = pd.DataFrame(
        relationships
    )

    total = len(relationship_df)

    pending = int(
        (relationship_df["status"] == "Pending").sum()
    )

    approved = int(
        (relationship_df["status"] == "Approved").sum()
    )

    rejected = int(
        (relationship_df["status"] == "Rejected").sum()
    )

    flagged = int(
        (relationship_df["status"] == "Flagged").sum()
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total", total)
    col2.metric("Pending", pending)
    col3.metric("Approved", approved)
    col4.metric("Rejected", rejected)
    col5.metric("Flagged", flagged)

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    selected_types = st.multiselect(
        "Filter relationship types",
        RELATIONSHIP_TYPES,
        default=RELATIONSHIP_TYPES,
        key="cross_modal_type_filter",
    )

    selected_statuses = st.multiselect(
        "Filter review status",
        REVIEW_STATUSES,
        default=REVIEW_STATUSES,
        key="cross_modal_status_filter",
    )

    visible = relationship_df[
        relationship_df["relationship_type"].isin(
            selected_types
        )
        & relationship_df["status"].isin(
            selected_statuses
        )
    ].copy()

    display_columns = [
        "relationship_id",
        "relationship_type",
        "source_observation",
        "target_observation",
        "confidence",
        "status",
        "origin",
    ]

    st.dataframe(
        visible[display_columns],
        width="stretch",
        hide_index=True,
    )

    if visible.empty:
        st.info(
            "No relationships match the selected filters."
        )
        return

    # --------------------------------------------------------
    # RELATIONSHIP DETAILS
    # --------------------------------------------------------

    st.divider()
    st.subheader("Inspect Relationship")

    relationship_options = {}

    for _, row in visible.iterrows():

        label = (
            str(row["relationship_type"])
            + " | "
            + str(row["source_observation"])
            + " → "
            + str(row["target_observation"])
            + " | "
            + str(row["status"])
        )

        relationship_options[label] = str(
            row["relationship_id"]
        )

    selected_label = st.selectbox(
        "Select a relationship",
        list(relationship_options.keys()),
        key="cross_modal_selected_relationship",
    )

    selected_id = relationship_options[
        selected_label
    ]

    selected = relationship_df[
        relationship_df["relationship_id"].astype(str)
        == selected_id
    ].iloc[0]

    st.write(
        "**Relationship type:**",
        selected["relationship_type"],
    )

    st.write(
        "**Source observation:**",
        selected["source_observation"],
    )

    st.write(
        "**Target observation:**",
        selected["target_observation"],
    )

    st.write(
        "**Proposed confidence:**",
        f"{float(selected['confidence']):.3f}",
    )

    st.write(
        "**Current status:**",
        selected["status"],
    )

    st.markdown("**Explanation**")

    st.info(
        str(selected["explanation"])
    )

    # --------------------------------------------------------
    # REVIEW CONTROLS
    # --------------------------------------------------------

    st.divider()
    st.subheader("Investigator Verification")

    reviewer = st.text_input(
        "Investigator / reviewer",
        key="cross_modal_reviewer",
    )

    decision = st.selectbox(
        "Review decision",
        REVIEW_STATUSES,
        index=REVIEW_STATUSES.index(
            str(selected["status"])
        ),
        key="cross_modal_decision",
    )

    notes = st.text_area(
        "Investigator notes",
        key="cross_modal_review_notes",
    )

    if st.button(
        "Save Investigator Decision",
        type="primary",
        key="save_cross_modal_review",
    ):

        if not reviewer.strip():
            st.error(
                "Enter the investigator's name before saving."
            )

        else:

            complete_notes = (
                f"Reviewer: {reviewer.strip()}\n"
                + notes.strip()
            )

            try:

                changed = review_cross_modal_relationship(
                    selected_id,
                    decision,
                    complete_notes,
                )

                if changed:
                    st.success(
                        "Decision saved in the existing "
                        "Evidence Graph review history."
                    )
                else:
                    st.info(
                        "The relationship already has this status."
                    )

            except (ValueError, sqlite3.Error) as error:
                st.error(str(error))

    # --------------------------------------------------------
    # REVIEW HISTORY
    # --------------------------------------------------------

    st.divider()
    st.subheader("Relationship Review History")

    history = get_cross_modal_review_history(
        selected_id
    )

    if history:

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df[
                [
                    "previous_status",
                    "new_status",
                    "reviewer_notes",
                    "reviewed_at",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    else:
        st.info(
            "No investigator review decisions recorded yet."
        )

    st.caption(
        "This module reuses the existing evidence graph. "
        "Review decisions are recorded in graph_review_history. "
        "Reviewer names are manually entered and are not "
        "authenticated identities."
    )
