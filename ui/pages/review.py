"""Investigator review: decisions on AI findings and the audit trail."""

from datetime import datetime

import streamlit as st

from storage import db_manager
from ui.components import page_header
from ui.state import (
    clean_text as _clean_text,
    sync_session_from_database,
)


def render():
    page_header(
        "Investigator review",
        "Every AI detection and transcript starts as Pending. Decisions and notes are stored separately from the "
        "AI output, which is never overwritten, and every change is written to the audit trail.",
        eyebrow="Review",
    )

    all_obs_df = db_manager.get_all_observations_df()

    if all_obs_df.empty:
        st.info("No observations available for review.")
    else:
        # Status counts
        stat_counts = all_obs_df["Review Status"].value_counts()
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Pending", stat_counts.get("Pending", 0))
        v2.metric("Approved", stat_counts.get("Approved", 0))
        v3.metric("Rejected", stat_counts.get("Rejected", 0))
        v4.metric("Flagged for further review", stat_counts.get("Flagged", 0))

        st.divider()

        # Review single item or table editor
        st.subheader("Review observations")
        rf1, rf2, rf3 = st.columns(3)
        show_status = rf1.multiselect("Show status", ["Pending", "Approved", "Rejected", "Flagged"],
                                      default=["Pending", "Flagged"], key="rev_status_filter")
        show_cams = rf2.multiselect("Camera / channel", sorted(all_obs_df["Camera ID"].dropna().unique()),
                                    key="rev_cam_filter", placeholder="All")
        show_mod = rf3.multiselect("Modality", sorted(all_obs_df["Modality"].unique()), key="rev_mod_filter",
                                   placeholder="All")
        queue = all_obs_df
        if show_status:
            queue = queue[queue["Review Status"].isin(show_status)]
        if show_cams:
            queue = queue[queue["Camera ID"].isin(show_cams)]
        if show_mod:
            queue = queue[queue["Modality"].isin(show_mod)]
        st.caption(f"{len(queue)} item(s) in this view. Edit the status or notes, then save.")

        review_cols = [
            "Observation ID", "Modality", "Camera ID", "Source", "Timestamp",
            "Object", "Confidence", "Observation", "Review Status", "Review Notes"
        ]
        valid_rev_cols = [c for c in review_cols if c in all_obs_df.columns]

        edited_df = st.data_editor(
            queue[valid_rev_cols],
            width="stretch",
            hide_index=True,
            disabled=["Observation ID", "Modality", "Camera ID", "Source", "Timestamp", "Object", "Confidence", "Observation"],
            column_config={
                "Review Status": st.column_config.SelectboxColumn(
                    "Review Status",
                    options=["Pending", "Approved", "Rejected", "Flagged"],
                    required=True
                ),
                "Review Notes": st.column_config.TextColumn("Investigator Notes")
            },
            key="investigator_table_editor"
        )

        if st.button("Save decisions", type="primary", key="btn_save_reviews"):
            updated_count = 0
            originals = all_obs_df.set_index("Observation ID")
            for _, row in edited_df.iterrows():
                obs_id = row["Observation ID"]
                if obs_id not in originals.index:
                    continue
                curr_status = _clean_text(row["Review Status"]) or "Pending"
                curr_notes = _clean_text(row.get("Review Notes"))
                orig_stat = _clean_text(originals.at[obs_id, "Review Status"]) or "Pending"
                orig_note = _clean_text(originals.at[obs_id, "Review Notes"])

                # Compare normalised values. Comparing "" with None/NaN made every row look
                # edited, so each save rewrote and audit-logged the whole table.
                if curr_status != orig_stat or curr_notes != orig_note:
                    db_manager.update_observation_review(obs_id, curr_status, curr_notes)
                    updated_count += 1

            sync_session_from_database()
            st.toast(f"Saved {updated_count} review decision(s)." if updated_count else "No changes to save.")
            st.rerun()

        st.divider()
        st.subheader("📜 Complete Forensic Audit Trail & Chain of Custody")
        st.write("Immutable log of all investigator decisions, status updates, and cryptographic integrity checks.")
        full_audit_df = db_manager.get_audit_logs(limit=None)
        if not full_audit_df.empty:
            st.dataframe(
                full_audit_df[["log_id", "timestamp", "action", "entity_type", "entity_id", "details"]],
                width="stretch",
                hide_index=True
            )
            csv_bytes = full_audit_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Formal Forensic Audit Trail (CSV)",
                data=csv_bytes,
                file_name=f"Forensic_Audit_Trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="btn_download_audit_trail_csv"
            )
        else:
            st.info("No audit entries recorded yet.")


if __name__ == "__main__":
    render()
