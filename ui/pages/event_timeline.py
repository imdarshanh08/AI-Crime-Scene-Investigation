"""Event timeline: candidate events (with investigator review) and the unified chronology."""

import streamlit as st

from storage import db_manager
from ui.components import page_header
from ui.views.event_review_view import render_event_reconstruction
from ui.views.timeline_view import render_unified_evidence_timeline


def render():
    page_header(
        "Event timeline",
        "Group tracked observations into candidate events, review them, and browse every timestamped item in order.",
        eyebrow="Analysis",
    )
    obs = db_manager.get_all_observations_df()
    t_events, t_timeline = st.tabs(["Candidate events", "Unified timeline"])
    with t_events:
        render_event_reconstruction(obs)
    with t_timeline:
        render_unified_evidence_timeline(obs)


if __name__ == "__main__":
    render()
