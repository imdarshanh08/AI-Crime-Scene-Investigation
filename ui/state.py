"""Session state and small shared helpers for the Streamlit pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core import config as _config
from services.detection import yolo_detector
from storage import db_manager

REVIEW_STATUSES = _config.REVIEW_STATUSES


def sync_session_from_database() -> None:
    """Load persistent records from SQLite into session state (used by legacy views)."""
    db_sources = db_manager.get_all_sources()
    st.session_state.video_records = {s["source_id"]: s for s in db_sources if s["modality"] == "CCTV"}
    st.session_state.photo_records = {s["source_id"]: s for s in db_sources if s["modality"] == "Photo"}
    st.session_state.audio_records = {s["source_id"]: s for s in db_sources if s["modality"] == "Audio"}

    obs_df = db_manager.get_all_observations_df()
    if not obs_df.empty:
        st.session_state.detection_rows = obs_df[obs_df["Modality"] == "CCTV"].to_dict(orient="records")
        st.session_state.photo_rows = obs_df[obs_df["Modality"] == "Photo"].to_dict(orient="records")
        st.session_state.audio_rows = obs_df[obs_df["Modality"] == "Audio"].to_dict(orient="records")
    else:
        st.session_state.detection_rows = []
        st.session_state.photo_rows = []
        st.session_state.audio_rows = []


def init_session() -> None:
    if "initialized" in st.session_state:
        return
    st.session_state.initialized = True
    st.session_state.preview_images = []
    st.session_state.last_report_path = ""
    st.session_state.active_annotated_video = ""
    sync_session_from_database()


@st.cache_resource(show_spinner=False)
def device_info():
    """(device id for Ultralytics, human-readable description). Probed once per process."""
    return yolo_detector.get_optimal_device()


def clean_text(value) -> str:
    """Normalise None/NaN to '' so edited and stored notes compare correctly."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def fmt_pct(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if pd.isna(v) else f"{v:.0%}"


def fmt_secs(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if pd.isna(v) else f"{v:.2f}s"


def unprocessed_uploads(files, state_key):
    """Uploaded files not yet handled in this session.

    The uploader keeps its files across reruns, so without this every rerun re-read and
    re-hashed every uploaded file (up to 2 GB each).
    """
    seen = st.session_state.setdefault(state_key, set())
    return [f for f in files if getattr(f, "file_id", f.name) not in seen], seen


# ---------------------------------------------------------------------------
# Evidence graph, rebuilt only when case data changes
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=2)
def _graph_for(fingerprint: str, _data):
    from services.evidence_graph.engine import build_evidence_graph

    return build_evidence_graph(_data)


def get_evidence_graph():
    """Current evidence graph. Loading the inputs is cheap; the build is cached by fingerprint."""
    from services.evidence_graph.engine import load_case_data

    data = load_case_data()
    return _graph_for(data.fingerprint(), data)
