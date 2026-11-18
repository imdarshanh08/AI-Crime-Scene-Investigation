"""AI Crime Scene & CCTV Investigation Assistant: application entry point.

Run with:  streamlit run app.py

Layout
    core/       configuration, logging, shared timebase
    storage/    SQLite case database and migrations
    services/   media intake, detection, audio, events, correlation,
                evidence graph, incident reconstruction, search, reports
    ui/         theme, shared components, page modules and views
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage import db_manager  # noqa: E402
from ui.state import device_info, init_session  # noqa: E402
from ui.theme import apply_theme  # noqa: E402

st.set_page_config(
    page_title="Crime Scene & CCTV Investigation",
    page_icon=":material/policy:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()
init_session()


def page(path: str, title: str, icon: str, default: bool = False):
    return st.Page(path, title=title, icon=icon, default=default)


navigation = st.navigation({
    "Case": [
        page("ui/pages/dashboard.py", "Investigation dashboard", ":material/space_dashboard:", default=True),
        page("ui/pages/evidence_upload.py", "Evidence upload", ":material/upload_file:"),
    ],
    "Analysis": [
        page("ui/pages/cctv_analysis.py", "CCTV analysis", ":material/videocam:"),
        page("ui/pages/audio_analysis.py", "Audio analysis", ":material/graphic_eq:"),
        page("ui/pages/image_evidence.py", "Image evidence", ":material/image:"),
    ],
    "Investigation": [
        page("ui/pages/repository.py", "Evidence repository", ":material/inventory_2:"),
        page("ui/pages/evidence_graph.py", "Evidence graph", ":material/hub:"),
        page("ui/pages/event_timeline.py", "Event timeline", ":material/timeline:"),
        page("ui/pages/incident_reconstruction.py", "Incident reconstruction", ":material/manage_search:"),
    ],
    "Review & reporting": [
        page("ui/pages/review.py", "Investigator review", ":material/fact_check:"),
        page("ui/pages/workstation.py", "Multi-camera workstation", ":material/view_quilt:"),
        page("ui/pages/reports.py", "Investigation reports", ":material/description:"),
    ],
    "Administration": [
        page("ui/pages/settings.py", "Settings", ":material/settings:"),
    ],
})

with st.sidebar:
    sources = db_manager.get_all_sources()
    counts = {m: sum(1 for s in sources if s["modality"] == m) for m in ("CCTV", "Photo", "Audio")}
    st.caption("CASE FILE")
    st.markdown(f"{counts['CCTV']} recordings · {counts['Photo']} photos · {counts['Audio']} audio")
    _, device_desc = device_info()
    st.caption(f"Detection device: {device_desc}")
    st.divider()
    st.caption("AI outputs are investigative leads. They are not evidence of identity until reviewed by an investigator.")

navigation.run()
