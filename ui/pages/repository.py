"""Evidence repository: every observation with its provenance, plus keyword search."""

from pathlib import Path

import pandas as pd
import streamlit as st

from services.evidence_graph import schema as S
from storage import db_manager
from ui.components import page_header
from ui.state import REVIEW_STATUSES, fmt_pct, get_evidence_graph
from ui.views.search_view import render_evidence_search


def _browse(obs: pd.DataFrame) -> None:
    f1, f2, f3, f4 = st.columns(4)
    mods = f1.multiselect("Modality", sorted(obs["Modality"].unique()), key="rep_filter_mod", placeholder="All")
    cams = f2.multiselect("Camera / channel", sorted(obs["Camera ID"].dropna().unique()), key="rep_filter_cam", placeholder="All")
    objs = f3.multiselect("Object", sorted(obs["Object"].dropna().unique()), key="rep_filter_obj", placeholder="All")
    stats = f4.multiselect("Review status", REVIEW_STATUSES, key="rep_filter_stat", placeholder="All")
    view = obs
    for col, chosen in (("Modality", mods), ("Camera ID", cams), ("Object", objs), ("Review Status", stats)):
        if chosen:
            view = view[view[col].isin(chosen)]
    st.caption(f"{len(view)} of {len(obs)} observations")
    cols = ["Observation ID", "Modality", "Camera ID", "Source", "Timestamp", "Object", "Track ID", "Confidence",
            "Observation", "Review Status", "Created At"]
    st.dataframe(view[[c for c in cols if c in view.columns]], width="stretch", hide_index=True, height=420,
                 column_config={"Confidence": st.column_config.NumberColumn(format="%.2f")})
    st.download_button("Export filtered observations (CSV)", view.to_csv(index=False).encode("utf-8"),
                       "evidence_repository.csv", "text/csv", key="btn_dl_repo_csv")


def _record(obs: pd.DataFrame) -> None:
    """Full evidence record: AI output, investigator decision, provenance and graph links, kept separate."""
    ids = obs["Observation ID"].tolist()
    oid = st.selectbox("Observation", ids, key="rep_record_pick",
                       format_func=lambda i: f"{i} · {obs.loc[obs['Observation ID'] == i, 'Object'].iloc[0]} · "
                                             f"{obs.loc[obs['Observation ID'] == i, 'Camera ID'].iloc[0]}")
    row = obs[obs["Observation ID"] == oid].iloc[0]
    src = db_manager.get_source_by_id(row["Source ID"]) or {}
    left, right = st.columns([1, 2])
    with left:
        path = row.get("Evidence Path") or ""
        if path and Path(path).is_file() and Path(path).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            st.image(path, width="stretch")
    with right:
        st.markdown("**AI finding** (as produced by the model; never edited)")
        st.dataframe(pd.DataFrame([
            ("Evidence ID", oid), ("Type", row["Modality"]), ("Object / content", row["Object"]),
            ("Timestamp", f"{row['Timestamp']} ({row['Timestamp (seconds)']} s)" if pd.notna(row["Timestamp (seconds)"]) else row["Timestamp"]),
            ("Confidence", fmt_pct(row["Confidence"])), ("Track", row.get("Track ID") or "-"),
            ("Bounding box", row.get("Bounding Box") or "-"), ("Description", row["Observation"]),
        ], columns=["Field", "Value"]), hide_index=True, width="stretch")
        st.markdown("**Investigator decision**")
        st.dataframe(pd.DataFrame([
            ("Status", row["Review Status"]), ("Notes", row.get("Review Notes") or "-"),
            ("Reviewed at", row.get("Reviewed At") or "-"),
        ], columns=["Field", "Value"]), hide_index=True, width="stretch")
        st.markdown("**Provenance**")
        st.dataframe(pd.DataFrame([
            ("Original file", src.get("source_name", "-")), ("Source ID", row["Source ID"]),
            ("Camera / device", row["Camera ID"]), ("File SHA-256", src.get("sha256_hash", "-")),
            ("Uploaded", src.get("upload_time", "-")), ("Recording start", src.get("recorded_start") or "Not set"),
            ("Processing status", src.get("processing_status") or "-"), ("Derived image", path or "-"),
            ("Recorded in case", row.get("Created At") or "-"),
        ], columns=["Field", "Value"]), hide_index=True, width="stretch")

    eg = get_evidence_graph()
    node = f"obs:{oid}"
    if node in eg.graph:
        G = eg.graph
        links = [(v, d) for _, v, d in G.out_edges(node, data=True)] + [(u, d) for u, _, d in G.in_edges(node, data=True)]
        st.markdown("**Linked events and related evidence**")
        rows = [{"Relationship": d["relation"], "Item": G.nodes[o].get("label", o), "Type": G.nodes[o]["type"],
                 "Confidence": f"{d['confidence']:.0%}", "Origin": d["origin"], "Reasoning": d["reasoning"]}
                for o, d in links if d["relation"] != "DETECTED_IN"]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("No relationships beyond its source file.")
        trk = G.nodes[node].get("track_id")
        trk_node = f"trk:{row['Source ID']}:{trk}"
        if trk_node in G and G.nodes[trk_node]["type"] in S.ENTITY_TYPES:
            st.caption(f"Part of anonymous track {S.short_track(trk)} with "
                       f"{G.nodes[trk_node].get('observation_count')} sightings.")


def render():
    page_header("Evidence repository",
                "Every AI observation with its source file, hash, timing, investigator decision and graph links.",
                eyebrow="Evidence")
    obs = db_manager.get_all_observations_df()
    if obs.empty:
        st.info("No observations yet. Run CCTV, image or audio analysis.")
        st.page_link("ui/pages/cctv_analysis.py", label="CCTV analysis", icon=":material/videocam:")
        return
    t_browse, t_record, t_search = st.tabs(["Browse", "Evidence record", "Keyword search"])
    with t_browse:
        _browse(obs)
    with t_record:
        _record(obs)
    with t_search:
        render_evidence_search(obs)


if __name__ == "__main__":
    render()
