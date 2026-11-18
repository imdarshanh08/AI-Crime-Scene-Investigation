"""Investigation dashboard: case status at a glance. Every figure is read from the case data."""

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import REPORTS_DIR
from services.evidence_graph import analysis as A
from services.evidence_graph import schema as S
from services.media import media_processor
from storage import db_manager
from ui.components import page_header, workflow_strip
from ui.state import get_evidence_graph


def _integrity_audit(sources):
    results, verified, tampered, missing = [], 0, 0, 0
    for s in sources:
        fpath = Path(s["file_path"])
        stored = (s["sha256_hash"] or "").lower()
        if not fpath.exists():
            status, live = "Missing on disk", ""
            missing += 1
        else:
            try:
                _, live = media_processor.hash_file_stream(fpath)
                if live.lower() == stored:
                    status = "Verified"
                    verified += 1
                else:
                    status = "Hash mismatch"
                    tampered += 1
            except OSError as e:
                status, live = f"Read error: {e}", ""
                tampered += 1
        results.append({"Camera / ID": s["camera_id"], "File": s["source_name"], "Modality": s["modality"],
                        "Registered SHA-256": stored, "Computed SHA-256": live, "Result": status})
    summary = f"Total {len(sources)} | verified {verified} | mismatched {tampered} | missing {missing}"
    db_manager.log_investigator_action("Cryptographic integrity audit", "EvidenceSource", "ALL", summary)
    return {"results": results, "verified": verified, "tampered": tampered, "missing": missing,
            "total": len(sources), "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def render():
    page_header("Investigation dashboard", "Current state of the case: evidence intake, analysis, review and reporting.",
                eyebrow="Overview")

    sources = db_manager.get_all_sources()
    obs = db_manager.get_all_observations_df()
    eg = get_evidence_graph()
    summary = eg.summary()

    cctv = [s for s in sources if s["modality"] == "CCTV"]
    processed = sum(1 for s in cctv if s.get("processing_status") == "Processed")
    n_obs = 0 if obs.empty else len(obs)
    visual = 0 if obs.empty else int(obs["Modality"].isin(["CCTV", "Photo"]).sum())
    audio_segments = 0 if obs.empty else int((obs["Modality"] == "Audio").sum())
    pending = 0 if obs.empty else int((obs["Review Status"] == "Pending").sum())
    reviewed = n_obs - pending
    events = summary["node_types"].get(S.EVENT, 0)
    evidential = sum(c for rel, c in summary["relations"].items() if rel in S.EVIDENTIAL_RELATIONS)
    reports = len(list(REPORTS_DIR.glob("*.pdf")))

    workflow_strip([
        ("Evidence", len(sources), "Upload files", len(sources) > 0),
        ("Analysed", f"{processed}/{len(cctv)}" if cctv else "0", "Run CCTV analysis", bool(cctv) and processed == len(cctv)),
        ("Events", events, "Reconstruct events", events > 0),
        ("Relationships", evidential, "Build the graph", evidential > 0),
        ("Reviewed", f"{reviewed}/{n_obs}" if n_obs else "0", "Review AI findings", n_obs > 0 and pending == 0),
        ("Reports", reports, "Generate a report", reports > 0),
    ])

    m = st.columns(7)
    m[0].metric("Evidence files", len(sources))
    m[1].metric("CCTV processed", f"{processed} / {len(cctv)}")
    m[2].metric("Objects detected", visual)
    m[3].metric("Events extracted", events)
    m[4].metric("Audio segments", audio_segments)
    m[5].metric("Relationships", evidential, help="Evidential graph relationships (registry links excluded).")
    m[6].metric("Pending review", pending)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Key evidence")
        st.caption("Ranked by betweenness centrality: items that connect otherwise separate parts of the case.")
        key = A.key_evidence(eg.graph, top_n=6)
        if key:
            st.dataframe(pd.DataFrame([{"Item": k["label"], "Type": k["type"], "Bridging score": k["betweenness"],
                                        "Connections": k["degree"], "Review": k["status"]} for k in key]),
                         hide_index=True, width="stretch")
        else:
            st.info("No evidential relationships yet. Analyse footage and reconstruct events to populate the graph.")

        if eg.contradictions:
            st.warning(f"{len(eg.contradictions)} contradiction(s) detected between pieces of evidence. "
                       "See Incident Reconstruction or the Evidence Graph.")

        st.subheader("Evidence by camera")
        if not obs.empty:
            by_cam = (obs.groupby(["Camera ID", "Modality"]).size().unstack(fill_value=0)
                      .reset_index().rename(columns={"Camera ID": "Camera / channel"}))
            st.dataframe(by_cam, hide_index=True, width="stretch")
        else:
            st.caption("No observations yet.")

    with right:
        st.subheader("Recent activity")
        logs = db_manager.get_audit_logs(limit=10)
        if logs.empty:
            st.caption("No activity recorded yet.")
        else:
            st.dataframe(logs[["timestamp", "action", "details"]].rename(columns=str.title),
                         hide_index=True, width="stretch", height=380)

    st.divider()
    st.subheader("Chain of custody")
    st.caption("SHA-256 recorded at intake. Re-hashing from disk detects any later modification of the stored files.")
    if sources:
        st.dataframe(pd.DataFrame([{
            "Camera / ID": s["camera_id"], "File": s["source_name"], "Modality": s["modality"],
            "Status": s.get("processing_status") or "Registered", "Uploaded": s["upload_time"],
            "Recording start": s.get("recorded_start") or "Not set", "SHA-256": s["sha256_hash"],
        } for s in sources]), hide_index=True, width="stretch")
        if st.button("Verify integrity of all evidence files", key="btn_run_integrity_audit"):
            with st.spinner("Re-computing SHA-256 for every stored file..."):
                st.session_state["last_integrity_audit"] = _integrity_audit(sources)
        audit = st.session_state.get("last_integrity_audit")
        if audit:
            if audit["tampered"] == 0 and audit["missing"] == 0:
                st.success(f"All {audit['total']} files match their registered hashes (checked {audit['timestamp']}).")
            else:
                st.error(f"{audit['tampered']} mismatched and {audit['missing']} missing file(s) "
                         f"(checked {audit['timestamp']}).")
            st.dataframe(pd.DataFrame(audit["results"]), hide_index=True, width="stretch")
    else:
        st.info("No evidence registered yet.")
        st.page_link("ui/pages/evidence_upload.py", label="Upload evidence", icon=":material/upload:")


if __name__ == "__main__":
    render()
