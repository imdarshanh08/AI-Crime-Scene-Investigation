"""Investigation reports and data export."""

import html
from pathlib import Path

import streamlit as st

from services.events.event_reconstruction import get_reconstructed_events
from services.reports import report_generator
from storage import db_manager
from ui.components import page_header
from ui.state import get_evidence_graph


def render():
    page_header(
        "Investigation reports",
        "PDF dossier with evidence inventory, chronology, graph relationships, supporting and contradicting evidence, "
        "and a clear split between investigator-confirmed and AI-generated findings.",
        eyebrow="Reporting",
    )

    col_rep_meta, col_rep_actions = st.columns([2, 1.5])

    with col_rep_meta:
        rep_title = st.text_input("Case title", placeholder="e.g. Theft at North Gate, 6 Nov 2026", key="txt_rep_title")
        rep_id = st.text_input("Case reference", placeholder="e.g. CR-2026-0142", key="txt_rep_id")
        rep_officer = st.text_input("Lead investigator", placeholder="Name and unit", key="txt_rep_officer")
        rep_notes = st.text_area("Investigator summary", placeholder="Your summary of the case (optional)",
                                 key="txt_rep_notes")

    with col_rep_actions:
        st.subheader("Generate PDF")
        missing = [n for n, v in (("case title", rep_title), ("case reference", rep_id), ("lead investigator", rep_officer))
                   if not v.strip()]
        if missing:
            st.caption("Required: " + ", ".join(missing) + ".")
        if st.button("Generate PDF report", type="primary", key="btn_gen_pdf_report", disabled=bool(missing)):
            with st.spinner("Compiling dossier and generating PDF with ReportLab..."):
                sources = db_manager.get_all_sources()
                all_obs = db_manager.get_all_observations_df()
                events = get_reconstructed_events(active_only=True)
                logs = db_manager.get_audit_logs(limit=None)

                try:
                    pdf_path = report_generator.generate_pdf_report(
                        case_title=rep_title,
                        case_id=rep_id,
                        investigator_name=rep_officer,
                        sources=sources,
                        observations_df=all_obs,
                        events_df=events,
                        audit_logs_df=logs,
                        notes=rep_notes,
                        evidence_graph=get_evidence_graph(),
                    )
                except Exception as e:
                    st.error(f"Report generation failed: {e}")
                else:
                    st.session_state.last_report_path = pdf_path
                    db_manager.log_investigator_action(
                        action="Generated PDF report",
                        entity_type="Report",
                        entity_id=rep_id,
                        details=Path(pdf_path).name
                    )
                    st.success("Report generated.")

        if st.session_state.last_report_path and Path(st.session_state.last_report_path).is_file():
            with open(st.session_state.last_report_path, "rb") as f_pdf:
                st.download_button(
                    "Download PDF report",
                    f_pdf.read(),
                    file_name=Path(st.session_state.last_report_path).name,
                    mime="application/pdf",
                    key="btn_dl_final_pdf"
                )

        st.divider()
        st.subheader("Data export")
        all_obs = db_manager.get_all_observations_df()
        if not all_obs.empty:
            st.download_button(
                "Export all observations (CSV)",
                all_obs.to_csv(index=False).encode("utf-8"),
                file_name=f"observations_{rep_id or 'case'}.csv",
                mime="text/csv",
                key="btn_dl_final_csv"
            )

    st.divider()
    st.subheader("CCTV summaries included in the report")
    st.caption("AI-synthesized activity overview for each camera feed, included directly in Section 3 of the PDF report.")

    sources = db_manager.get_all_sources()
    all_obs = db_manager.get_all_observations_df()
    summaries = report_generator.build_video_analytical_summaries(sources, all_obs)

    if summaries:
        for vid in summaries:
            with st.expander(f"📹 {vid['camera_id']}: {vid['source_name']} ({vid['duration_str']} | {vid['total_detections']} Detections)", expanded=False):
                c_info1, c_info2 = st.columns([1.5, 2.5])
                with c_info1:
                    st.markdown(f"""
                    - **Camera**: `{vid['camera_id']}`
                    - **Modality**: `{vid['modality']}`
                    - **Duration**: `{vid['duration_str']}`
                    - **Resolution**: `{vid['resolution_str']} @ {vid['fps_str']}`
                    - **File Size**: `{vid['size_mb_str']}`
                    - **SHA-256**: `{vid['sha256'][:16]}...`
                    """)
                with c_info2:
                    st.markdown(f"""
                    - **Total Sightings**: **{vid['total_detections']}**
                    - **Target Classes**: *{vid['class_summary']}*
                    - **Trajectories**: {vid['tracks_summary']}
                    - **Active Window**: `{vid['active_window']}`
                    - **Verification**: `{vid['status_map']['Approved']} Approved | {vid['status_map']['Pending']} Pending | {vid['status_map']['Rejected']} Rejected`
                    """)
                st.info(html.unescape(vid["narrative"].replace("<b>", "**").replace("</b>", "**").replace("<i>", "*").replace("</i>", "*")))
    else:
        st.info("No CCTV video sources registered yet.")


if __name__ == "__main__":
    render()
