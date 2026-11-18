"""TRACE-AI Dual Workstation Module.
Inspired by the user's reference investigation design:
- Left Pane: Multi-Camera CCTV Split Matrix (Synchronized 2x2 or 2x3 video/frame grid with bounding box tags)
- Right Pane: Live Interactive Evidence Graph + Chronological Incident Progression + Forensic Verification Bar
"""

import html
from pathlib import Path

import pandas as pd
import streamlit as st
import base64 as _b64

from storage import db_manager
from services.evidence_graph import schema as S
from services.evidence_graph.visualization import render_graph_html
from services.media import media_processor
from core.config import ROOT  # noqa: E402
ANNOTATED_VIDEOS_DIR = ROOT / "outputs" / "annotated_videos"
EVIDENCE_FRAMES_DIR = ROOT / "outputs" / "evidence_frames"


def render_trace_ai_workstation():
    """Render the dual-pane TRACE-AI Forensic Workstation."""

    all_obs_df = db_manager.get_all_observations_df()
    cctv_sources = db_manager.get_all_sources("CCTV")

    if not cctv_sources and all_obs_df.empty:
        st.info("No CCTV footage or evidence recorded yet. Please load sample videos or upload recordings to launch the TRACE-AI Workstation.")
        return

    # Two Main Columns: Left = Multi-Camera Matrix, Right = Evidence Graph & Incident Flow
    col_matrix, col_graph = st.columns([1.15, 1.0], gap="medium")

    # =========================================================
    # LEFT PANE: MULTI-CAMERA CCTV SPLIT MATRIX
    # =========================================================
    with col_matrix:
        st.markdown("**Camera feeds**")

        view_mode = st.radio(
            "Matrix Display Mode",
            ["🎬 Annotated Video Feeds (H.264)", "🖼️ Forensic Keyframes Matrix"],
            horizontal=True,
            key="trace_view_mode",
            label_visibility="collapsed"
        )

        # Previously only the first four cameras were ever shown, with no way to pick others.
        if len(cctv_sources) > 4:
            cam_labels = {f"{s['camera_id']} - {s['source_name']}": s for s in cctv_sources}
            chosen = st.multiselect(
                "Cameras in grid (max 4)",
                list(cam_labels.keys()),
                default=list(cam_labels.keys())[:4],
                max_selections=4,
                key="trace_grid_cameras"
            )
            display_cams = [cam_labels[c] for c in chosen]
        else:
            display_cams = cctv_sources

        if not display_cams:
            st.info("No active camera recordings available.")
        else:
            # 2x2 Grid Layout
            for row_idx in range(0, len(display_cams), 2):
                grid_c1, grid_c2 = st.columns(2)
                cam_pair = display_cams[row_idx:row_idx + 2]

                for col_widget, src in zip([grid_c1, grid_c2], cam_pair):
                    with col_widget:
                        cam_id = src["camera_id"]
                        src_name = src["source_name"]
                        cam_obs = 0
                        if not all_obs_df.empty and "Source ID" in all_obs_df.columns:
                            cam_obs = int((all_obs_df["Source ID"] == src["source_id"]).sum())
                        src_id = src["source_id"]

                        # Check for matching annotated video
                        ann_candidates = list(ANNOTATED_VIDEOS_DIR.glob(f"{src_id}*_annotated.mp4"))
                        ann_path = str(ann_candidates[0]) if ann_candidates else ""

                        st.markdown(f"""
                        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 6px; padding: 6px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: bold; color: #e2e8f0; margin-bottom: 4px;">
                                <span>📹 {html.escape(str(cam_id))}</span>
                                <span style="color: #38bdf8;">{cam_obs} observation(s)</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if "Video" in view_mode and ann_path and Path(ann_path).is_file():
                            web_ann = media_processor.convert_to_web_h264(ann_path)
                            with open(web_ann, "rb") as f_v:
                                st.video(f_v.read(), format="video/mp4")
                        else:
                            # Show thumbnail keyframe
                            meta = media_processor.get_video_metadata(src["file_path"])
                            t_path = meta.get("thumbnail_path")
                            if t_path and Path(t_path).is_file():
                                st.image(t_path, width="stretch")
                            elif Path(src["file_path"]).suffix.lower() == ".mp4" and Path(src["file_path"]).is_file():
                                # Pass the path so Streamlit streams the file instead of loading it all into memory.
                                st.video(src["file_path"])
                            else:
                                st.info(f"{cam_id} Video Stream")

                        st.caption(f"{src_name[:24]} | {src['duration_seconds']:.1f}s")

    # =========================================================
    # RIGHT PANE: LIVE EVIDENCE GRAPH + INCIDENT PROGRESSION
    # =========================================================
    with col_graph:
        st.markdown("**Evidence graph for these cameras**")

        # 1. Interactive Evidence Graph (PyVis)
        with st.spinner("Synthesizing Graph..."):
            from ui.state import get_evidence_graph

            eg = get_evidence_graph()
            shown_sources = {c["source_id"] for c in display_cams}
            ws_nodes = {n for n, d in eg.graph.nodes(data=True)
                        if d["type"] != S.OBSERVATION and (d.get("source_id") in shown_sources
                                                          or d["type"] in (S.CAMERA, S.LOCATION))}
            graph_html = render_graph_html(eg, ws_nodes, height_px=400)

        st.iframe("data:text/html;base64," + _b64.b64encode(graph_html.encode("utf-8")).decode("ascii"), height=420)

        # 2. Chronological Incident Flow (Matching reference video list)
        st.markdown("**Earliest observations**")

        timeline_items = []
        if not all_obs_df.empty:
            sorted_obs = all_obs_df.sort_values("Timestamp (seconds)", na_position="last").head(8)
            for _, r in sorted_obs.iterrows():
                t_str = r.get("Timestamp", "00:00")
                obj = r.get("Object", "Target")
                cam = r.get("Camera ID", "CAM")
                status = r.get("Review Status", "Pending")
                timeline_items.append({
                    "time": t_str,
                    "desc": f"{obj} detected in {cam}",
                    "id": r.get("Observation ID"),
                    "status": status,
                    "obj": obj,
                    "cam": cam
                })

        with st.container(height=180, border=True):
            if timeline_items:
                for item in timeline_items:
                    icon_color = "#22c55e" if item["status"] == "Approved" else ("#ef4444" if item["status"] == "Rejected" else "#f59e0b")
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; font-size: 13px; padding: 3px 0; border-bottom: 1px solid #1e293b;">
                        <span style="color: {icon_color}; margin-right: 8px;">●</span>
                        <span style="color: #38bdf8; font-weight: bold; width: 75px;">{html.escape(str(item['time']))}</span>
                        <span style="color: #f1f5f9; flex-grow: 1;">{html.escape(str(item['desc']))}</span>
                        <span style="color: #64748b; font-size: 11px;">[{html.escape(str(item['status']))}]</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No progression logs recorded yet.")

        # 3. Forensic Human Verification Action Bar (Exact match to reference [CONFIRMED] [NEEDS REVIEW] [INCORRECT])
        st.markdown("**Quick review**")

        if not all_obs_df.empty:
            obs_dict = {f"{r['Object']} ({r['Camera ID']} @ {r['Timestamp']}) [{str(r['Observation ID'])[:8]}]": r for _, r in all_obs_df.iterrows()}
            selected_key = st.selectbox("Target Observation", list(obs_dict.keys()), key="ws_target_obs", label_visibility="collapsed")
            target_item = obs_dict[selected_key]
            conf_val = target_item.get("Confidence")
            # No invented fallback: show N/A when the model gave no confidence.
            conf_txt = f"{float(conf_val):.0%}" if conf_val is not None and not pd.isna(conf_val) else "N/A"

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background: #0f172a; padding: 6px 12px; border-radius: 6px; margin-bottom: 10px; border: 1px solid #1e293b;">
                <span style="font-size: 13px; color: #94a3b8;">AI Observation Confidence:</span>
                <span style="font-size: 14px; font-weight: bold; color: #38bdf8;">{conf_txt}</span>
            </div>
            <div style="font-size: 12px; color: #94a3b8; margin: -4px 0 10px 2px;">
                Current status: <b style="color:#e2e8f0">{html.escape(str(target_item.get("Review Status", "Pending")))}</b>
            </div>
            """, unsafe_allow_html=True)

            b_col1, b_col2, b_col3 = st.columns(3)

            with b_col1:
                if st.button("✅ CONFIRMED", key=f"ws_btn_confirm_{target_item['Observation ID']}", type="primary", width="stretch"):
                    db_manager.update_observation_review(target_item["Observation ID"], "Approved", "Confirmed via TRACE-AI Workstation")
                    st.toast("Observation confirmed.")
                    st.rerun()

            with b_col2:
                if st.button("⚠️ NEEDS REVIEW", key=f"ws_btn_flag_{target_item['Observation ID']}", width="stretch"):
                    db_manager.update_observation_review(target_item["Observation ID"], "Flagged", "Flagged for manual review via TRACE-AI Workstation")
                    st.toast("Observation flagged for review.")
                    st.rerun()

            with b_col3:
                if st.button("❌ INCORRECT", key=f"ws_btn_reject_{target_item['Observation ID']}", width="stretch"):
                    db_manager.update_observation_review(target_item["Observation ID"], "Rejected", "Rejected as false detection via TRACE-AI Workstation")
                    st.toast("Observation marked incorrect.")
                    st.rerun()

