"""TRACE-AI Dual Workstation Module.
Inspired by the user's reference investigation design:
- Left Pane: Multi-Camera CCTV Split Matrix (Synchronized 2x2 or 2x3 video/frame grid with bounding box tags)
- Right Pane: Live Interactive Evidence Graph + Chronological Incident Progression + Forensic Verification Bar
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import db_manager
import evidence_graph_builder
import media_processor

ROOT = Path(__file__).resolve().parent
ANNOTATED_VIDEOS_DIR = ROOT / "outputs" / "annotated_videos"
EVIDENCE_FRAMES_DIR = ROOT / "outputs" / "evidence_frames"


def render_trace_ai_workstation():
    """Render the dual-pane TRACE-AI Forensic Workstation."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(8, 15, 30, 0.98) 100%); 
                padding: 14px 22px; 
                border-radius: 10px; 
                border: 1px solid rgba(56, 189, 248, 0.3); 
                border-left: 5px solid #0284c7; 
                margin-bottom: 15px; 
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 20px; font-weight: 800; background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 1px;">
                    TRACE-AI WORKSTATION
                </span>
                <span style="font-size: 13px; color: #94a3b8; border-left: 1px solid #334155; padding-left: 12px;">
                    Multi-Camera Command Matrix & Real-Time Incident Corroboration
                </span>
            </div>
            <div>
                <span style="background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%); color: white; padding: 5px 12px; border-radius: 6px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; border: 1px solid rgba(56, 189, 248, 0.5);">
                    ● LIVE INCIDENT MONITORING
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h4 style="margin: 0; color: #38bdf8;">📹 Multi-Camera CCTV Grid</h4>
            <span style="color: #64748b; font-size: 12px;">Active Synchronized Feeds</span>
        </div>
        """, unsafe_allow_html=True)

        view_mode = st.radio(
            "Matrix Display Mode",
            ["🎬 Annotated Video Feeds (H.264)", "🖼️ Forensic Keyframes Matrix"],
            horizontal=True,
            key="trace_view_mode",
            label_visibility="collapsed"
        )

        display_cams = cctv_sources[:4] if len(cctv_sources) >= 4 else cctv_sources

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
                        src_id = src["source_id"]

                        # Check for matching annotated video
                        ann_candidates = list(ANNOTATED_VIDEOS_DIR.glob(f"{src_id}*_annotated.mp4"))
                        ann_path = str(ann_candidates[0]) if ann_candidates else ""

                        st.markdown(f"""
                        <div style="background: #111827; border: 1px solid #1e293b; border-radius: 6px; padding: 6px; margin-bottom: 8px;">
                            <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: bold; color: #e2e8f0; margin-bottom: 4px;">
                                <span>📹 {cam_id}</span>
                                <span style="color: #0284c7;">Target: Subject A / Object X</span>
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
                                st.image(t_path, use_container_width=True)
                            elif Path(src["file_path"]).suffix.lower() == ".mp4":
                                with open(src["file_path"], "rb") as f_raw:
                                    st.video(f_raw.read(), format="video/mp4")
                            else:
                                st.info(f"{cam_id} Video Stream")

                        st.caption(f"{src_name[:24]} | {src['duration_seconds']:.1f}s")

    # =========================================================
    # RIGHT PANE: LIVE EVIDENCE GRAPH + INCIDENT PROGRESSION
    # =========================================================
    with col_graph:
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h4 style="margin: 0; color: #38bdf8;">🕸️ Evidence Graph & Incident Flow</h4>
            <span style="color: #10b981; font-size: 12px; font-weight: bold;">● Active Graph Synthesis</span>
        </div>
        """, unsafe_allow_html=True)

        # 1. Interactive Evidence Graph (PyVis)
        with st.spinner("Synthesizing Graph..."):
            G, graph_html, graph_intel = evidence_graph_builder.build_advanced_evidence_graph(
                observations_df=all_obs_df,
                layout_mode="Force-Directed Physics (Cluster)",
                enable_trajectory_inference=True,
                enable_audio_visual_fusion=True,
                max_nodes=80
            )

        components.html(graph_html, height=420, scrolling=False)

        # 2. Chronological Incident Flow (Matching reference video list)
        st.markdown("##### 📁 Chronological Incident Progression")

        timeline_items = []
        if not all_obs_df.empty:
            sorted_obs = all_obs_df.sort_values("Timestamp (seconds)", na_position="last").head(8)
            for _, r in sorted_obs.iterrows():
                t_str = r.get("Timestamp", "00:00")
                obj = r.get("Object", "Target")
                cam = r.get("Camera ID", "CAM")
                obs_text = r.get("Observation", f"{obj} detected")
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
                        <span style="color: {icon_color}; margin-right: 8px;">📁</span>
                        <span style="color: #38bdf8; font-weight: bold; width: 75px;">{item['time']}</span>
                        <span style="color: #f1f5f9; flex-grow: 1;">{item['desc']}</span>
                        <span style="color: #64748b; font-size: 11px;">[{item['status']}]</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No progression logs recorded yet.")

        # 3. Forensic Human Verification Action Bar (Exact match to reference [CONFIRMED] [NEEDS REVIEW] [INCORRECT])
        st.markdown("##### 🧑‍⚖️ Human-in-the-Loop Verification Bar")

        if not all_obs_df.empty:
            obs_dict = {f"{r['Object']} ({r['Camera ID']} @ {r['Timestamp']}) [{r['Observation ID'][:6]}]": r for _, r in all_obs_df.iterrows()}
            selected_key = st.selectbox("Target Observation", list(obs_dict.keys()), key="ws_target_obs", label_visibility="collapsed")
            target_item = obs_dict[selected_key]
            conf_val = target_item.get("Confidence", 0.85)
            conf_pct = int(conf_val * 100) if conf_val is not None and not pd.isna(conf_val) else 85

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; background: #0f172a; padding: 6px 12px; border-radius: 6px; margin-bottom: 10px; border: 1px solid #1e293b;">
                <span style="font-size: 13px; color: #94a3b8;">AI Observation Confidence:</span>
                <span style="font-size: 14px; font-weight: bold; color: #38bdf8;">{conf_pct}%</span>
            </div>
            """, unsafe_allow_html=True)

            b_col1, b_col2, b_col3 = st.columns(3)

            with b_col1:
                if st.button("✅ CONFIRMED", key=f"ws_btn_confirm_{target_item['Observation ID']}", type="primary", use_container_width=True):
                    db_manager.update_observation_review(target_item["Observation ID"], "Approved", "Confirmed via TRACE-AI Live Workstation")
                    st.success("Observation confirmed!")
                    st.rerun()

            with b_col2:
                if st.button("⚠️ NEEDS REVIEW", key=f"ws_btn_flag_{target_item['Observation ID']}", use_container_width=True):
                    db_manager.update_observation_review(target_item["Observation ID"], "Flagged", "Flagged for manual review via TRACE-AI Workstation")
                    st.warning("Observation flagged for review.")
                    st.rerun()

            with b_col3:
                if st.button("❌ INCORRECT", key=f"ws_btn_reject_{target_item['Observation ID']}", use_container_width=True):
                    db_manager.update_observation_review(target_item["Observation ID"], "Rejected", "Rejected as false detection via TRACE-AI Workstation")
                    st.error("Observation marked incorrect.")
                    st.rerun()

