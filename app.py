"""AI CRIME SCENE & CCTV INVESTIGATION ASSISTANT
Subtitle: Multimodal AI for Evidence Linking and Incident Reconstruction

Architecture:
- Frontend: Streamlit (Dark Forensic Theme, 12 Workflow Tabs)
- Vision: YOLOv8 Nano (Ultralytics) + ByteTrack Tracking
- Audio: Faster-Whisper (Transcription & Translation) + PyAV
- Data & Storage: SQLite Persistent Master Database (db_manager.py)
- Graph: NetworkX & PyVis Multimodal Knowledge Graph
- Reporting: ReportLab Forensic PDF Report Generator
- Hardware Target: ASUS TUF A15 / NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) / Windows 11
"""

import hashlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch

# Ensure project root is in system path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Internal modules
import db_manager
import media_processor
import yolo_detector
import whisper_transcriber
import evidence_graph_builder
import report_generator
import trace_workstation

# Import existing specialized modules
from event_reconstruction import (
    reconstruct_events,
    get_reconstructed_events,
    normalise_observations
)
import cross_camera_linking
import evidence_search_engine
import unified_evidence_timeline
import cross_modal_correlation

# =========================================================
# 1. STREAMLIT PAGE CONFIGURATION & FORENSIC THEME
# =========================================================

st.set_page_config(
    page_title="AI Crime Scene & CCTV Investigation Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-contrast forensic styling and attractive command center aesthetic
st.markdown("""
<style>
    /* Deep Cyber-Forensic Radial Gradient Background */
    .stApp, [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 50% 0%, #0d1e38 0%, #080f1e 40%, #03060d 100%) !important;
        background-attachment: fixed !important;
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Subtle High-Tech Hex/Grid Overlay */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background-image: 
            linear-gradient(rgba(56, 189, 248, 0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(56, 189, 248, 0.025) 1px, transparent 1px);
        background-size: 44px 44px;
        pointer-events: none;
        z-index: 0;
    }

    /* Main Content Area */
    .main .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 96% !important;
    }

    /* Header & Toolbar transparency */
    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Obsidian Command Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070d18 0%, #040810 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
    }

    /* Glassmorphism KPI Metrics */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.65) 0%, rgba(15, 23, 42, 0.85) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.22) !important;
        border-radius: 10px !important;
        padding: 14px 16px !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35) !important;
        transition: transform 0.2s ease, border-color 0.2s ease !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.15) !important;
    }
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
        font-size: 1.8rem !important;
        letter-spacing: -0.5px;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
    }

    /* Sleek Forensic Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px !important;
        background: rgba(15, 23, 42, 0.65) !important;
        padding: 6px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px !important;
        white-space: pre-wrap !important;
        background-color: transparent !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        color: #94a3b8 !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        border: 1px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0 !important;
        background: rgba(56, 189, 248, 0.08) !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.45) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
    }

    /* Primary Buttons */
    .stButton > button[kind="primary"], .stButton > button[type="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.5) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover, .stButton > button[type="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.55) !important;
    }
    .stButton > button[kind="secondary"], .stButton > button[type="secondary"] {
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }

    /* Glassmorphism Expanders */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(56, 189, 248, 0.15) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #070d18;
    }
    ::-webkit-scrollbar-thumb {
        background: #1e3a5f;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #0284c7;
    }

    /* Status Badges */
    .status-badge-approved {
        background-color: #15803d;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .status-badge-pending {
        background-color: #b45309;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .status-badge-rejected {
        background-color: #b91c1c;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Required project directories
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
EVIDENCE_DIR = OUTPUT_DIR / "evidence_frames"
ANNOTATED_VIDEOS_DIR = OUTPUT_DIR / "annotated_videos"
PHOTO_EVIDENCE_DIR = OUTPUT_DIR / "photo_evidence"
AUDIO_EVIDENCE_DIR = OUTPUT_DIR / "audio_evidence"
REPORT_DIR = OUTPUT_DIR / "reports"

for folder in [UPLOAD_DIR, OUTPUT_DIR, EVIDENCE_DIR, ANNOTATED_VIDEOS_DIR, PHOTO_EVIDENCE_DIR, AUDIO_EVIDENCE_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. SESSION STATE MANAGEMENT & PERSISTENT SYNC
# =========================================================

def sync_session_from_database():
    """Load persistent records from SQLite into Streamlit session state."""
    db_sources = db_manager.get_all_sources()
    st.session_state.video_records = {
        s["source_id"]: s for s in db_sources if s["modality"] == "CCTV"
    }
    st.session_state.photo_records = {
        s["source_id"]: s for s in db_sources if s["modality"] == "Photo"
    }
    st.session_state.audio_records = {
        s["source_id"]: s for s in db_sources if s["modality"] == "Audio"
    }

    obs_df = db_manager.get_all_observations_df()
    if not obs_df.empty:
        st.session_state.detection_rows = obs_df[obs_df["Modality"] == "CCTV"].to_dict(orient="records")
        st.session_state.photo_rows = obs_df[obs_df["Modality"] == "Photo"].to_dict(orient="records")
        st.session_state.audio_rows = obs_df[obs_df["Modality"] == "Audio"].to_dict(orient="records")
    else:
        st.session_state.detection_rows = []
        st.session_state.photo_rows = []
        st.session_state.audio_rows = []


if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.video_records = {}
    st.session_state.photo_records = {}
    st.session_state.audio_records = {}
    st.session_state.detection_rows = []
    st.session_state.photo_rows = []
    st.session_state.audio_rows = []
    st.session_state.preview_images = []
    st.session_state.last_report_path = ""
    st.session_state.active_annotated_video = ""
    sync_session_from_database()

# =========================================================
# 3. SIDEBAR: SYSTEM DIAGNOSTICS & CONTROL PANEL
# =========================================================

device_id, device_str = yolo_detector.get_optimal_device()

with st.sidebar:
    st.title("🛡️ Forensic Control")
    st.caption("AI Crime Scene & CCTV Assistant")
    st.divider()

    st.subheader("🖥️ Hardware & AI Engine")
    st.write(f"• **Device:** {device_str}")
    st.write(f"• **Vision Model:** YOLOv8 Nano (CUDA)")
    st.write(f"• **Tracker:** ByteTrack (Multi-object)")
    st.write(f"• **Audio Engine:** Faster-Whisper (CPU int8)")
    st.write(f"• **Graph Engine:** NetworkX + PyVis")
    st.write(f"• **Database:** SQLite WAL (Persistent)")

    if torch.cuda.is_available():
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        st.success(f"GPU Active: {torch.cuda.get_device_name(0)} ({vram_mb:.0f} MB VRAM)")

    st.divider()

    # Case Metrics
    st.subheader("📊 Case Metrics")
    cctv_cnt = len(st.session_state.video_records)
    photo_cnt = len(st.session_state.photo_records)
    audio_cnt = len(st.session_state.audio_records)
    total_obs = (
        len(st.session_state.detection_rows) +
        len(st.session_state.photo_rows) +
        len(st.session_state.audio_rows)
    )
    st.write(f"📹 **CCTV Cameras/Files:** {cctv_cnt}")
    st.write(f"📷 **Scene Photographs:** {photo_cnt}")
    st.write(f"🎙️ **Audio Recordings:** {audio_cnt}")
    st.write(f"🎯 **Total Observations:** {total_obs}")

    st.divider()

    # Archived Sample Loader
    archive_dir = ROOT / "archived_videos"
    if archive_dir.exists() and any(archive_dir.glob("*.mp4")):
        if st.button("📂 Load Archived Sample Videos", key="btn_load_archived_samples", help="Copy sample videos into uploads"):
            for sample_file in archive_dir.glob("*.mp4"):
                target = UPLOAD_DIR / sample_file.name
                if not target.exists():
                    shutil.copy2(sample_file, target)
                meta = media_processor.get_video_metadata(str(target))
                short_id, full_sha256 = media_processor.hash_file_stream(target)
                cam_id = f"CAM-0{len(st.session_state.video_records) + 1}"
                db_manager.save_source_record(
                    source_id=short_id,
                    source_name=sample_file.name,
                    file_path=str(target),
                    modality="CCTV",
                    camera_id=cam_id,
                    file_size_bytes=target.stat().st_size,
                    sha256_hash=full_sha256,
                    duration_seconds=meta["duration_seconds"],
                    fps=meta["fps"],
                    total_frames=meta["total_frames"],
                    width=meta["width"],
                    height=meta["height"]
                )
            sync_session_from_database()
            st.success("Sample CCTV videos registered!")
            st.rerun()

    # Case Reset Utility
    with st.expander("⚠️ Case Management & Reset", expanded=False):
        st.write("Reset current case observations or all registered sources.")
        if st.button("🗑️ Clear Observations Only", key="btn_clear_obs", type="secondary"):
            db_manager.clear_all_observations()
            sync_session_from_database()
            st.success("All observations cleared.")
            st.rerun()

# =========================================================
# 4. APPLICATION BANNER & JURY EVALUATION SHOWCASE
# =========================================================

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(8, 15, 30, 0.98) 100%); 
            border: 1px solid rgba(56, 189, 248, 0.35); 
            border-left: 6px solid #0284c7; 
            border-radius: 12px; 
            padding: 18px 24px; 
            margin-bottom: 15px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.55);">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
                <span style="font-size: 26px;">🔍</span>
                <span style="font-size: 24px; font-weight: 800; background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px;">
                    AI CRIME SCENE & CCTV INVESTIGATION ASSISTANT
                </span>
                <span style="background: rgba(2, 132, 199, 0.25); color: #38bdf8; border: 1px solid #0284c7; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px;">
                    TRACE-AI v2.0
                </span>
            </div>
            <div style="color: #94a3b8; font-size: 13.5px; font-weight: 500;">
                Multimodal AI for Evidence Linking, Multi-Camera Trajectory Tracking, and Incident Reconstruction
            </div>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
            <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                ⚡ RTX 3050 CUDA ACTIVE
            </span>
            <span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                🛡️ SHA-256 VERIFIED
            </span>
            <span style="background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                ⚖️ HUMAN-IN-THE-LOOP AUDIT
            </span>
            <span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); padding: 4px 10px; border-radius: 6px; font-size: 11px; font-weight: 600;">
                🏛️ JURY EVALUATION READY
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("🏛️ Evaluation Jury Panel Guide — Recommended 5-Step Project Showcase", expanded=False):
    st.markdown("""
    Welcome, Evaluators and Jury Members. This application is architected as an **end-to-end, multimodal forensic intelligence workstation**.
    Follow this **5-Step Showcase Workflow** to review all key deliverables and core innovations:

    1. **🎬 Step 1 — TRACE-AI Workstation (Tab 0)**:
       - Split-screen command center view: Synchronized multi-camera CCTV feeds on the left, live interactive evidence graph and incident progression on the right.
       - Demonstrate 1-click **`[CONFIRMED]`** / **`[NEEDS REVIEW]`** human verification.
    2. **📊 Step 2 — Cryptographic Anti-Tamper Audit (Tab 1)**:
       - Click **`Run Cryptographic Evidence Integrity Audit`** to verify live SHA-256 hash match against the database to guarantee zero evidence tampering.
    3. **👥 Step 3 — Multi-Camera Sighting Comparison & Transit Linker (Tab 8, Subtab 3)**:
       - Select Sighting A from Camera 1 and Sighting B from Camera 2.
       - Inspect zoomed subject crops side-by-side with full frame bounding boxes and automatic **Δt transit time calculation**.
       - Click **`Record Forensic Link`** to link sightings directly into the knowledge graph.
    4. **🕸️ Step 4 — Advanced Multimodal Knowledge Graph (Tab 9)**:
       - Graph nodes embed actual 100x100 crop photographs with real-time glassmorphism HUD inspector.
       - Toggle between **Force-Directed**, **Timeline (Left-to-Right)**, and **Forensic Hierarchy** layouts.
       - Review NetworkX **Primary Evidence Hub** and **Critical Bridge** centrality analytics.
    5. **📄 Step 5 — Forensic Investigation PDF Dossier (Tab 12)**:
       - Review the auto-generated **CCTV Video Analytical Summaries** and download the publication-grade ReportLab PDF.
    """)

# =========================================================
# 5. FORENSIC WORKFLOW TABS
# =========================================================

tab_names = [
    "🎬 TRACE-AI Workstation",
    "📊 Dashboard",
    "📹 CCTV Upload",
    "🤖 Video Detection",
    "📷 Photo Evidence",
    "🎙️ Audio Intelligence",
    "🗄️ Evidence Repository",
    "⏱️ Event Reconstruction",
    "🌐 Cross-Camera Linking",
    "🕸️ Evidence Graph",
    "🔍 Investigation Search",
    "🧑‍⚖️ Verification Centre",
    "📄 Reports & Export"
]

(
    tab_trace_workstation,
    tab_dashboard,
    tab_cctv_upload,
    tab_detection,
    tab_photos,
    tab_audio,
    tab_repository,
    tab_events,
    tab_cross_camera,
    tab_graph,
    tab_search,
    tab_verification,
    tab_reports
) = st.tabs(tab_names)


# =========================================================
# TAB 0: TRACE-AI DUAL INVESTIGATION WORKSTATION
# =========================================================

with tab_trace_workstation:
    trace_workstation.render_trace_ai_workstation()


# =========================================================
# TAB 1: DASHBOARD & CASE OVERVIEW
# =========================================================

with tab_dashboard:
    st.header("📊 Investigation Dashboard & Case Summary")
    st.write("Real-time summary of digital evidence chain of custody and multimodal analysis.")

    # KPI Metrics
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("CCTV Recordings", len(st.session_state.video_records))
    kpi2.metric("Scene Photos", len(st.session_state.photo_records))
    kpi3.metric("Audio Evidence", len(st.session_state.audio_records))
    kpi4.metric("Total Detections", len(st.session_state.detection_rows))

    obs_df = db_manager.get_all_observations_df()
    approved_num = 0
    if not obs_df.empty and "Review Status" in obs_df.columns:
        approved_num = int((obs_df["Review Status"] == "Approved").sum())
    kpi5.metric("Verified Items", approved_num)

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🛡️ Digital Evidence Chain of Custody & SHA-256 Hashes")
        all_sources = db_manager.get_all_sources()
        if all_sources:
            src_display = []
            for s in all_sources:
                src_display.append({
                    "Camera ID": s["camera_id"],
                    "Source Name": s["source_name"],
                    "Modality": s["modality"],
                    "Duration": f"{s['duration_seconds']:.1f}s" if s['duration_seconds'] > 0 else "N/A",
                    "Resolution": f"{s['width']}x{s['height']}" if s['width'] > 0 else "N/A",
                    "Size (MB)": round(s["file_size_bytes"] / (1024 * 1024), 2),
                    "SHA-256 Hash": f"{s['sha256_hash'][:16]}...{s['sha256_hash'][-8:]}",
                    "Upload Time": s["upload_time"]
                })
            st.dataframe(pd.DataFrame(src_display), use_container_width=True, hide_index=True)
        else:
            st.info("No evidence files registered yet. Upload CCTV videos or photos to begin.")

    with col_right:
        st.subheader("🎯 Modality Breakdown")
        if total_obs > 0:
            breakdown_data = pd.DataFrame([
                {"Modality": "CCTV Detections", "Count": len(st.session_state.detection_rows)},
                {"Modality": "Photo Observations", "Count": len(st.session_state.photo_rows)},
                {"Modality": "Audio Transcripts", "Count": len(st.session_state.audio_rows)}
            ])
            st.dataframe(breakdown_data, use_container_width=True, hide_index=True)
        else:
            st.info("No observations recorded yet.")

        st.subheader("📋 Recent Investigator Audit Logs")
        logs_df = db_manager.get_audit_logs()
        if not logs_df.empty:
            st.dataframe(
                logs_df[["timestamp", "action", "entity_type", "details"]].head(6),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.caption("No review actions logged yet.")

    st.divider()
    st.subheader("🔐 Cryptographic Evidence Integrity & Anti-Tamper Verification")
    st.write(
        "Independently audit all stored digital evidence by recalculating streaming SHA-256 "
        "cryptographic hashes directly from disk to detect file tampering, bit rot, or unauthorized alterations."
    )

    col_btn_audit, col_status_audit = st.columns([1.5, 3])
    with col_btn_audit:
        run_audit_clicked = st.button("🔐 Run Cryptographic Evidence Integrity Audit", type="primary", key="btn_run_integrity_audit")

    if run_audit_clicked:
        with st.spinner("Computing live streaming SHA-256 hashes across all evidence files..."):
            all_sources = db_manager.get_all_sources()
            if not all_sources:
                st.warning("No registered evidence sources found to audit.")
            else:
                audit_results = []
                verified_count = 0
                tampered_count = 0
                missing_count = 0

                for s in all_sources:
                    fpath = Path(s["file_path"])
                    stored_hash = s["sha256_hash"]
                    if not fpath.exists():
                        status = "⚠️ MISSING FILE"
                        live_hash = "N/A"
                        missing_count += 1
                    else:
                        try:
                            _, live_hash = media_processor.hash_file_stream(fpath)
                            if live_hash.lower() == stored_hash.lower():
                                status = "🟢 VERIFIED (Unmodified)"
                                verified_count += 1
                            else:
                                status = "🔴 TAMPERED (Mismatch)"
                                tampered_count += 1
                        except Exception as e:
                            status = f"❌ READ ERROR ({e})"
                            live_hash = "ERROR"
                            tampered_count += 1

                    audit_results.append({
                        "Camera ID": s["camera_id"],
                        "Source Name": s["source_name"],
                        "Modality": s["modality"],
                        "File Name": fpath.name,
                        "Registered SHA-256": f"{stored_hash[:16]}...{stored_hash[-8:]}",
                        "Live Computed SHA-256": f"{live_hash[:16]}...{live_hash[-8:]}" if live_hash not in ("N/A", "ERROR") else live_hash,
                        "Integrity Status": status
                    })

                audit_summary = f"Total: {len(all_sources)} | Verified: {verified_count} | Tampered: {tampered_count} | Missing: {missing_count}"
                db_manager.log_investigator_action(
                    action="Cryptographic Integrity Audit",
                    entity_type="EvidenceSource",
                    entity_id="ALL",
                    details=audit_summary
                )

                st.session_state["last_integrity_audit"] = {
                    "results": audit_results,
                    "total": len(all_sources),
                    "verified": verified_count,
                    "tampered": tampered_count,
                    "missing": missing_count,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

    if "last_integrity_audit" in st.session_state:
        audit_data = st.session_state["last_integrity_audit"]
        st.caption(f"Last audit completed at: {audit_data['timestamp']}")

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total Files Audited", audit_data["total"])
        a2.metric("🟢 Verified Untampered", audit_data["verified"])
        a3.metric("🔴 Tampered / Mismatched", audit_data["tampered"])
        a4.metric("⚠️ Missing on Disk", audit_data["missing"])

        st.dataframe(pd.DataFrame(audit_data["results"]), use_container_width=True, hide_index=True)

        if audit_data["tampered"] == 0 and audit_data["missing"] == 0:
            st.success("✅ Chain of Custody Strictly Verified: 100% of evidence files match their registered cryptographic hashes.")
        else:
            st.error("⚠️ Evidence Chain of Custody Alert: Detected file discrepancy or missing data on disk.")


# =========================================================
# TAB 2: CCTV UPLOAD & CAMERA MANAGEMENT
# =========================================================

with tab_cctv_upload:
    st.header("📹 CCTV Video Upload & Camera Management")
    st.write(
        "Upload multiple CCTV recordings simultaneously. The system validates file containers, "
        "extracts resolution, duration, frame rate, computes SHA-256 hashes, and assigns unique Camera IDs."
    )

    # Multi-file uploader
    uploaded_cctv_files = st.file_uploader(
        "Select CCTV recordings (MP4, AVI, MOV, MKV)",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
        key="cctv_file_uploader_widget"
    )

    if uploaded_cctv_files:
        new_uploads = 0
        with st.spinner("Streaming and validating uploaded video files..."):
            for up_file in uploaded_cctv_files:
                save_info = media_processor.save_uploaded_stream(up_file, UPLOAD_DIR)
                vid_id = save_info["id"]

                # Check if already registered
                existing = db_manager.get_source_by_id(vid_id)
                if not existing:
                    meta = media_processor.get_video_metadata(save_info["path"])
                    if not meta["is_valid"]:
                        st.error(f"Error opening {up_file.name}: {meta['error']}")
                        continue

                    # Assign camera ID
                    cam_idx = len(st.session_state.video_records) + 1
                    cam_id = f"CAM-{cam_idx:02d}"

                    db_manager.save_source_record(
                        source_id=vid_id,
                        source_name=up_file.name,
                        file_path=save_info["path"],
                        modality="CCTV",
                        camera_id=cam_id,
                        file_size_bytes=save_info["size_bytes"],
                        sha256_hash=save_info["sha256"],
                        duration_seconds=meta["duration_seconds"],
                        fps=meta["fps"],
                        total_frames=meta["total_frames"],
                        width=meta["width"],
                        height=meta["height"]
                    )
                    db_manager.log_investigator_action(
                        action="Uploaded CCTV Video",
                        entity_type="CCTV",
                        entity_id=vid_id,
                        details=f"File: {up_file.name}, Camera: {cam_id}"
                    )
                    new_uploads += 1

        if new_uploads > 0:
            sync_session_from_database()
            st.success(f"Successfully processed and registered {new_uploads} new CCTV recording(s)!")
            st.rerun()

    # Display Registered Recordings
    st.divider()
    st.subheader("📂 Registered CCTV Recordings")

    cctv_sources = [s for s in db_manager.get_all_sources("CCTV")]

    if cctv_sources:
        st.write(f"**Total Cameras / Recordings Registered:** {len(cctv_sources)}")

        for src in cctv_sources:
            with st.container(border=True):
                c_thumb, c_meta, c_ctrl = st.columns([1.5, 3.5, 2])

                with c_thumb:
                    # Show thumbnail or video player
                    meta = media_processor.get_video_metadata(src["file_path"])
                    if meta.get("thumbnail_path") and Path(meta["thumbnail_path"]).is_file():
                        st.image(meta["thumbnail_path"], caption=f"Thumbnail ({src['camera_id']})", use_container_width=True)
                    elif Path(src["file_path"]).suffix.lower() == ".mp4":
                        st.video(src["file_path"])

                with c_meta:
                    st.markdown(f"#### 📹 {src['camera_id']} : {src['source_name']}")
                    st.write(f"• **Duration:** {src['duration_seconds']:.1f}s | **FPS:** {src['fps']} | **Frames:** {src['total_frames']}")
                    st.write(f"• **Resolution:** {src['width']}x{src['height']} | **File Size:** {src['file_size_bytes'] / (1024*1024):.2f} MB")
                    st.caption(f"SHA-256: `{src['sha256_hash']}`")

                with c_ctrl:
                    new_cam = st.text_input("Edit Camera ID", value=src["camera_id"], key=f"edit_cam_{src['source_id']}")
                    if new_cam != src["camera_id"] and st.button("Update Camera ID", key=f"btn_upd_cam_{src['source_id']}"):
                        db_manager.update_source_camera_id(src["source_id"], new_cam)
                        sync_session_from_database()
                        st.success(f"Updated Camera ID to {new_cam}")
                        st.rerun()

                    if st.button("🗑️ Remove Video", key=f"btn_del_vid_{src['source_id']}", type="secondary"):
                        db_manager.delete_source_record(src["source_id"])
                        sync_session_from_database()
                        st.warning(f"Removed {src['source_name']}")
                        st.rerun()
    else:
        st.info("No CCTV videos registered yet. Use the uploader above or load sample recordings from the sidebar.")


# =========================================================
# TAB 3: VIDEO OBJECT DETECTION & TRACKING (YOLOv8)
# =========================================================

with tab_detection:
    st.header("🤖 CCTV Object Detection & ByteTrack Tracking")
    st.write(
        "Runs YOLOv8 Nano on NVIDIA RTX 3050 (or CPU fallback). Tracks multi-object movements, "
        "assigns persistent tracking IDs, saves evidence keyframes, and generates an annotated review video."
    )

    cctv_sources = [s for s in db_manager.get_all_sources("CCTV")]

    if not cctv_sources:
        st.warning("Please upload at least one CCTV recording in the 'CCTV Upload' tab before running detection.")
    else:
        src_map = {f"{s['camera_id']} - {s['source_name']}": s for s in cctv_sources}

        col_opt1, col_opt2, col_opt3 = st.columns(3)

        with col_opt1:
            selected_video_keys = st.multiselect(
                "Select CCTV Recording(s) to Analyze",
                options=list(src_map.keys()),
                default=list(src_map.keys())[:1],
                key="select_vids_for_yolo"
            )

        with col_opt2:
            conf_thresh = st.slider(
                "Detection Confidence Threshold",
                min_value=0.10,
                max_value=0.90,
                value=0.35,
                step=0.05,
                key="yolo_conf_slider",
                help="Minimum confidence score for saving detected objects"
            )

        with col_opt3:
            frame_interval = st.slider(
                "Evidence Sampling Interval (seconds)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                key="yolo_interval_slider",
                help="ByteTrack tracks continuously; evidence snapshots are extracted at this interval."
            )

        target_classes_option = st.multiselect(
            "Filter Object Classes (Leave empty for All COCO Classes)",
            options=["person", "backpack", "handbag", "suitcase", "car", "motorcycle", "bicycle", "truck", "bus", "knife", "cell phone"],
            default=["person", "backpack", "handbag", "car", "motorcycle", "bicycle"],
            key="yolo_class_filter"
        )

        if st.button("🚀 Start YOLO Detection & Tracking", type="primary", key="btn_run_yolo"):
            if not selected_video_keys:
                st.warning("Please select at least one recording.")
            else:
                progress_bar = st.progress(0.0)
                status_txt = st.empty()

                total_new_detections = 0
                all_previews = []

                for v_key in selected_video_keys:
                    rec = src_map[v_key]
                    status_txt.write(f"Analyzing **{rec['camera_id']} ({rec['source_name']})** on {device_str}...")

                    def update_progress(pct, msg):
                        progress_bar.progress(pct)
                        status_txt.write(f"**{rec['camera_id']}:** {msg}")

                    detections, annotated_vid_path, previews = yolo_detector.run_cctv_detection_tracking(
                        video_path=rec["file_path"],
                        video_id=rec["source_id"],
                        source_name=rec["source_name"],
                        camera_id=rec["camera_id"],
                        conf_threshold=conf_thresh,
                        frame_interval_sec=frame_interval,
                        target_classes=target_classes_option if target_classes_option else None,
                        progress_callback=update_progress
                    )

                    db_manager.save_observations_batch(detections)
                    total_new_detections += len(detections)
                    all_previews.extend(previews)
                    st.session_state.active_annotated_video = annotated_vid_path

                sync_session_from_database()
                st.session_state.preview_images = all_previews

                status_txt.empty()
                progress_bar.progress(1.0)
                st.success(f"Analysis complete! {total_new_detections} object observations recorded across selected recordings.")

    # Detection Results & Annotated Video Player
    det_df = pd.DataFrame(st.session_state.detection_rows)
    if not det_df.empty:
        st.divider()
        st.subheader("🎬 Detection Results & Annotated Video Output")

        col_player, col_table = st.columns([1.5, 2.5])

        with col_player:
            st.markdown("##### 🎬 Annotated Video Playback")
            annotated_list = list(ANNOTATED_VIDEOS_DIR.glob("*_annotated.mp4"))
            if annotated_list:
                vid_choices = {v.name: v for v in sorted(annotated_list, key=os.path.getmtime, reverse=True)}
                default_idx = 0
                if st.session_state.active_annotated_video:
                    act_name = Path(st.session_state.active_annotated_video).name
                    if act_name in vid_choices:
                        default_idx = list(vid_choices.keys()).index(act_name)

                chosen_vid_name = st.selectbox(
                    "Select Recording to Watch",
                    options=list(vid_choices.keys()),
                    index=default_idx,
                    key="select_ann_video_to_play"
                )
                target_vid = vid_choices[chosen_vid_name]

                # Ensure web-browser H.264 compatibility
                web_vid_path = media_processor.convert_to_web_h264(str(target_vid))

                with open(web_vid_path, "rb") as f_vid:
                    vid_bytes = f_vid.read()

                st.video(vid_bytes, format="video/mp4")

                st.download_button(
                    "⬇️ Download Annotated Video (MP4)",
                    vid_bytes,
                    file_name=target_vid.name,
                    mime="video/mp4",
                    key=f"btn_dl_ann_{target_vid.stem}"
                )
            else:
                st.info("Run detection above to generate and preview annotated video.")

        with col_table:
            st.markdown("##### Detections Summary Table")
            display_cols = ["Observation ID", "Camera ID", "Timestamp", "Object", "Track ID", "Confidence", "Review Status"]
            valid_cols = [c for c in display_cols if c in det_df.columns]
            st.dataframe(det_df[valid_cols].head(50), use_container_width=True, hide_index=True)

            st.download_button(
                "⬇️ Download Detections CSV",
                det_df.to_csv(index=False).encode("utf-8"),
                file_name="cctv_detections.csv",
                mime="text/csv",
                key="btn_dl_det_csv"
            )

        # Evidence Snapshot Previews
        if st.session_state.preview_images:
            st.divider()
            st.subheader("🖼️ Sample Evidence Keyframes")
            preview_cols = st.columns(min(4, len(st.session_state.preview_images)))
            for idx, prev in enumerate(st.session_state.preview_images[:4]):
                with preview_cols[idx]:
                    st.image(prev["image"], caption=prev["caption"], use_container_width=True)


# =========================================================
# TAB 4: PHOTO EVIDENCE ANALYSIS
# =========================================================

with tab_photos:
    st.header("📷 Crime Scene Photograph Analysis")
    st.write("Upload still photographs from the scene. The YOLO model identifies relevant objects, weapons, vehicles, and evidence.")

    uploaded_photos = st.file_uploader(
        "Upload Scene Photographs (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        key="photo_uploader_widget"
    )

    if uploaded_photos:
        new_photos = 0
        for p_file in uploaded_photos:
            s_info = media_processor.save_uploaded_stream(p_file, UPLOAD_DIR)
            existing = db_manager.get_source_by_id(s_info["id"])
            if not existing:
                db_manager.save_source_record(
                    source_id=s_info["id"],
                    source_name=p_file.name,
                    file_path=s_info["path"],
                    modality="Photo",
                    camera_id="SCENE-PHOTO",
                    file_size_bytes=s_info["size_bytes"],
                    sha256_hash=s_info["sha256"]
                )
                new_photos += 1
        if new_photos > 0:
            sync_session_from_database()
            st.success(f"Registered {new_photos} scene photograph(s)!")
            st.rerun()

    photo_sources = [s for s in db_manager.get_all_sources("Photo")]

    if photo_sources:
        st.divider()
        st.subheader("Analyze Uploaded Scene Photographs")

        p_options = {f"{s['source_name']} ({s['source_id']})": s for s in photo_sources}
        selected_photo_keys = st.multiselect("Select Photographs", options=list(p_options.keys()), key="select_photos_for_analysis")
        photo_conf = st.slider("Photo Detection Confidence", 0.10, 0.90, 0.35, 0.05, key="slider_photo_conf")

        if st.button("🔎 Analyze Selected Photographs", type="primary", key="btn_run_photo_analysis"):
            if not selected_photo_keys:
                st.warning("Please select at least one photograph.")
            else:
                total_p_obs = 0
                for p_key in selected_photo_keys:
                    p_rec = p_options[p_key]
                    detections, ann_path, rgb_img = yolo_detector.run_photo_detection(
                        image_path=p_rec["file_path"],
                        photo_id=p_rec["source_id"],
                        source_name=p_rec["source_name"],
                        conf_threshold=photo_conf
                    )
                    db_manager.save_observations_batch(detections)
                    total_p_obs += len(detections)
                    st.image(rgb_img, caption=f"Analyzed: {p_rec['source_name']} ({len(detections)} objects found)", use_container_width=True)

                sync_session_from_database()
                st.success(f"Photo analysis complete: {total_p_obs} object observation(s) recorded.")
    else:
        st.info("No photographs uploaded yet.")


# =========================================================
# TAB 5: AUDIO INTELLIGENCE & TRANSCRIPTION
# =========================================================

with tab_audio:
    st.header("🎙️ Audio Intelligence & Multilingual Speech Analysis")
    st.write(
        "Extract audio from CCTV recordings or upload standalone dispatch / 911 audio recordings. "
        "Faster-Whisper transcribes speech in its original language, translates into English, and indexes timestamps."
    )

    audio_mode = st.radio(
        "Audio Input Method",
        ["📹 Extract Audio from Uploaded CCTV Video", "📁 Upload Standalone Audio File"],
        horizontal=True,
        key="radio_audio_mode"
    )

    if audio_mode == "📹 Extract Audio from Uploaded CCTV Video":
        cctv_sources = [s for s in db_manager.get_all_sources("CCTV")]
        if not cctv_sources:
            st.info("No CCTV videos registered. Please upload a CCTV recording first.")
        else:
            cctv_select_map = {f"{s['camera_id']} - {s['source_name']}": s for s in cctv_sources}
            chosen_cctv_label = st.selectbox("Select CCTV Recording", options=list(cctv_select_map.keys()), key="select_cctv_for_audio")
            chosen_cctv = cctv_select_map[chosen_cctv_label]

            if st.button("🎵 Extract & Transcribe CCTV Audio", type="primary", key="btn_extract_cctv_audio"):
                with st.spinner("Extracting audio stream using PyAV..."):
                    wav_path = media_processor.extract_audio_from_cctv(chosen_cctv["file_path"])

                if not wav_path:
                    st.warning(f"No audio stream found in {chosen_cctv['source_name']}. (This CCTV recording is video-only).")
                else:
                    st.success("Audio stream extracted successfully! Transcribing with Faster-Whisper...")
                    with st.spinner("Transcribing speech..."):
                        audio_obs = whisper_transcriber.transcribe_and_translate_audio(
                            audio_file_path=wav_path,
                            source_id=chosen_cctv["source_id"],
                            source_name=f"Audio_{chosen_cctv['source_name']}",
                            camera_id=chosen_cctv["camera_id"],
                            model_size="base",
                            device="cpu"
                        )
                        db_manager.save_observations_batch(audio_obs)
                        sync_session_from_database()
                        st.success(f"Transcribed {len(audio_obs)} speech segments from CCTV recording!")

    else:
        uploaded_audio_files = st.file_uploader(
            "Upload Audio Files (WAV, MP3, M4A, FLAC, OGG)",
            type=["wav", "mp3", "m4a", "flac", "ogg", "aac"],
            accept_multiple_files=True,
            key="audio_file_uploader_widget"
        )
        if uploaded_audio_files:
            new_aud = 0
            for a_file in uploaded_audio_files:
                s_info = media_processor.save_uploaded_stream(a_file, UPLOAD_DIR)
                existing = db_manager.get_source_by_id(s_info["id"])
                if not existing:
                    db_manager.save_source_record(
                        source_id=s_info["id"],
                        source_name=a_file.name,
                        file_path=s_info["path"],
                        modality="Audio",
                        camera_id="AUDIO-01",
                        file_size_bytes=s_info["size_bytes"],
                        sha256_hash=s_info["sha256"]
                    )
                    new_aud += 1
            if new_aud > 0:
                sync_session_from_database()
                st.success(f"Registered {new_aud} audio recording(s)!")
                st.rerun()

        audio_sources = [s for s in db_manager.get_all_sources("Audio")]
        if audio_sources:
            aud_map = {f"{s['source_name']} ({s['source_id']})": s for s in audio_sources}
            sel_aud_key = st.selectbox("Select Audio File to Transcribe", options=list(aud_map.keys()), key="select_standalone_audio")
            whisper_size = st.selectbox("Whisper Model Size", ["tiny", "base", "small"], index=1, key="select_whisper_size")

            if st.button("🎧 Transcribe & Translate", type="primary", key="btn_run_whisper"):
                chosen_aud = aud_map[sel_aud_key]
                with st.spinner("Processing speech transcription and translation..."):
                    audio_obs = whisper_transcriber.transcribe_and_translate_audio(
                        audio_file_path=chosen_aud["file_path"],
                        source_id=chosen_aud["source_id"],
                        source_name=chosen_aud["source_name"],
                        camera_id="AUDIO-01",
                        model_size=whisper_size,
                        device="cpu"
                    )
                    db_manager.save_observations_batch(audio_obs)
                    sync_session_from_database()
                    st.success(f"Completed! {len(audio_obs)} speech segments generated.")

    # Audio Transcripts Display
    audio_df = pd.DataFrame(st.session_state.audio_rows)
    if not audio_df.empty:
        st.divider()
        st.subheader("📜 Transcribed Speech Records")
        for _, a_row in audio_df.iterrows():
            with st.container(border=True):
                st.write(f"🎙️ **{a_row['Source']}** | **Camera/Channel:** {a_row.get('Camera ID', 'AUDIO')} | **Timestamp:** {a_row['Timestamp']}")
                st.write(f"• **Original ({a_row.get('Detected Language', 'Unknown')}):** {a_row['Original Transcript']}")
                st.write(f"• **English Translation:** {a_row['English Translation']}")
                st.caption(f"Observation ID: `{a_row['Observation ID']}` | Review Status: {a_row.get('Review Status', 'Pending')}")


# =========================================================
# TAB 6: UNIFIED EVIDENCE REPOSITORY
# =========================================================

with tab_repository:
    st.header("🗄️ Consolidated Evidence Repository")
    st.write("Master repository of all recorded observations across CCTV video, photographic evidence, and audio transcripts.")

    all_obs_df = db_manager.get_all_observations_df()

    if all_obs_df.empty:
        st.info("No evidence records available. Run CCTV Detection, Photo Analysis, or Audio Transcription.")
    else:
        # Filters
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            mod_filter = st.selectbox("Filter Modality", ["All"] + sorted(all_obs_df["Modality"].unique().tolist()), key="rep_filter_mod")
        with f2:
            cam_filter = st.selectbox("Filter Camera", ["All"] + sorted(all_obs_df["Camera ID"].unique().tolist()), key="rep_filter_cam")
        with f3:
            obj_filter = st.selectbox("Filter Object/Type", ["All"] + sorted(all_obs_df["Object"].unique().tolist()), key="rep_filter_obj")
        with f4:
            stat_filter = st.selectbox("Filter Status", ["All", "Pending", "Approved", "Rejected", "Flagged"], key="rep_filter_stat")

        filtered_df = all_obs_df.copy()
        if mod_filter != "All":
            filtered_df = filtered_df[filtered_df["Modality"] == mod_filter]
        if cam_filter != "All":
            filtered_df = filtered_df[filtered_df["Camera ID"] == cam_filter]
        if obj_filter != "All":
            filtered_df = filtered_df[filtered_df["Object"] == obj_filter]
        if stat_filter != "All":
            filtered_df = filtered_df[filtered_df["Review Status"] == stat_filter]

        st.metric("Matching Observations", len(filtered_df))

        display_cols = [
            "Observation ID", "Modality", "Camera ID", "Source", "Timestamp",
            "Object", "Track ID", "Confidence", "Observation", "Review Status"
        ]
        valid_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[valid_cols], use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Export Evidence Repository (CSV)",
            filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="evidence_repository_export.csv",
            mime="text/csv",
            key="btn_dl_repo_csv"
        )


# =========================================================
# TAB 7: MULTI-CAMERA EVENT RECONSTRUCTION & TIMELINE
# =========================================================

with tab_events:
    subtab_reconstruct, subtab_timeline = st.tabs(["⚡ Candidate Event Reconstruction", "📅 Unified Chronological Evidence Timeline"])

    with subtab_reconstruct:
        st.header("⏱️ Multi-Camera Event Reconstruction")
        st.write(
            "Automatically group continuous observations into candidate events based on time proximity, "
            "ByteTrack IDs, and camera provenance."
        )

        all_obs_df = db_manager.get_all_observations_df()
        if all_obs_df.empty:
            st.info("Please run CCTV object detection to generate observations first.")
        else:
            norm_obs = normalise_observations(all_obs_df)

            col_ev1, col_ev2 = st.columns(2)
            with col_ev1:
                max_gap = st.slider("Maximum Gap Between Observations (seconds)", 5, 120, 30, 5, key="slider_ev_max_gap")
            with col_ev2:
                min_obs = st.slider("Minimum Observations per Event", 2, 10, 2, 1, key="slider_ev_min_obs")

            if st.button("⚡ Reconstruct Candidate Events", type="primary", key="btn_reconstruct_events"):
                events_df = reconstruct_events(
                    observations=norm_obs,
                    max_gap_seconds=max_gap,
                    min_observations=min_obs
                )
                st.success(f"Generated {len(events_df)} candidate reconstructed event(s).")

            events = get_reconstructed_events(active_only=True)
            if not events.empty:
                st.divider()
                st.subheader(f"Identified Candidate Events ({len(events)})")
                for _, ev in events.iterrows():
                    with st.container(border=True):
                        e_col1, e_col2, e_col3 = st.columns([3, 2, 1])
                        with e_col1:
                            st.markdown(f"#### ⚡ {ev['event_title']}")
                            st.write(f"• **Source / Camera:** {ev['source']} | **Category:** {ev['object_category']}")
                            st.caption(f"Explanation: {ev['event_explanation']}")
                        with e_col2:
                            s_time = ev['start_seconds']
                            e_time = ev['end_seconds']
                            st.write(f"• **Time:** {int(s_time//60):02d}:{s_time%60:04.1f} - {int(e_time//60):02d}:{e_time%60:04.1f}")
                            st.write(f"• **Observations:** {ev['observation_count']} | **Avg Confidence:** {ev['average_confidence']:.1%}")
                        with e_col3:
                            st.write(f"Status: **{ev['review_status']}**")
            else:
                st.caption("No reconstructed events generated yet. Click 'Reconstruct Candidate Events' above.")

    with subtab_timeline:
        all_obs_df = db_manager.get_all_observations_df()
        unified_evidence_timeline.render_unified_evidence_timeline(all_obs_df)


# =========================================================
# TAB 8: CROSS-CAMERA EVIDENCE LINKING & CORRELATION
# =========================================================

with tab_cross_camera:
    subtab_cross_cam, subtab_multimodal, subtab_side_by_side = st.tabs([
        "🌐 Cross-Camera Sighting Correlation",
        "🔗 Cross-Modal Correlation (Audio-Visual)",
        "👥 Side-by-Side Sighting Comparison"
    ])

    with subtab_cross_cam:
        st.header("🌐 Cross-Camera Evidence Linking & Movement Tracking")
        st.write(
            "Correlate candidate object sightings across multiple CCTV cameras. "
            "Important: Cross-camera associations represent algorithmic candidate links and do NOT prove physical object identity."
        )
        cross_camera_linking.render_cross_camera_linking()

    with subtab_multimodal:
        all_obs_df = db_manager.get_all_observations_df()
        cross_modal_correlation.render_cross_modal_correlation(all_obs_df)

    with subtab_side_by_side:
        st.header("👥 Side-by-Side CCTV Sighting Comparison & Forensic Linker")
        st.write(
            "Compare two candidate CCTV sightings side-by-side to inspect visual characteristics, "
            "spatio-temporal transit time (Δt), and establish or reject forensic relationships in the master knowledge graph."
        )

        cctv_obs = db_manager.get_all_observations_df(modality="CCTV")
        if cctv_obs.empty or len(cctv_obs) < 2:
            st.info("At least two CCTV object sightings are required to perform a side-by-side forensic comparison.")
        else:
            avail_classes = ["All Classes"] + sorted(list(cctv_obs["Object"].dropna().unique()))
            col_filter, _ = st.columns([2, 3])
            with col_filter:
                selected_class = st.selectbox(
                    "Filter candidate sightings by Object Class:",
                    avail_classes,
                    key="sb_sighting_class_filter"
                )

            filtered_cctv = cctv_obs if selected_class == "All Classes" else cctv_obs[cctv_obs["Object"] == selected_class]

            if filtered_cctv.empty or len(filtered_cctv) < 2:
                st.warning(f"Not enough sightings found for class '{selected_class}'. Try 'All Classes'.")
            else:
                def format_sighting_label(r):
                    oid = str(r["Observation ID"])[:8]
                    cam = r.get("Camera ID", "CAM-01")
                    ts = r.get("Timestamp", "00:00.00")
                    obj = r.get("Object", "Target")
                    conf = r.get("Confidence")
                    conf_str = f"{conf:.0%}" if pd.notna(conf) else "N/A"
                    track = r.get("Track ID", "Untracked")
                    return f"[{oid}] {cam} @ {ts} | {obj} ({conf_str}, {track})"

                options_records = filtered_cctv.to_dict("records")
                labels = [format_sighting_label(r) for r in options_records]
                record_lookup = {labels[i]: options_records[i] for i in range(len(options_records))}

                col_pick_a, col_pick_b = st.columns(2)
                with col_pick_a:
                    choice_a = st.selectbox(
                        "🔍 Select Sighting A (Origin Observation):",
                        labels,
                        index=0,
                        key="sb_choice_sighting_a"
                    )
                with col_pick_b:
                    choice_b = st.selectbox(
                        "🎯 Select Sighting B (Candidate Transit / Target Observation):",
                        labels,
                        index=1 if len(labels) > 1 else 0,
                        key="sb_choice_sighting_b"
                    )

                rec_a = record_lookup[choice_a]
                rec_b = record_lookup[choice_b]

                # Helper to render bounding box and crop
                import ast

                def render_sighting_crop(evidence_path, bbox_str, label_text):
                    if not evidence_path or not os.path.exists(evidence_path):
                        return None, None
                    bgr = cv2.imread(evidence_path)
                    if bgr is None:
                        return None, None
                    crop_rgb = None
                    if bbox_str and bbox_str.strip():
                        try:
                            coords = ast.literal_eval(bbox_str)
                            if isinstance(coords, (list, tuple)) and len(coords) == 4:
                                h, w = bgr.shape[:2]
                                x1 = max(0, int(round(float(coords[0]))))
                                y1 = max(0, int(round(float(coords[1]))))
                                x2 = min(w, int(round(float(coords[2]))))
                                y2 = min(h, int(round(float(coords[3]))))
                                if x2 > x1 and y2 > y1:
                                    crop_rgb = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)
                                if label_text:
                                    cv2.putText(bgr, label_text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        except Exception:
                            pass
                    full_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    return full_rgb, crop_rgb

                st.divider()

                # Visual Comparison Columns
                col_view_a, col_view_b = st.columns(2)

                with col_view_a:
                    st.markdown(f"#### 📹 Sighting A: {rec_a['Camera ID']} ({rec_a['Timestamp']})")
                    full_a, crop_a = render_sighting_crop(rec_a.get("Evidence Path", ""), rec_a.get("Bounding Box", ""), f"{rec_a['Object']} ({rec_a.get('Track ID', '')})")

                    sub_col_crop, sub_col_full = st.columns([1, 2])
                    with sub_col_crop:
                        if crop_a is not None:
                            st.caption("Zoomed Subject Crop")
                            st.image(crop_a, use_container_width=True)
                        else:
                            st.caption("Crop not available")
                    with sub_col_full:
                        if full_a is not None:
                            st.caption("Full Frame Context")
                            st.image(full_a, use_container_width=True)
                        else:
                            st.info("Full frame not found on disk.")

                    st.markdown(f"""
                    - **Observation ID**: `{rec_a['Observation ID']}`
                    - **Camera**: `{rec_a['Camera ID']}` | **Track**: `{rec_a.get('Track ID', 'N/A')}`
                    - **Detected Object**: `{rec_a['Object']}` (Confidence: `{rec_a.get('Confidence', 0):.0%}`)
                    - **Timestamp**: `{rec_a['Timestamp']}` (`{rec_a.get('Timestamp (seconds)', 0):.2f}s`)
                    - **Review Status**: `{rec_a.get('Review Status', 'Pending')}`
                    """)

                with col_view_b:
                    st.markdown(f"#### 📹 Sighting B: {rec_b['Camera ID']} ({rec_b['Timestamp']})")
                    full_b, crop_b = render_sighting_crop(rec_b.get("Evidence Path", ""), rec_b.get("Bounding Box", ""), f"{rec_b['Object']} ({rec_b.get('Track ID', '')})")

                    sub_col_crop_b, sub_col_full_b = st.columns([1, 2])
                    with sub_col_crop_b:
                        if crop_b is not None:
                            st.caption("Zoomed Subject Crop")
                            st.image(crop_b, use_container_width=True)
                        else:
                            st.caption("Crop not available")
                    with sub_col_full_b:
                        if full_b is not None:
                            st.caption("Full Frame Context")
                            st.image(full_b, use_container_width=True)
                        else:
                            st.info("Full frame not found on disk.")

                    st.markdown(f"""
                    - **Observation ID**: `{rec_b['Observation ID']}`
                    - **Camera**: `{rec_b['Camera ID']}` | **Track**: `{rec_b.get('Track ID', 'N/A')}`
                    - **Detected Object**: `{rec_b['Object']}` (Confidence: `{rec_b.get('Confidence', 0):.0%}`)
                    - **Timestamp**: `{rec_b['Timestamp']}` (`{rec_b.get('Timestamp (seconds)', 0):.2f}s`)
                    - **Review Status**: `{rec_b.get('Review Status', 'Pending')}`
                    """)

                # Spatio-Temporal Transit Analysis Card
                st.divider()
                st.subheader("⏱️ Spatio-Temporal Transit & Correlation Analysis")

                sec_a = float(rec_a.get("Timestamp (seconds)", 0) or 0)
                sec_b = float(rec_b.get("Timestamp (seconds)", 0) or 0)
                delta_t = abs(sec_b - sec_a)

                if sec_a < sec_b:
                    seq_text = f"Sighting A ({rec_a['Camera ID']} @ {rec_a['Timestamp']}) preceded Sighting B ({rec_b['Camera ID']} @ {rec_b['Timestamp']}) by **+{delta_t:.2f} seconds**."
                elif sec_b < sec_a:
                    seq_text = f"Sighting B ({rec_b['Camera ID']} @ {rec_b['Timestamp']}) preceded Sighting A ({rec_a['Camera ID']} @ {rec_a['Timestamp']}) by **+{delta_t:.2f} seconds**."
                else:
                    seq_text = f"Both sightings occurred simultaneously at **{rec_a['Timestamp']}** (Δt = 0.00s)."

                # Transit plausibility assessment
                if delta_t < 4.0:
                    transit_eval = "🟢 **Simultaneous / Direct Overlap**: Sightings are within 4 seconds. Highly consistent with overlapping camera fields of view or adjacent angles."
                elif delta_t <= 90.0:
                    transit_eval = f"🟢 **Plausible Pedestrian Transit Window**: Transit time of {delta_t:.1f}s is anatomically and geographically consistent with walking or running between adjacent camera zones."
                elif delta_t <= 300.0:
                    transit_eval = f"🟡 **Extended Transit Gap**: Transit time of {delta_t:.1f}s indicates possible dwell time, stops, or indirect path between camera zones."
                else:
                    transit_eval = f"⚪ **Long Temporal Separation**: Δt = {delta_t:.1f}s (> 5 minutes). High probability of independent incidents unless stationary dwell is verified."

                match_eval = "✅ **Semantic Match**: Both detections share identical object classification." if rec_a["Object"].lower() == rec_b["Object"].lower() else f"⚠️ **Class Divergence**: Sightings classified as '{rec_a['Object']}' vs '{rec_b['Object']}'."

                card_col1, card_col2 = st.columns([3, 2])
                with card_col1:
                    st.info(f"**Chronological Sequence**: {seq_text}\n\n**Transit Plausibility**: {transit_eval}\n\n**Classification**: {match_eval}")
                with card_col2:
                    st.warning(
                        "⚠️ **Forensic Notice (Investigator Responsibility)**:\n\n"
                        "Computer vision similarity and transit proximity do **NOT** constitute definitive biometric proof of identity. "
                        "Forensic investigators must verify facial features, unique attire, luggage markings, and continuous route custody."
                    )

                # Action Bar to Record Link into Knowledge Graph
                st.subheader("🔗 Record Forensic Relationship in Knowledge Graph")
                col_rel_type, col_rel_stat = st.columns([2, 1.5])

                with col_rel_type:
                    rel_type = st.selectbox(
                        "Relationship Type:",
                        [
                            "TRANSIT_PATH (+Δt)",
                            "SAME_ENTITY_CANDIDATE",
                            "CO_OCCURRENCE",
                            "SUSPICIOUS_PROXIMITY",
                            "DISTINCT_ENTITIES (Rule Out)"
                        ],
                        key="sb_link_rel_type"
                    )

                with col_rel_stat:
                    rel_status = st.selectbox(
                        "Decision Status:",
                        ["Approved", "Flagged", "Pending", "Rejected"],
                        key="sb_link_rel_status"
                    )

                link_notes = st.text_input(
                    "Investigator Notes / Rationale:",
                    value=f"Correlated {rec_a['Camera ID']} and {rec_b['Camera ID']} sightings with Δt = {delta_t:.1f}s",
                    key="txt_link_notes"
                )

                if st.button("💾 Record Forensic Link in Knowledge Graph", type="primary", key="btn_save_sighting_link"):
                    rel_id = db_manager.record_graph_relationship(
                        source_observation=str(rec_a["Observation ID"]),
                        target_observation=str(rec_b["Observation ID"]),
                        relationship_type=rel_type,
                        explanation=f"{seq_text}. {link_notes}",
                        confidence=0.85 if rel_status == "Approved" else 0.5,
                        status=rel_status,
                        origin="Investigator Sighting Comparison",
                        reviewer_notes=link_notes
                    )
                    st.success(f"✅ Forensic Link `{rel_id}` successfully recorded! It is now actively connected in Tab 9 (Knowledge Graph).")
                    st.rerun()


# =========================================================
# TAB 9: INTERACTIVE EVIDENCE GRAPH (NetworkX + PyVis)
# =========================================================

with tab_graph:
    st.header("🕸️ Advanced Multimodal Evidence Knowledge Graph")
    st.caption(
        "Forensic AI graph linking CCTV Cameras, Sensor Provenance, Detected Objects, "
        "ByteTrack Continuity, Spatio-Temporal Trajectories, and Audio-Visual Corroborations."
    )

    all_obs_df = db_manager.get_all_observations_df()

    if all_obs_df.empty:
        st.info("No observations recorded yet. Upload and analyze evidence in previous tabs to generate the graph.")
    else:
        # Layout & Advanced AI Innovation Controls
        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1.5, 1.5, 2])
        with ctrl_c1:
            graph_layout = st.selectbox(
                "📐 Graph Forensic Perspective",
                [
                    "Force-Directed Physics (Cluster)",
                    "Timeline (Left-to-Right)",
                    "Forensic Hierarchy Tree"
                ],
                key="select_graph_layout",
                help="Switch between cluster force-directed physics, chronological left-to-right timeline, or top-down chain of custody tree."
            )
        with ctrl_c2:
            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
            traj_toggle = st.checkbox(
                "⚡ Spatio-Temporal Trajectory",
                value=True,
                key="chk_traj_inference",
                help="Infers cross-camera transit paths with elapsed transit seconds (delta-t)."
            )
        with ctrl_c3:
            st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
            av_toggle = st.checkbox(
                "🎙️ Audio-Visual Cross-Corroboration",
                value=True,
                key="chk_av_fusion",
                help="Semantically links spoken keywords from audio transcripts to visual object detections in the same time frame."
            )

        # Forensic Filter Bar
        g_c1, g_c2, g_c3, g_c4, g_c5 = st.columns(5)
        with g_c1:
            g_cam = st.selectbox("Filter Camera", ["All Cameras"] + sorted(all_obs_df["Camera ID"].unique().tolist()), key="g_filter_cam")
        with g_c2:
            g_mod = st.selectbox("Filter Modality", ["All Modalities"] + sorted(all_obs_df["Modality"].unique().tolist()), key="g_filter_mod")
        with g_c3:
            g_obj = st.selectbox("Filter Object", ["All Objects"] + sorted(all_obs_df["Object"].unique().tolist()), key="g_filter_obj")
        with g_c4:
            available_tracks = [t for t in all_obs_df["Track ID"].unique().tolist() if t not in ("Untracked", "", "nan", "None", "Not applicable")]
            g_track = st.selectbox("Focus Trajectory / Track", ["All Tracks / Objects"] + sorted(available_tracks), key="g_filter_track")
        with g_c5:
            g_search = st.text_input("Search Graph Nodes", placeholder="e.g. backpack, person, car", key="g_search_kw")

        events_df = get_reconstructed_events(active_only=True)
        cross_links_df = cross_camera_linking.get_links()

        with st.spinner("Synthesizing Multimodal Evidence Knowledge Graph..."):
            nx_graph, graph_html, graph_intel = evidence_graph_builder.build_advanced_evidence_graph(
                observations_df=all_obs_df,
                events_df=events_df,
                cross_links_df=cross_links_df,
                layout_mode=graph_layout,
                filter_camera=g_cam,
                filter_modality=g_mod,
                filter_object=g_obj,
                focus_track_id=g_track,
                search_keyword=g_search,
                enable_trajectory_inference=traj_toggle,
                enable_audio_visual_fusion=av_toggle,
                max_nodes=150
            )

        # Graph Centrality & Forensic AI Analytics Panel
        st.divider()
        st.markdown("##### 🧠 Forensic AI Graph Intelligence & Centrality Analytics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Graph Scale", f"{graph_intel['total_nodes']} Nodes / {graph_intel['total_edges']} Edges")
        m_col2.metric("🏆 Primary Evidence Hub", f"{graph_intel['key_hub'][:22]}", help="Node with highest Degree Centrality in the graph")
        m_col3.metric("🌉 Critical Forensic Bridge", f"{graph_intel['critical_bridge'][:22]}", help="Node with highest Betweenness Centrality connecting disparate evidence clusters")
        m_col4.metric("Network Density", f"{graph_intel['density']:.3f}", help="Completeness of evidence connectivity")

        # Interactive PyVis Canvas
        components.html(graph_html, height=720, scrolling=True)

        # Node Dossier Inspector & Graph Export
        col_inspect, col_export = st.columns([3, 1.2])

        with col_inspect:
            with st.expander("🔍 Interactive Evidence Node Dossier & Inspector", expanded=False):
                obs_choices = {
                    f"{r['Observation ID']} - {r['Object']} [{r['Camera ID']} @ {r['Timestamp']}]": r
                    for _, r in all_obs_df.iterrows()
                }
                selected_obs_key = st.selectbox(
                    "Select Evidence Node to Inspect",
                    options=list(obs_choices.keys()),
                    key="select_node_inspector"
                )
                if selected_obs_key:
                    ins_rec = obs_choices[selected_obs_key]
                    i_c1, i_c2 = st.columns([1.5, 2.5])
                    with i_c1:
                        if ins_rec.get("Evidence Path") and Path(ins_rec["Evidence Path"]).is_file():
                            st.image(ins_rec["Evidence Path"], caption=f"{ins_rec['Object']} ({ins_rec['Camera ID']})", use_container_width=True)
                        else:
                            st.info("No snapshot image attached.")
                    with i_c2:
                        st.markdown(f"#### {ins_rec['Object']} (`{ins_rec['Observation ID']}`)")
                        st.write(f"• **Camera:** {ins_rec['Camera ID']} | **Source:** {ins_rec['Source']}")
                        st.write(f"• **Timestamp:** {ins_rec['Timestamp']} | **Track:** {ins_rec.get('Track ID', 'Untracked')}")
                        st.write(f"• **Confidence:** {ins_rec.get('Confidence', 'N/A')}")
                        st.write(f"• **Observation:** {ins_rec['Observation']}")
                        if ins_rec.get("Modality") == "Audio":
                            st.write(f"• **Original Speech:** {ins_rec.get('Original Transcript', '')}")
                            st.write(f"• **Translation:** {ins_rec.get('English Translation', '')}")

                        # Quick verification
                        q_stat = st.selectbox(
                            "Update Review Decision",
                            ["Pending", "Approved", "Rejected", "Flagged"],
                            index=["Pending", "Approved", "Rejected", "Flagged"].index(ins_rec.get("Review Status", "Pending")),
                            key=f"q_rev_{ins_rec['Observation ID']}"
                        )
                        q_note = st.text_input("Investigator Note", value=ins_rec.get("Review Notes", ""), key=f"q_note_{ins_rec['Observation ID']}")
                        if st.button("Apply Review", key=f"btn_apply_q_rev_{ins_rec['Observation ID']}"):
                            db_manager.update_observation_review(ins_rec["Observation ID"], q_stat, q_note)
                            sync_session_from_database()
                            st.success(f"Updated status to {q_stat}")
                            st.rerun()

        with col_export:
            st.markdown("##### 💾 Graph Dossier Export")
            st.download_button(
                "⬇️ Download Interactive HTML Graph",
                graph_html.encode("utf-8"),
                file_name="Forensic_Evidence_Graph_Interactive.html",
                mime="text/html",
                key="btn_dl_graph_html",
                help="Export standalone interactive graph view for court or examiner presentation."
            )
            # GraphML Export for Gephi / Cytoscape
            try:
                # Prepare G without complex objects for GraphML
                G_export = nx.DiGraph()
                for n, data in nx_graph.nodes(data=True):
                    G_export.add_node(str(n), label=str(data.get("label", n)), type=str(data.get("type", "Node")))
                for u, v, data in nx_graph.edges(data=True):
                    G_export.add_edge(str(u), str(v), relation=str(data.get("relation", "LINK")))
                import io
                gml_buffer = io.BytesIO()
                nx.write_graphml(G_export, gml_buffer)
                st.download_button(
                    "⬇️ Export GraphML (Gephi / Analyst)",
                    gml_buffer.getvalue(),
                    file_name="evidence_graph.graphml",
                    mime="application/xml",
                    key="btn_dl_graphml"
                )
            except Exception:
                pass


# =========================================================
# TAB 10: NATURAL LANGUAGE INVESTIGATION SEARCH
# =========================================================

with tab_search:
    st.header("🔍 Natural Language & Parametric Evidence Search")
    st.write(
        "Query evidence across cameras, timestamps, detected objects, and audio transcripts using natural language or keywords."
    )

    all_obs_df = db_manager.get_all_observations_df()
    evidence_search_engine.render_evidence_search(all_obs_df)


# =========================================================
# TAB 11: HUMAN-IN-THE-LOOP VERIFICATION CENTRE
# =========================================================

with tab_verification:
    st.header("🧑‍⚖️ Investigator Review & Verification Centre")
    st.write(
        "Human-in-the-loop audit workstation. Every AI-generated detection, transcript, and event starts as 'Pending'. "
        "Approve, reject, or flag observations and attach formal case notes. Original AI findings are never overwritten."
    )

    all_obs_df = db_manager.get_all_observations_df()

    if all_obs_df.empty:
        st.info("No observations available for review.")
    else:
        # Status counts
        stat_counts = all_obs_df["Review Status"].value_counts()
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("Pending Review", stat_counts.get("Pending", 0))
        v2.metric("Approved Evidence", stat_counts.get("Approved", 0))
        v3.metric("Rejected Evidence", stat_counts.get("Rejected", 0))
        v4.metric("Flagged Evidence", stat_counts.get("Flagged", 0))

        st.divider()

        # Review single item or table editor
        st.subheader("Interactive Observation Reviewer")

        review_cols = [
            "Observation ID", "Modality", "Camera ID", "Source", "Timestamp",
            "Object", "Confidence", "Observation", "Review Status", "Review Notes"
        ]
        valid_rev_cols = [c for c in review_cols if c in all_obs_df.columns]

        edited_df = st.data_editor(
            all_obs_df[valid_rev_cols],
            use_container_width=True,
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

        if st.button("💾 Save Review Decisions to Database", type="primary", key="btn_save_reviews"):
            updated_count = 0
            for _, row in edited_df.iterrows():
                obs_id = row["Observation ID"]
                curr_status = row["Review Status"]
                curr_notes = row["Review Notes"] if pd.notna(row["Review Notes"]) else ""

                # Check if changed
                orig_match = all_obs_df[all_obs_df["Observation ID"] == obs_id]
                if not orig_match.empty:
                    orig_stat = orig_match.iloc[0]["Review Status"]
                    orig_note = orig_match.iloc[0]["Review Notes"]
                    if curr_status != orig_stat or curr_notes != orig_note:
                        db_manager.update_observation_review(obs_id, curr_status, curr_notes)
                        updated_count += 1

            sync_session_from_database()
            st.success(f"Saved {updated_count} review decision(s) into database!")
            st.rerun()

        st.divider()
        st.subheader("📜 Complete Forensic Audit Trail & Chain of Custody")
        st.write("Immutable log of all investigator decisions, status updates, and cryptographic integrity checks.")
        full_audit_df = db_manager.get_audit_logs()
        if not full_audit_df.empty:
            st.dataframe(
                full_audit_df[["log_id", "timestamp", "action", "entity_type", "entity_id", "details"]],
                use_container_width=True,
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


# =========================================================
# TAB 12: FORENSIC INVESTIGATION REPORTS & EXPORT
# =========================================================

with tab_reports:
    st.header("📄 Investigation Reports & Evidence Export")
    st.write("Generate formal, publication-quality Forensic Investigation PDF reports and export case datasets.")

    col_rep_meta, col_rep_actions = st.columns([2, 1.5])

    with col_rep_meta:
        rep_title = st.text_input("Case Title", value="Armed Incident & CCTV Tracking Investigation", key="txt_rep_title")
        rep_id = st.text_input("Case Reference Number", value="CRIME-2026-X89", key="txt_rep_id")
        rep_officer = st.text_input("Lead Investigating Officer", value="Inspector R. Sharma (Forensic AI Unit)", key="txt_rep_officer")
        rep_notes = st.text_area("Case Summary & Additional Remarks", placeholder="Enter investigator summary...", key="txt_rep_notes")

    with col_rep_actions:
        st.subheader("Generate PDF Report")
        if st.button("📑 Generate Forensic PDF Report", type="primary", key="btn_gen_pdf_report"):
            with st.spinner("Compiling dossier and generating PDF with ReportLab..."):
                sources = db_manager.get_all_sources()
                all_obs = db_manager.get_all_observations_df()
                events = get_reconstructed_events(active_only=True)
                logs = db_manager.get_audit_logs()

                pdf_path = report_generator.generate_pdf_report(
                    case_title=rep_title,
                    case_id=rep_id,
                    investigator_name=rep_officer,
                    sources=sources,
                    observations_df=all_obs,
                    events_df=events,
                    audit_logs_df=logs,
                    notes=rep_notes
                )
                st.session_state.last_report_path = pdf_path
                st.success("PDF Report generated successfully!")

        if st.session_state.last_report_path and Path(st.session_state.last_report_path).is_file():
            with open(st.session_state.last_report_path, "rb") as f_pdf:
                st.download_button(
                    "⬇️ Download Investigation PDF Report",
                    f_pdf.read(),
                    file_name=Path(st.session_state.last_report_path).name,
                    mime="application/pdf",
                    key="btn_dl_final_pdf"
                )

        st.divider()
        st.subheader("Export CSV Data")
        all_obs = db_manager.get_all_observations_df()
        if not all_obs.empty:
            st.download_button(
                "⬇️ Export All Observations (CSV)",
                all_obs.to_csv(index=False).encode("utf-8"),
                file_name=f"observations_{rep_id}.csv",
                mime="text/csv",
                key="btn_dl_final_csv"
            )

    st.divider()
    st.subheader("📹 CCTV Video Forensic Summaries Included in Dossier")
    st.caption("AI-synthesized activity overview for each camera feed, included directly in Section 3 of the PDF report.")

    sources = db_manager.get_all_sources()
    all_obs = db_manager.get_all_observations_df()
    summaries = report_generator.build_video_analytical_summaries(sources, all_obs)

    if summaries:
        for vid in summaries:
            with st.expander(f"📹 {vid['camera_id']}: {vid['source_name']} ({vid['duration_str']} | {vid['total_detections']} Detections)", expanded=True):
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
                st.info(vid["narrative"].replace("<b>", "**").replace("</b>", "**").replace("<i>", "*").replace("</i>", "*"))
    else:
        st.info("No CCTV video sources registered yet.")

# =========================================================
# 6. APPLICATION FOOTER
# =========================================================

st.divider()
st.caption(
    "AI Crime Scene & CCTV Investigation Assistant | Multimodal AI for Evidence Linking and Incident Reconstruction. "
    "All AI-generated detections and proposed links require human verification. Digital chain of custody protected via SHA-256."
)
