"""Settings: evidence-source registry (camera IDs, locations, recording clocks), hardware, sample data and case reset."""

import shutil
from datetime import datetime

import pandas as pd
import streamlit as st
import torch

from core.config import ARCHIVE_DIR, MASTER_DB_PATH, UPLOAD_DIR
from services.events.event_reconstruction import deactivate_all_events
from services.media import media_processor
from storage import db_manager
from ui.components import page_header
from ui.state import device_info, sync_session_from_database


def _sources_registry():
    st.subheader("Evidence sources")
    st.caption(
        "Recording start times put recordings on a shared clock, which is what allows cross-camera and audio-to-video "
        "comparisons. Enter only verified times (from the recorder or an evidence log), not file or message timestamps. "
        "Locations are free-text site labels; cameras sharing a label are treated as the same place."
    )
    sources = db_manager.get_all_sources()
    if not sources:
        st.info("No evidence registered yet.")
        return
    df = pd.DataFrame([{
        "source_id": s["source_id"], "Camera / channel": s["camera_id"], "File": s["source_name"],
        "Modality": s["modality"], "Location": s.get("location") or "",
        "Recording start (YYYY-MM-DD HH:MM:SS)": s.get("recorded_start") or "",
    } for s in sources])
    edited = st.data_editor(
        df, hide_index=True, width="stretch", key="settings_sources_editor",
        disabled=["source_id", "Camera / channel", "File", "Modality"],
        column_config={"source_id": st.column_config.TextColumn("Source ID")},
    )
    if st.button("Save source details", type="primary", key="btn_save_sources"):
        changed, errors = 0, []
        original = df.set_index("source_id")
        for _, row in edited.iterrows():
            sid = row["source_id"]
            loc = str(row["Location"] or "").strip()
            start = str(row["Recording start (YYYY-MM-DD HH:MM:SS)"] or "").strip()
            if loc == original.at[sid, "Location"] and start == original.at[sid, "Recording start (YYYY-MM-DD HH:MM:SS)"]:
                continue
            if start:
                try:
                    datetime.fromisoformat(start)
                except ValueError:
                    errors.append(f"{row['File']}: '{start}' is not a valid date-time (use YYYY-MM-DD HH:MM:SS).")
                    continue
            db_manager.update_source_metadata(sid, location=loc, recorded_start=start)
            changed += 1
        for e in errors:
            st.error(e)
        if changed:
            st.toast(f"Updated {changed} source(s). Changes are recorded in the audit trail.")
            st.rerun()
        elif not errors:
            st.toast("No changes to save.")


def _hardware():
    st.subheader("Processing hardware")
    _, device_str = device_info()
    rows = {
        "Detection device": device_str,
        "Vision model": "YOLOv8 Nano (Ultralytics) with ByteTrack",
        "Speech model": "Faster-Whisper on CPU (int8), keeping GPU memory for detection",
        "Graph engine": "NetworkX (Louvain, Dijkstra, centrality) rendered with vis-network",
        "Case database": f"SQLite WAL, schema v{db_manager.get_schema_version()} ({MASTER_DB_PATH.name})",
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        rows["GPU memory"] = f"{props.total_memory / 1024 ** 3:.1f} GB"
    st.dataframe(pd.DataFrame(rows.items(), columns=["Component", "Value"]), hide_index=True, width="stretch")
    if not torch.cuda.is_available():
        st.warning("No CUDA GPU detected. Detection runs on the CPU and will be slower.")


def _sample_loader():
    samples = sorted(ARCHIVE_DIR.glob("*.mp4")) if ARCHIVE_DIR.exists() else []
    if not samples:
        return
    st.subheader("Archived sample footage")
    st.caption(f"{len(samples)} video(s) in {ARCHIVE_DIR.name}/. Already-registered files are skipped (matched by SHA-256).")
    if st.button("Register archived videos", key="btn_load_archived_samples"):
        loaded = 0
        for sample_file in samples:
            target = UPLOAD_DIR / sample_file.name
            if not target.exists():
                shutil.copy2(sample_file, target)
            short_id, full_sha256 = media_processor.hash_file_stream(target)
            if db_manager.get_source_by_id(short_id):
                continue
            meta = media_processor.get_video_metadata(str(target))
            cam_id = db_manager.next_camera_id()
            db_manager.save_source_record(
                source_id=short_id, source_name=sample_file.name, file_path=str(target), modality="CCTV",
                camera_id=cam_id, file_size_bytes=target.stat().st_size, sha256_hash=full_sha256,
                duration_seconds=meta["duration_seconds"], fps=meta["fps"], total_frames=meta["total_frames"],
                width=meta["width"], height=meta["height"],
            )
            db_manager.log_investigator_action("Registered sample CCTV video", "CCTV", short_id,
                                               f"File: {sample_file.name}, Camera: {cam_id}")
            loaded += 1
        sync_session_from_database()
        st.toast(f"{loaded} new video(s) registered." if loaded else "All archived videos are already registered.")
        st.rerun()


def _case_reset():
    st.subheader("Reset case data")
    st.caption("Files on disk are never deleted, and every reset is written to the audit trail.")
    confirm = st.checkbox("I understand this cannot be undone", key="chk_confirm_reset")
    c1, c2 = st.columns(2)
    if c1.button("Clear all observations", key="btn_clear_obs", disabled=not confirm):
        removed = db_manager.clear_all_observations()
        deactivate_all_events()
        sync_session_from_database()
        st.session_state.preview_images = []
        st.toast(f"Cleared {removed} observation(s) and retired candidate events.")
        st.rerun()
    if c2.button("Remove all sources and observations", key="btn_clear_all_sources", disabled=not confirm):
        for src in db_manager.get_all_sources():
            db_manager.delete_source_record(src["source_id"])
        db_manager.clear_all_observations()
        deactivate_all_events()
        sync_session_from_database()
        st.session_state.preview_images = []
        st.session_state.active_annotated_video = ""
        st.toast("All sources and observations removed from the case.")
        st.rerun()


def render():
    page_header("Settings", "Evidence-source details, processing hardware, sample data and case reset.",
                eyebrow="Administration")
    _sources_registry()
    st.divider()
    _hardware()
    st.divider()
    _sample_loader()
    _case_reset()


if __name__ == "__main__":
    render()
