"""Evidence intake: upload, hash and register CCTV recordings, photographs and audio."""

from pathlib import Path

import pandas as pd
import streamlit as st

from core.config import AUDIO_TYPES, IMAGE_TYPES, VIDEO_TYPES
from services.media import media_processor
from services.media.intake import register_upload
from storage import db_manager
from ui.components import page_header
from ui.state import sync_session_from_database, unprocessed_uploads


def _intake(files, modality: str, state_key: str) -> None:
    pending, seen = unprocessed_uploads(files or [], state_key)
    if not pending:
        return
    registered = 0
    with st.spinner(f"Hashing and registering {len(pending)} file(s)..."):
        for f in pending:
            seen.add(getattr(f, "file_id", f.name))
            result = register_upload(f, modality)
            if result.status == "registered":
                registered += 1
            elif result.status == "duplicate":
                st.info(result.message)
            else:
                st.error(result.message)
    if registered:
        sync_session_from_database()
        st.toast(f"Registered {registered} {modality.lower()} file(s).")
        st.rerun()


def _cctv_registry():
    cctv_sources = db_manager.get_all_sources("CCTV")
    if not cctv_sources:
        st.caption("No CCTV recordings registered yet.")
        return
    for src in cctv_sources:
        with st.container(border=True):
            c_thumb, c_meta, c_ctrl = st.columns([1.4, 3.6, 2])
            with c_thumb:
                meta = media_processor.get_video_metadata(src["file_path"])
                if meta.get("thumbnail_path") and Path(meta["thumbnail_path"]).is_file():
                    st.image(meta["thumbnail_path"], width="stretch")
                elif not Path(src["file_path"]).is_file():
                    st.error("File missing on disk")
            with c_meta:
                st.markdown(f"**{src['camera_id']}** · {src['source_name']}")
                st.caption(
                    f"{src['duration_seconds']:.1f}s · {src['fps']} fps · {src['total_frames']} frames · "
                    f"{src['width']}x{src['height']} · {src['file_size_bytes'] / 1024 ** 2:.1f} MB · "
                    f"{src.get('processing_status') or 'Registered'}"
                )
                st.caption(f"Recording start: {src.get('recorded_start') or 'not set'} · "
                           f"Location: {src.get('location') or 'not set'}")
                st.code(src["sha256_hash"], language=None)
            with c_ctrl:
                new_cam = st.text_input("Camera ID", value=src["camera_id"], key=f"edit_cam_{src['source_id']}").strip()
                if new_cam != src["camera_id"] and st.button("Rename camera", key=f"btn_upd_cam_{src['source_id']}"):
                    taken = {s["camera_id"] for s in cctv_sources if s["source_id"] != src["source_id"]}
                    if not new_cam:
                        st.error("Camera ID cannot be empty.")
                    elif new_cam in taken:
                        st.error(f"{new_cam} is already used by another recording.")
                    else:
                        db_manager.update_source_camera_id(src["source_id"], new_cam)
                        sync_session_from_database()
                        st.toast(f"Camera renamed to {new_cam}")
                        st.rerun()
                confirm_del = st.checkbox("Confirm removal", key=f"chk_del_vid_{src['source_id']}")
                if st.button("Remove from case", key=f"btn_del_vid_{src['source_id']}", disabled=not confirm_del):
                    db_manager.delete_source_record(src["source_id"])
                    sync_session_from_database()
                    st.toast(f"Removed {src['source_name']} (file kept on disk; logged in the audit trail).")
                    st.rerun()


def _simple_registry(modality: str):
    sources = db_manager.get_all_sources(modality)
    if not sources:
        st.caption(f"No {modality.lower()} evidence registered yet.")
        return
    st.dataframe(pd.DataFrame([{
        "ID": s["camera_id"], "File": s["source_name"], "Size (MB)": round(s["file_size_bytes"] / 1024 ** 2, 2),
        "Status": s.get("processing_status") or "Registered", "Uploaded": s["upload_time"], "SHA-256": s["sha256_hash"],
    } for s in sources]), hide_index=True, width="stretch")


def render():
    page_header(
        "Evidence upload",
        "Files are stored unmodified, fingerprinted with SHA-256 at intake, validated and given a unique source ID. "
        "Re-uploading an identical file is detected and not registered twice.",
        eyebrow="Intake",
    )
    t_cctv, t_photo, t_audio = st.tabs(["CCTV recordings", "Photographs", "Audio recordings"])
    with t_cctv:
        files = st.file_uploader("CCTV recordings", type=VIDEO_TYPES, accept_multiple_files=True,
                                 key="cctv_file_uploader_widget")
        _intake(files, "CCTV", "_seen_cctv_uploads")
        st.caption("Set each recording's verified start time and location in Settings so recordings can be compared "
                   "on one clock.")
        _cctv_registry()
    with t_photo:
        files = st.file_uploader("Scene photographs", type=IMAGE_TYPES, accept_multiple_files=True,
                                 key="photo_uploader_widget")
        _intake(files, "Photo", "_seen_photo_uploads")
        _simple_registry("Photo")
    with t_audio:
        files = st.file_uploader("Audio recordings", type=AUDIO_TYPES, accept_multiple_files=True,
                                 key="audio_file_uploader_widget")
        _intake(files, "Audio", "_seen_audio_uploads")
        _simple_registry("Audio")


if __name__ == "__main__":
    render()
