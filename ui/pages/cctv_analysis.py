"""CCTV analysis: YOLOv8 detection and ByteTrack tracking."""

import os
from pathlib import Path

import streamlit as st

from core.config import (
    ANNOTATED_VIDEOS_DIR,
)
from services.detection import yolo_detector
from services.media import media_processor
from storage import db_manager
from ui.components import page_header
from ui.state import (
    device_info,
    sync_session_from_database,
)


def render():
    device_id, device_str = device_info()
    page_header(
        "CCTV analysis",
        f"YOLOv8 Nano detection with ByteTrack tracking on {device_str}. Every frame is tracked; evidence keyframes "
        "are sampled at the chosen interval. Track IDs are anonymous and never identify a person.",
        eyebrow="Analysis",
    )

    cctv_sources = [s for s in db_manager.get_all_sources("CCTV")]

    if not cctv_sources:
        st.info("No CCTV recordings registered yet.")
        st.page_link("ui/pages/evidence_upload.py", label="Upload CCTV recordings", icon=":material/upload:")
    else:
        src_map = {f"{s['camera_id']} - {s['source_name']}": s for s in cctv_sources}

        col_opt1, col_opt2, col_opt3 = st.columns(3)

        with col_opt1:
            selected_video_keys = st.multiselect(
                "Recordings to analyse",
                options=list(src_map.keys()),
                default=[k for k, s in src_map.items() if s.get("processing_status") != "Processed"][:1] or list(src_map.keys())[:1],
                key="select_vids_for_yolo"
            )

        with col_opt2:
            conf_thresh = st.slider(
                "Minimum confidence",
                min_value=0.10,
                max_value=0.90,
                value=0.35,
                step=0.05,
                key="yolo_conf_slider",
                help="Minimum confidence score for saving detected objects"
            )

        with col_opt3:
            frame_interval = st.slider(
                "Keyframe interval (seconds)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.5,
                key="yolo_interval_slider",
                help="ByteTrack tracks continuously; evidence snapshots are extracted at this interval."
            )

        target_classes_option = st.multiselect(
            "Object classes (empty = all 80 COCO classes)",
            options=["person", "backpack", "handbag", "suitcase", "car", "motorcycle", "bicycle", "truck", "bus", "knife", "cell phone"],
            default=["person", "backpack", "handbag", "car", "motorcycle", "bicycle"],
            key="yolo_class_filter"
        )

        if st.button("Run detection and tracking", type="primary", key="btn_run_yolo"):
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

                    try:
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
                    except Exception as e:
                        # One bad file no longer aborts the whole batch.
                        st.error(f"Detection failed for {rec['camera_id']} ({rec['source_name']}): {e}")
                        continue

                    db_manager.save_observations_batch(detections)
                    db_manager.set_processing_status(rec["source_id"], "Processed")
                    db_manager.log_investigator_action(
                        action="Ran YOLO detection",
                        entity_type="CCTV",
                        entity_id=rec["source_id"],
                        details=f"{rec['camera_id']}: {len(detections)} observation(s), conf>={conf_thresh}, interval {frame_interval}s"
                    )
                    total_new_detections += len(detections)
                    all_previews.extend(previews)
                    st.session_state.active_annotated_video = annotated_vid_path

                sync_session_from_database()
                st.session_state.preview_images = all_previews

                status_txt.empty()
                progress_bar.progress(1.0)
                st.success(f"Analysis complete: {total_new_detections} observation(s) recorded. Re-running a recording "
                           "updates existing observations instead of duplicating them.")

    # Detection Results & Annotated Video Player
    det_df = db_manager.get_all_observations_df(modality="CCTV")
    if not det_df.empty:
        st.divider()
        st.subheader("Results")

        col_player, col_table = st.columns([1.5, 2.5])

        with col_player:
            st.markdown("**Annotated video**")
            annotated_list = list(ANNOTATED_VIDEOS_DIR.glob("*_annotated.mp4"))
            if annotated_list:
                cams = {s["source_id"]: s["camera_id"] for s in db_manager.get_all_sources("CCTV")}
                vid_choices = {f"{cams.get(v.name.split('_')[0], '?')} · {v.name}": v
                               for v in sorted(annotated_list, key=os.path.getmtime, reverse=True)}
                default_idx = 0
                if st.session_state.active_annotated_video:
                    act_name = Path(st.session_state.active_annotated_video).name
                    names = [v.name for v in vid_choices.values()]
                    if act_name in names:
                        default_idx = names.index(act_name)

                chosen_vid_name = st.selectbox(
                    "Recording",
                    options=list(vid_choices.keys()),
                    index=default_idx,
                    key="select_ann_video_to_play"
                )
                target_vid = vid_choices[chosen_vid_name]

                # Ensure web-browser H.264 compatibility
                web_vid_path = media_processor.convert_to_web_h264(str(target_vid))

                # Stream from disk instead of loading the whole video into memory.
                st.video(web_vid_path)

                with open(web_vid_path, "rb") as f_vid:
                    vid_bytes = f_vid.read()
                st.download_button(
                    "Download annotated video (MP4)",
                    vid_bytes,
                    file_name=target_vid.name,
                    mime="video/mp4",
                    key=f"btn_dl_ann_{target_vid.stem}"
                )
            else:
                st.info("Run detection above to generate and preview annotated video.")

        with col_table:
            st.markdown("**Detections**")
            display_cols = ["Observation ID", "Camera ID", "Timestamp", "Object", "Track ID", "Confidence", "Review Status"]
            valid_cols = [c for c in display_cols if c in det_df.columns]
            st.dataframe(det_df[valid_cols], width="stretch", hide_index=True, height=360)

            st.download_button(
                "Download detections (CSV)",
                det_df.to_csv(index=False).encode("utf-8"),
                file_name="cctv_detections.csv",
                mime="text/csv",
                key="btn_dl_det_csv"
            )

        # Evidence Snapshot Previews
        if st.session_state.preview_images:
            st.divider()
            st.subheader("Keyframes from the last run")
            preview_cols = st.columns(min(4, len(st.session_state.preview_images)))
            for idx, prev in enumerate(st.session_state.preview_images[:4]):
                with preview_cols[idx]:
                    st.image(prev["image"], caption=prev["caption"], width="stretch")


if __name__ == "__main__":
    render()
