"""Image evidence: object detection on scene photographs."""

from pathlib import Path

import streamlit as st

from services.detection import yolo_detector
from storage import db_manager
from ui.components import page_header
from ui.state import sync_session_from_database


def render():
    page_header(
        "Image evidence",
        "Run object detection on scene photographs. Photographs carry no verified capture time, so they are linked "
        "to video only by object class and remain low-confidence context until an investigator reviews them.",
        eyebrow="Analysis",
    )
    photo_sources = db_manager.get_all_sources("Photo")
    if not photo_sources:
        st.info("No photographs registered yet.")
        st.page_link("ui/pages/evidence_upload.py", label="Upload photographs", icon=":material/upload:")
        return

    options = {f"{s['camera_id']} · {s['source_name']}": s for s in photo_sources}
    unprocessed = [k for k, s in options.items() if s.get("processing_status") != "Processed"]
    c1, c2 = st.columns([3, 1])
    selected = c1.multiselect("Photographs to analyse", list(options), default=unprocessed, key="select_photos_for_analysis")
    conf = c2.slider("Minimum confidence", 0.10, 0.90, 0.35, 0.05, key="slider_photo_conf")

    if st.button("Analyse selected photographs", type="primary", key="btn_run_photo_analysis", disabled=not selected):
        total = 0
        progress = st.progress(0.0)
        for i, key in enumerate(selected, 1):
            rec = options[key]
            try:
                detections, _, _ = yolo_detector.run_photo_detection(
                    image_path=rec["file_path"], photo_id=rec["source_id"], source_name=rec["source_name"],
                    camera_id=rec["camera_id"], conf_threshold=conf)
            except Exception as e:
                st.error(f"Could not analyse {rec['source_name']}: {e}")
                continue
            db_manager.save_observations_batch(detections)
            db_manager.set_processing_status(rec["source_id"], "Processed")
            db_manager.log_investigator_action("Ran photo detection", "Photo", rec["source_id"],
                                               f"{len(detections)} object(s), conf>={conf}")
            total += len(detections)
            progress.progress(i / len(selected))
        sync_session_from_database()
        st.success(f"Analysis complete: {total} object observation(s) recorded.")

    obs = db_manager.get_all_observations_df(modality="Photo")
    if obs.empty:
        return
    st.subheader("Results")
    for sid, group in obs.groupby("Source ID"):
        src = db_manager.get_source_by_id(sid) or {}
        with st.container(border=True):
            left, right = st.columns([2, 3])
            annotated = group["Evidence Path"].dropna().iloc[0] if not group["Evidence Path"].dropna().empty else ""
            with left:
                if annotated and Path(annotated).is_file():
                    st.image(annotated, width="stretch")
                elif src.get("file_path") and Path(src["file_path"]).is_file():
                    st.image(src["file_path"], width="stretch")
            with right:
                st.markdown(f"**{src.get('camera_id', sid)}** · {src.get('source_name', '')}")
                summary = group["Object"].value_counts()
                st.caption(", ".join(f"{n} × {c}" for c, n in summary.items()))
                st.dataframe(group[["Observation ID", "Object", "Confidence", "Review Status"]], hide_index=True,
                             width="stretch")


if __name__ == "__main__":
    render()
