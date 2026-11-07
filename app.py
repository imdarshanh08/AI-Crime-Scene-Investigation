
import hashlib
import re
from datetime import datetime
from pathlib import Path

import cv2
import networkx as nx
import pandas as pd
import streamlit as st
import torch
from ultralytics import YOLO

from faster_whisper import WhisperModel

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    LongTable
)


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Crime Scene Investigation Assistant",
    page_icon="🔍",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent

UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
EVIDENCE_DIR = OUTPUT_DIR / "evidence_frames"
PHOTO_EVIDENCE_DIR = OUTPUT_DIR / "photo_evidence"
AUDIO_EVIDENCE_DIR = OUTPUT_DIR / "audio_evidence"
REPORT_DIR = OUTPUT_DIR / "reports"

for folder in [
    UPLOAD_DIR,
    OUTPUT_DIR,
    EVIDENCE_DIR,
    PHOTO_EVIDENCE_DIR,
    AUDIO_EVIDENCE_DIR,
    REPORT_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. AI MODEL LOADERS
# =========================================================

@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")


@st.cache_resource
def load_whisper_model(model_size):
    # CPU int8 avoids competing with YOLO for 4 GB GPU VRAM.
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8"
    )


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def format_timestamp(seconds):
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes:02d}:{remaining:05.2f}"


def safe_name(name):
    return re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        Path(name).name
    )


def make_id(data):
    return hashlib.sha256(data).hexdigest()[:12]


def class_slug(name):
    return re.sub(
        r"[^a-z0-9]+",
        "-",
        str(name).lower()
    ).strip("-")


def dot_escape(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
    )


def current_time():
    return datetime.now().astimezone().strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def save_uploaded_file(uploaded_file, folder):
    data = uploaded_file.getvalue()
    file_id = make_id(data)

    path = folder / f"{file_id}_{safe_name(uploaded_file.name)}"

    if not path.exists():
        path.write_bytes(data)

    return {
        "id": file_id,
        "name": uploaded_file.name,
        "path": str(path)
    }


def initialise_review(row):
    row.setdefault("Review Status", "Pending")
    row.setdefault("Review Notes", "")
    row.setdefault("Reviewed At", "")
    return row


def all_observations():
    rows = []

    rows.extend(st.session_state.detection_rows)
    rows.extend(st.session_state.photo_rows)
    rows.extend(st.session_state.audio_rows)

    return rows


def observation_dataframe():
    rows = all_observations()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =========================================================
# 4. SESSION STATE
# =========================================================

defaults = {
    "video_records": {},
    "photo_records": {},
    "audio_records": {},
    "detection_rows": [],
    "photo_rows": [],
    "audio_rows": [],
    "preview_images": [],
    "last_report_path": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 5. APPLICATION HEADER
# =========================================================

st.title("🔍 AI Crime Scene & CCTV Investigation Assistant")

st.markdown(
    "### Multimodal AI for Evidence Linking and Incident Reconstruction"
)

st.caption(
    "AI-assisted CCTV analysis, image evidence, audio transcription, "
    "evidence relationships and incident reconstruction."
)

st.divider()


# =========================================================
# 6. SIDEBAR
# =========================================================

st.sidebar.title("Investigation Panel")

device = 0 if torch.cuda.is_available() else "cpu"

st.sidebar.write(
    "**YOLO Processing:**",
    "NVIDIA GPU (CUDA)" if device == 0 else "CPU"
)

st.sidebar.write("**Object Detection:** YOLOv8 Nano")
st.sidebar.write("**Object Tracking:** ByteTrack")
st.sidebar.write("**Audio Transcription:** Faster-Whisper")
st.sidebar.write("**Evidence Graph:** NetworkX")

if torch.cuda.is_available():
    st.sidebar.write(
        f"**GPU:** {torch.cuda.get_device_name(0)}"
    )

st.sidebar.caption(
    "Audio transcription uses CPU int8 to reduce GPU memory usage."
)


# =========================================================
# 7. DASHBOARD METRICS
# =========================================================

detection_df = pd.DataFrame(
    st.session_state.detection_rows
)

photo_df = pd.DataFrame(
    st.session_state.photo_rows
)

audio_df = pd.DataFrame(
    st.session_state.audio_rows
)

all_df = observation_dataframe()

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "CCTV Videos",
    len(st.session_state.video_records)
)

m2.metric(
    "Photos",
    len(st.session_state.photo_records)
)

m3.metric(
    "Audio Files",
    len(st.session_state.audio_records)
)

m4.metric(
    "Total Observations",
    len(all_df)
)

st.divider()


# =========================================================
# 8. CCTV VIDEO UPLOAD
# =========================================================

st.header("📹 CCTV Video Upload")

uploaded_videos = st.file_uploader(
    "Upload CCTV recordings",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=True,
    key="cctv_upload"
)

if uploaded_videos:

    for uploaded_file in uploaded_videos:

        record = save_uploaded_file(
            uploaded_file,
            UPLOAD_DIR
        )

        st.session_state.video_records[record["id"]] = record

    st.success("CCTV recording(s) added successfully.")


if st.session_state.video_records:

    st.write("**Available CCTV recordings**")

    for record in st.session_state.video_records.values():
        st.write(f"📁 {record['name']}")

else:

    st.info("Upload a CCTV recording to begin.")


# =========================================================
# 9. CCTV OBJECT DETECTION AND TRACKING
# =========================================================

st.divider()

st.header("🤖 CCTV Object Detection & Tracking")

if st.session_state.video_records:

    video_options = {
        f"{record['name']} ({record['id']})": record
        for record in st.session_state.video_records.values()
    }

    selected_videos = st.multiselect(
        "Select recordings to analyse",
        options=list(video_options.keys())
    )

    frame_interval = st.slider(
        "Evidence frame interval (seconds)",
        0.5,
        5.0,
        1.0,
        0.5
    )

    confidence = st.slider(
        "Detection confidence",
        0.10,
        0.90,
        0.35,
        0.05
    )

    st.caption(
        "ByteTrack processes every frame. Detection records and "
        "evidence images are saved at the selected interval."
    )

    if st.button(
        "🚀 Start Detection & Tracking",
        type="primary"
    ):

        if not selected_videos:

            st.warning("Select at least one CCTV recording.")

        else:

            all_detections = []
            all_previews = []

            try:

                with st.spinner("Loading YOLO model..."):
                    model = load_yolo_model()

                for video_label in selected_videos:

                    record = video_options[video_label]

                    st.write(f"Analysing: **{record['name']}**")

                    model.predictor = None

                    cap = cv2.VideoCapture(record["path"])

                    if not cap.isOpened():
                        st.error(
                            f"Could not open {record['name']}."
                        )
                        continue

                    try:

                        fps = cap.get(cv2.CAP_PROP_FPS)

                        if fps <= 0:
                            fps = 25.0

                        total_frames = int(
                            cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        )

                        frame_step = max(
                            1,
                            int(round(fps * frame_interval))
                        )

                        frame_index = 0

                        progress = st.progress(0)

                        while True:

                            success, frame = cap.read()

                            if not success:
                                break

                            results = model.track(
                                source=frame,
                                persist=True,
                                tracker="bytetrack.yaml",
                                device=device,
                                imgsz=640,
                                conf=confidence,
                                verbose=False
                            )

                            result = results[0]

                            if frame_index % frame_step == 0:

                                timestamp = frame_index / fps

                                if (
                                    result.boxes is not None
                                    and len(result.boxes) > 0
                                ):

                                    annotated = result.plot()

                                    frame_path = (
                                        EVIDENCE_DIR
                                        / (
                                            f"frame_{record['id']}_"
                                            f"{frame_index}.jpg"
                                        )
                                    )

                                    image_saved = cv2.imwrite(
                                        str(frame_path),
                                        annotated
                                    )

                                    for box in result.boxes:

                                        class_id = int(box.cls[0])
                                        object_name = model.names[class_id]

                                        score = float(box.conf[0])

                                        coordinates = [
                                            round(float(v), 1)
                                            for v in box.xyxy[0].tolist()
                                        ]

                                        if box.id is not None:

                                            raw_id = int(
                                                box.id[0].item()
                                            )

                                            raw_track_id = f"T{raw_id}"

                                            track_id = (
                                                f"{record['id'][:6]}-"
                                                f"T{raw_id}-"
                                                f"{class_slug(object_name)}"
                                            )

                                        else:

                                            raw_track_id = "Untracked"
                                            track_id = "Untracked"

                                        observation_id = hashlib.sha1(
                                            (
                                                f"{record['id']}_"
                                                f"{frame_index}_"
                                                f"{track_id}_"
                                                f"{object_name}"
                                            ).encode("utf-8")
                                        ).hexdigest()[:12]

                                        row = {
                                            "Observation ID": observation_id,
                                            "Modality": "CCTV",
                                            "Source": record["name"],
                                            "Source ID": record["id"],
                                            "Timestamp": format_timestamp(timestamp),
                                            "Timestamp (seconds)": round(timestamp, 2),
                                            "Object": object_name,
                                            "Raw Track ID": raw_track_id,
                                            "Track ID": track_id,
                                            "Confidence": round(score, 3),
                                            "Bounding Box": str(coordinates),
                                            "Observation": (
                                                f"{object_name} detected in CCTV footage."
                                            ),
                                            "Original Transcript": "",
                                            "English Translation": "",
                                            "Evidence Path": (
                                                str(frame_path)
                                                if image_saved else ""
                                            )
                                        }

                                        all_detections.append(
                                            initialise_review(row)
                                        )

                                    if len(all_previews) < 2 * len(selected_videos):

                                        rgb = cv2.cvtColor(
                                            annotated,
                                            cv2.COLOR_BGR2RGB
                                        )

                                        all_previews.append({
                                            "image": rgb,
                                            "caption": (
                                                f"{record['name']} | "
                                                f"{format_timestamp(timestamp)}"
                                            )
                                        })

                            frame_index += 1

                            if (
                                total_frames > 0
                                and frame_index % 30 == 0
                            ):

                                progress.progress(
                                    min(
                                        frame_index / total_frames,
                                        1.0
                                    )
                                )

                    finally:
                        cap.release()

                    progress.progress(1.0)

                    st.success(
                        f"Completed: {record['name']}"
                    )

                # Preserve earlier analyses from other modalities.
                st.session_state.detection_rows = all_detections
                st.session_state.preview_images = all_previews

                st.success(
                    "CCTV detection and tracking completed."
                )

            except Exception as error:

                st.error(f"CCTV analysis failed: {error}")


# =========================================================
# 10. PHOTO EVIDENCE UPLOAD AND ANALYSIS
# =========================================================

st.divider()

st.header("📷 Photo Evidence Upload & Analysis")

uploaded_photos = st.file_uploader(
    "Upload photographs or image evidence",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
    key="photo_upload"
)

if uploaded_photos:

    for uploaded_file in uploaded_photos:

        record = save_uploaded_file(
            uploaded_file,
            UPLOAD_DIR
        )

        st.session_state.photo_records[record["id"]] = record

    st.success("Photo evidence added successfully.")


if st.session_state.photo_records:

    photo_options = {
        f"{record['name']} ({record['id']})": record
        for record in st.session_state.photo_records.values()
    }

    selected_photos = st.multiselect(
        "Select photographs to analyse",
        options=list(photo_options.keys()),
        key="selected_photos"
    )

    photo_confidence = st.slider(
        "Photo detection confidence",
        0.10,
        0.90,
        0.35,
        0.05,
        key="photo_confidence"
    )

    if st.button(
        "🔎 Analyse Selected Photos",
        type="primary"
    ):

        if not selected_photos:

            st.warning("Select at least one photograph.")

        else:

            try:

                model = load_yolo_model()
                new_photo_rows = []

                for photo_label in selected_photos:

                    record = photo_options[photo_label]

                    image = cv2.imread(record["path"])

                    if image is None:
                        st.error(
                            f"Could not read {record['name']}."
                        )
                        continue

                    results = model.predict(
                        source=image,
                        device=device,
                        imgsz=640,
                        conf=photo_confidence,
                        verbose=False
                    )

                    result = results[0]
                    annotated = result.plot()

                    evidence_path = (
                        PHOTO_EVIDENCE_DIR
                        / f"annotated_{record['id']}.jpg"
                    )

                    cv2.imwrite(
                        str(evidence_path),
                        annotated
                    )

                    boxes = result.boxes

                    if boxes is None or len(boxes) == 0:

                        st.info(
                            f"No objects detected in {record['name']}."
                        )
                        continue

                    for box_index, box in enumerate(boxes):

                        class_id = int(box.cls[0])
                        object_name = model.names[class_id]

                        score = float(box.conf[0])

                        coordinates = [
                            round(float(v), 1)
                            for v in box.xyxy[0].tolist()
                        ]

                        observation_id = hashlib.sha1(
                            (
                                f"{record['id']}_"
                                f"{box_index}_"
                                f"{object_name}"
                            ).encode("utf-8")
                        ).hexdigest()[:12]

                        row = {
                            "Observation ID": observation_id,
                            "Modality": "Photo",
                            "Source": record["name"],
                            "Source ID": record["id"],
                            "Timestamp": "Not available",
                            "Timestamp (seconds)": None,
                            "Object": object_name,
                            "Raw Track ID": "Not applicable",
                            "Track ID": "Not applicable",
                            "Confidence": round(score, 3),
                            "Bounding Box": str(coordinates),
                            "Observation": (
                                f"{object_name} detected in uploaded photograph."
                            ),
                            "Original Transcript": "",
                            "English Translation": "",
                            "Evidence Path": str(evidence_path)
                        }

                        new_photo_rows.append(
                            initialise_review(row)
                        )

                    st.image(
                        cv2.cvtColor(
                            annotated,
                            cv2.COLOR_BGR2RGB
                        ),
                        caption=f"Analysed: {record['name']}",
                        use_container_width=True
                    )

                st.session_state.photo_rows = new_photo_rows

                st.success(
                    f"Photo analysis completed. "
                    f"{len(new_photo_rows)} object observation(s) recorded."
                )

            except Exception as error:

                st.error(f"Photo analysis failed: {error}")


# =========================================================
# 11. AUDIO UPLOAD AND TRANSCRIPTION
# =========================================================

st.divider()

st.header("🎙️ Audio Evidence Upload & Transcription")

st.write(
    "Upload audio evidence to generate timestamped original-language "
    "transcripts and English translations."
)

uploaded_audio = st.file_uploader(
    "Upload audio recordings",
    type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
    accept_multiple_files=True,
    key="audio_upload"
)

if uploaded_audio:

    for uploaded_file in uploaded_audio:

        record = save_uploaded_file(
            uploaded_file,
            UPLOAD_DIR
        )

        st.session_state.audio_records[record["id"]] = record

    st.success("Audio evidence added successfully.")


if st.session_state.audio_records:

    audio_options = {
        f"{record['name']} ({record['id']})": record
        for record in st.session_state.audio_records.values()
    }

    selected_audio = st.multiselect(
        "Select audio files to transcribe",
        options=list(audio_options.keys())
    )

    whisper_size = st.selectbox(
        "Whisper model size",
        ["tiny", "base", "small"],
        index=1,
        help=(
            "Base is a practical starting point. Small may improve "
            "transcription quality but takes longer."
        )
    )

    if st.button(
        "🎧 Transcribe & Translate Audio",
        type="primary"
    ):

        if not selected_audio:

            st.warning("Select at least one audio file.")

        else:

            try:

                with st.spinner(
                    f"Loading Whisper {whisper_size} model on CPU..."
                ):
                    whisper = load_whisper_model(whisper_size)

                new_audio_rows = []

                for audio_label in selected_audio:

                    record = audio_options[audio_label]

                    st.write(
                        f"Processing: **{record['name']}**"
                    )

                    with st.spinner(
                        "Transcribing original speech..."
                    ):

                        original_segments, info = whisper.transcribe(
                            record["path"],
                            task="transcribe",
                            beam_size=5,
                            vad_filter=True
                        )

                        original_segments = list(original_segments)

                    with st.spinner(
                        "Generating English translation..."
                    ):

                        translated_segments, translation_info = (
                            whisper.transcribe(
                                record["path"],
                                task="translate",
                                beam_size=5,
                                vad_filter=True
                            )
                        )

                        translated_segments = list(
                            translated_segments
                        )

                    language = getattr(
                        info,
                        "language",
                        "unknown"
                    )

                    language_probability = getattr(
                        info,
                        "language_probability",
                        None
                    )

                    # Match translation segments by order.
                    # The timestamps are taken from the original segments.

                    for index, segment in enumerate(original_segments):

                        translation = ""

                        if index < len(translated_segments):
                            translation = (
                                translated_segments[index].text.strip()
                            )

                        start = float(segment.start)
                        end = float(segment.end)

                        original_text = segment.text.strip()

                        observation_id = hashlib.sha1(
                            (
                                f"{record['id']}_"
                                f"{index}_"
                                f"{start}"
                            ).encode("utf-8")
                        ).hexdigest()[:12]

                        row = {
                            "Observation ID": observation_id,
                            "Modality": "Audio",
                            "Source": record["name"],
                            "Source ID": record["id"],
                            "Timestamp": (
                                f"{format_timestamp(start)} - "
                                f"{format_timestamp(end)}"
                            ),
                            "Timestamp (seconds)": round(start, 2),
                            "Object": "Speech",
                            "Raw Track ID": "Not applicable",
                            "Track ID": "Not applicable",
                            "Confidence": None,
                            "Bounding Box": "",
                            "Observation": (
                                f"Speech segment from "
                                f"{format_timestamp(start)} to "
                                f"{format_timestamp(end)}."
                            ),
                            "Original Transcript": original_text,
                            "English Translation": translation,
                            "Detected Language": language,
                            "Language Probability": (
                                round(language_probability, 3)
                                if language_probability is not None
                                else None
                            ),
                            "Segment End (seconds)": round(end, 2),
                            "Evidence Path": record["path"]
                        }

                        new_audio_rows.append(
                            initialise_review(row)
                        )

                    st.success(
                        f"Completed audio processing: {record['name']}"
                    )

                    st.write(
                        f"**Detected language:** {language}"
                    )

                    if language_probability is not None:
                        st.write(
                            f"**Language detection probability:** "
                            f"{language_probability:.1%}"
                        )

                st.session_state.audio_rows = new_audio_rows

                st.success(
                    f"Audio processing completed. "
                    f"{len(new_audio_rows)} transcript segment(s) generated."
                )

            except Exception as error:

                st.error(
                    f"Audio processing failed: {error}"
                )

else:

    st.info("Upload an audio file to enable transcription.")


# =========================================================
# 12. REFRESH DATA
# =========================================================

detection_df = pd.DataFrame(
    st.session_state.detection_rows
)

photo_df = pd.DataFrame(
    st.session_state.photo_rows
)

audio_df = pd.DataFrame(
    st.session_state.audio_rows
)

all_df = observation_dataframe()


# =========================================================
# 13. CCTV DETECTION RESULTS
# =========================================================

st.divider()

st.header("📊 CCTV Detection Results")

if not detection_df.empty:

    a, b, c = st.columns(3)

    a.metric("CCTV Detections", len(detection_df))
    b.metric("Object Types", detection_df["Object"].nunique())

    tracked_count = detection_df[
        detection_df["Track ID"] != "Untracked"
    ]["Track ID"].nunique()

    c.metric("Class-Safe Track IDs", tracked_count)

    st.dataframe(
        detection_df.drop(
            columns=["Evidence Path"],
            errors="ignore"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download CCTV Detection CSV",
        detection_df.drop(
            columns=["Evidence Path"],
            errors="ignore"
        ).to_csv(index=False).encode("utf-8"),
        "cctv_detection_report.csv",
        "text/csv"
    )

    st.subheader("Annotated CCTV Frames")

    for preview in st.session_state.preview_images:

        st.image(
            preview["image"],
            caption=preview["caption"],
            use_container_width=True
        )

else:

    st.info("CCTV detection results will appear after analysis.")


# =========================================================
# 14. PHOTO RESULTS
# =========================================================

st.divider()

st.header("📷 Photo Evidence Results")

if not photo_df.empty:

    st.metric(
        "Photo Object Observations",
        len(photo_df)
    )

    st.dataframe(
        photo_df.drop(
            columns=["Evidence Path"],
            errors="ignore"
        ),
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("Photo analysis results will appear here.")


# =========================================================
# 15. AUDIO TRANSCRIPT RESULTS
# =========================================================

st.divider()

st.header("🎙️ Audio Transcript & Translation Results")

if not audio_df.empty:

    st.metric(
        "Transcript Segments",
        len(audio_df)
    )

    for _, row in audio_df.iterrows():

        with st.container(border=True):

            st.write(
                f"**{row['Source']} | {row['Timestamp']}**"
            )

            st.write(
                f"**Original ({row.get('Detected Language', 'Unknown')}):** "
                f"{row['Original Transcript']}"
            )

            st.write(
                f"**English translation:** "
                f"{row['English Translation']}"
            )

            st.caption(
                "Translation is an AI-generated rendering; "
                "check it against the original audio."
            )

    st.download_button(
        "⬇️ Download Audio Transcript CSV",
        audio_df.drop(
            columns=["Evidence Path"],
            errors="ignore"
        ).to_csv(index=False).encode("utf-8"),
        "audio_transcripts.csv",
        "text/csv"
    )

else:

    st.info("Audio transcripts will appear after processing.")


# =========================================================
# 16. TRACKING CONSISTENCY CHECK
# =========================================================

st.divider()

st.header("🛡️ Tracking Consistency Check")

if not detection_df.empty:

    tracked = detection_df[
        detection_df["Raw Track ID"] != "Untracked"
    ].copy()

    if not tracked.empty:

        counts = (
            tracked.groupby(
                ["Source ID", "Raw Track ID"]
            )["Object"].nunique()
        )

        conflicts = counts[counts > 1]

        if not conflicts.empty:

            st.warning(
                f"{len(conflicts)} raw tracking ID(s) "
                "were associated with multiple object classes."
            )

            conflict_rows = tracked[
                tracked.set_index(
                    ["Source ID", "Raw Track ID"]
                ).index.isin(conflicts.index)
            ]

            st.dataframe(
                conflict_rows[
                    ["Source", "Raw Track ID", "Object"]
                ].drop_duplicates(),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.success(
                "No sampled raw tracking ID class conflicts found."
            )

        st.caption(
            "This check identifies class conflicts in sampled detections. "
            "It does not prove physical object identity."
        )

    else:

        st.info("No tracked objects found.")


# =========================================================
# 17. OBJECT TRACKING SUMMARY
# =========================================================

st.divider()

st.header("🎯 Object Tracking Summary")

if not detection_df.empty:

    tracked = detection_df[
        detection_df["Track ID"] != "Untracked"
    ].copy()

    if not tracked.empty:

        summary = (
            tracked.groupby(
                ["Source", "Track ID", "Object"],
                as_index=False
            )
            .agg(
                first_seen=("Timestamp (seconds)", "min"),
                last_seen=("Timestamp (seconds)", "max"),
                observations=("Timestamp", "count"),
                average_confidence=("Confidence", "mean")
            )
        )

        summary["First Seen"] = summary[
            "first_seen"
        ].apply(format_timestamp)

        summary["Last Seen"] = summary[
            "last_seen"
        ].apply(format_timestamp)

        summary["Average Confidence"] = summary[
            "average_confidence"
        ].round(3)

        display_summary = summary[
            [
                "Source",
                "Track ID",
                "Object",
                "First Seen",
                "Last Seen",
                "observations",
                "Average Confidence"
            ]
        ].rename(
            columns={
                "Source": "Video",
                "observations": "Sampled Observations"
            }
        )

        st.dataframe(
            display_summary,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Tracking Summary",
            display_summary.to_csv(index=False).encode("utf-8"),
            "object_tracking_summary.csv",
            "text/csv"
        )

        st.caption(
            "First Seen and Last Seen refer to sampled observations, "
            "not necessarily exact entry and exit times."
        )


# =========================================================
# 18. UNIFIED EVIDENCE SEARCH
# =========================================================

st.divider()

st.header("🔎 Unified Evidence Search")

st.write(
    "Search observations across CCTV recordings, photographs "
    "and audio transcripts."
)

if not all_df.empty:

    search_term = st.text_input(
        "Enter an object, word, person description or keyword",
        placeholder="Example: backpack, car, door, help"
    )

    modality_filter = st.selectbox(
        "Evidence type",
        ["All", "CCTV", "Photo", "Audio"]
    )

    search_results = all_df.copy()

    if modality_filter != "All":

        search_results = search_results[
            search_results["Modality"] == modality_filter
        ]

    if search_term.strip():

        searchable_columns = [
            "Object",
            "Source",
            "Observation",
            "Original Transcript",
            "English Translation"
        ]

        mask = pd.Series(
            False,
            index=search_results.index
        )

        for column in searchable_columns:

            if column in search_results.columns:

                mask |= (
                    search_results[column]
                    .fillna("")
                    .astype(str)
                    .str.contains(
                        search_term,
                        case=False,
                        regex=False
                    )
                )

        search_results = search_results[mask]

    st.metric(
        "Matching observations",
        len(search_results)
    )

    st.dataframe(
        search_results.drop(
            columns=["Evidence Path"],
            errors="ignore"
        ),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "This is keyword-based evidence retrieval, not a full "
        "natural-language reasoning model."
    )

else:

    st.info("Upload and analyse evidence to enable unified search.")


# =========================================================
# 19. HUMAN VERIFICATION / INVESTIGATOR REVIEW
# =========================================================

st.divider()

st.header("🧑‍⚖️ Investigator Review & Verification")

st.write(
    "Every AI observation starts as Pending. An investigator can "
    "approve, reject or flag an observation. AI output is retained "
    "separately from the review decision."
)

all_df = observation_dataframe()

if not all_df.empty:

    review_columns = [
        "Observation ID",
        "Modality",
        "Source",
        "Timestamp",
        "Object",
        "Observation",
        "Original Transcript",
        "English Translation",
        "Review Status",
        "Review Notes",
        "Reviewed At"
    ]

    review_df = all_df.copy()

    for column in review_columns:

        if column not in review_df.columns:
            review_df[column] = ""

    review_df = review_df[review_columns]

    edited_review = st.data_editor(
        review_df,
        use_container_width=True,
        hide_index=True,
        disabled=[
            "Observation ID",
            "Modality",
            "Source",
            "Timestamp",
            "Object",
            "Observation",
            "Original Transcript",
            "English Translation",
            "Reviewed At"
        ],
        column_config={
            "Review Status": st.column_config.SelectboxColumn(
                "Review Status",
                options=[
                    "Pending",
                    "Approved",
                    "Rejected",
                    "Flagged"
                ],
                required=True
            ),
            "Review Notes": st.column_config.TextColumn(
                "Investigator Notes",
                help="Add a note explaining the review decision."
            )
        },
        key="investigator_review_editor"
    )

    edited_lookup = {
        row["Observation ID"]: row
        for _, row in edited_review.iterrows()
    }

    for list_name in [
        "detection_rows",
        "photo_rows",
        "audio_rows"
    ]:

        updated_rows = []

        for original in st.session_state[list_name]:

            row = dict(original)

            edited = edited_lookup.get(
                row["Observation ID"]
            )

            if edited is not None:

                old_status = row.get(
                    "Review Status",
                    "Pending"
                )

                new_status = edited["Review Status"]

                row["Review Notes"] = (
                    edited["Review Notes"]
                    if pd.notna(edited["Review Notes"])
                    else ""
                )

                if new_status != old_status:

                    row["Review Status"] = new_status

                    if new_status == "Pending":
                        row["Reviewed At"] = ""
                    else:
                        row["Reviewed At"] = current_time()

                else:

                    row["Review Status"] = old_status

            updated_rows.append(row)

        st.session_state[list_name] = updated_rows

    reviewed_df = observation_dataframe()

    if not reviewed_df.empty:

        counts = reviewed_df[
            "Review Status"
        ].value_counts()

        r1, r2, r3, r4 = st.columns(4)

        r1.metric("Pending", counts.get("Pending", 0))
        r2.metric("Approved", counts.get("Approved", 0))
        r3.metric("Rejected", counts.get("Rejected", 0))
        r4.metric("Flagged", counts.get("Flagged", 0))

        st.caption(
            "Review decisions are maintained in the current application "
            "session. Export the PDF report before closing or refreshing "
            "the application."
        )

else:

    st.info("There are no observations to review yet.")


# =========================================================
# 20. EVIDENCE RELATIONSHIP GRAPH
# =========================================================

st.divider()

st.header("🕸️ Multimodal Evidence Relationship Graph")

st.write(
    "Explore relationships between source files, observations, "
    "tracked objects, transcript segments and evidence images."
)

graph_df = observation_dataframe()

if not graph_df.empty:

    graph = nx.DiGraph()

    graph_observations = graph_df.copy()

    graph_observations["_sort_time"] = pd.to_numeric(
        graph_observations["Timestamp (seconds)"],
        errors="coerce"
    )

    graph_observations = graph_observations.sort_values(
        "_sort_time",
        na_position="last"
    )

    graph_limit = st.slider(
        "Maximum observations to display",
        min_value=20,
        max_value=300,
        value=min(80, max(20, len(graph_observations))),
        step=20
    )

    total_observations = len(graph_observations)

    graph_observations = graph_observations.head(graph_limit)

    for _, row in graph_observations.iterrows():

        modality = str(row["Modality"])
        source_id = str(row["Source ID"])
        source_name = str(row["Source"])

        source_node = f"source_{modality}_{source_id}"

        observation_id = str(row["Observation ID"])
        observation_node = f"observation_{observation_id}"

        graph.add_node(
            source_node,
            label=f"{modality}: {source_name}",
            kind="source"
        )

        graph.add_node(
            observation_node,
            label=(
                f"{row['Object']} | "
                f"{row['Timestamp']}"
            ),
            kind="observation"
        )

        graph.add_edge(
            source_node,
            observation_node
        )

        if modality == "CCTV":

            track_id = str(row["Track ID"])

            track_node = (
                f"track_{source_id}_{track_id}"
            )

            graph.add_node(
                track_node,
                label=f"Tracked object: {track_id}",
                kind="track"
            )

            graph.add_edge(
                source_node,
                track_node
            )

            graph.add_edge(
                track_node,
                observation_node
            )

        if modality == "Audio":

            transcript_node = (
                f"transcript_{observation_id}"
            )

            graph.add_node(
                transcript_node,
                label=(
                    "Transcript: "
                    + str(row["Original Transcript"])[:70]
                ),
                kind="transcript"
            )

            graph.add_edge(
                observation_node,
                transcript_node
            )

        evidence_path = str(row.get("Evidence Path", ""))

        if evidence_path and Path(evidence_path).is_file():

            evidence_node = (
                "evidence_"
                + hashlib.sha1(
                    evidence_path.encode("utf-8")
                ).hexdigest()[:12]
            )

            graph.add_node(
                evidence_node,
                label=f"Evidence: {Path(evidence_path).name}",
                kind="evidence"
            )

            graph.add_edge(
                observation_node,
                evidence_node
            )

    st.metric(
        "Graph Nodes",
        graph.number_of_nodes()
    )

    st.metric(
        "Graph Relationships",
        graph.number_of_edges()
    )

    if total_observations > len(graph_observations):

        st.caption(
            f"Showing {len(graph_observations)} of "
            f"{total_observations} observations."
        )

    dot_lines = [
        "digraph EvidenceGraph {",
        'rankdir="LR";',
        'bgcolor="transparent";',
        'node [style="rounded,filled", fontname="Arial"];',
        'edge [arrowsize=0.7];'
    ]

    for node, attributes in graph.nodes(data=True):

        label = dot_escape(
            attributes.get("label", node)
        )

        kind = attributes.get("kind")

        if kind == "source":

            shape = "folder"
            fill = "lightblue"

        elif kind == "track":

            shape = "box"
            fill = "lightgoldenrod"

        elif kind == "observation":

            shape = "ellipse"
            fill = "lightgreen"

        elif kind == "transcript":

            shape = "note"
            fill = "lightpink"

        else:

            shape = "component"
            fill = "lightgrey"

        dot_lines.append(
            f'"{dot_escape(node)}" '
            f'[label="{label}", shape="{shape}", '
            f'fillcolor="{fill}"];'
        )

    for source, target in graph.edges():

        dot_lines.append(
            f'"{dot_escape(source)}" -> '
            f'"{dot_escape(target)}";'
        )

    dot_lines.append("}")

    st.graphviz_chart(
        "\n".join(dot_lines),
        use_container_width=True
    )

    st.markdown("**Graph legend**")

    st.write(
        "📁 Source file → 🟨 Tracked object → "
        "🟢 Observation → 🩷 Transcript / Evidence"
    )

    st.caption(
        "The graph displays recorded relationships. A connection "
        "does not independently establish that two observations "
        "refer to the same real-world event or object."
    )

else:

    st.info("Upload and analyse evidence to generate the graph.")


# =========================================================
# 21. COMPREHENSIVE PDF INVESTIGATION REPORT
# =========================================================

st.divider()

st.header("📄 Comprehensive Investigation Report")

st.write(
    "Generate a PDF containing available CCTV observations, "
    "photo detections, audio transcripts, translations, tracking "
    "summaries and investigator review decisions."
)

all_df = observation_dataframe()

if not all_df.empty:

    st.write(
        f"**Observations available for report:** {len(all_df)}"
    )

    case_title = st.text_input(
        "Investigation / Case Title",
        value="AI-Assisted Evidence Analysis"
    )

    case_reference = st.text_input(
        "Case Reference",
        value="CASE-001"
    )

    investigator = st.text_input(
        "Investigator / Prepared By",
        value=""
    )

    if st.button(
        "📄 Generate Comprehensive PDF Report",
        type="primary"
    ):

        try:

            report_path = (
                REPORT_DIR
                / f"investigation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

            styles = getSampleStyleSheet()

            styles.add(
                ParagraphStyle(
                    name="ReportTitle",
                    parent=styles["Title"],
                    alignment=TA_CENTER,
                    fontSize=19,
                    leading=24,
                    spaceAfter=12
                )
            )

            styles.add(
                ParagraphStyle(
                    name="SmallText",
                    parent=styles["BodyText"],
                    fontSize=7,
                    leading=9
                )
            )

            styles.add(
                ParagraphStyle(
                    name="SectionHeading",
                    parent=styles["Heading2"],
                    fontSize=13,
                    leading=16,
                    spaceBefore=12,
                    spaceAfter=7
                )
            )

            document = SimpleDocTemplate(
                str(report_path),
                pagesize=landscape(A4),
                rightMargin=12 * mm,
                leftMargin=12 * mm,
                topMargin=13 * mm,
                bottomMargin=13 * mm
            )

            story = []

            def add_paragraph(text, style="BodyText"):
                story.append(
                    Paragraph(
                        str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                        styles[style]
                    )
                )

            def add_table(data, widths=None, font_size=7):

                table = LongTable(
                    data,
                    colWidths=widths,
                    repeatRows=1,
                    hAlign="LEFT"
                )

                table.setStyle(
                    TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263746")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), font_size),
                        ("LEADING", (0, 0), (-1, -1), font_size + 2),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                            colors.white,
                            colors.HexColor("#F1F4F6")
                        ]),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4)
                    ])
                )

                story.append(table)
                story.append(Spacer(1, 8))

            story.append(
                Paragraph(
                    "AI CRIME SCENE & CCTV INVESTIGATION ASSISTANT",
                    styles["ReportTitle"]
                )
            )

            story.append(
                Paragraph(
                    "Multimodal AI for Evidence Linking and Incident Reconstruction",
                    styles["Heading3"]
                )
            )

            story.append(Spacer(1, 10))

            metadata = [
                ["Case Title", case_title],
                ["Case Reference", case_reference],
                ["Prepared By", investigator or "Not specified"],
                ["Report Generated", current_time()],
                ["Total Observations", str(len(all_df))],
                ["CCTV Videos", str(len(st.session_state.video_records))],
                ["Photographs", str(len(st.session_state.photo_records))],
                ["Audio Files", str(len(st.session_state.audio_records))]
            ]

            add_table(
                [["Report Information", "Details"]] + metadata,
                widths=[55 * mm, 190 * mm],
                font_size=8
            )

            story.append(
                Paragraph(
                    "1. Executive Summary",
                    styles["SectionHeading"]
                )
            )

            modality_counts = (
                all_df["Modality"]
                .value_counts()
                .to_dict()
            )

            review_counts = (
                all_df["Review Status"]
                .value_counts()
                .to_dict()
            )

            add_paragraph(
                f"This report compiles {len(all_df)} AI-generated "
                f"observations from the available evidence. "
                f"CCTV observations: {modality_counts.get('CCTV', 0)}; "
                f"photo observations: {modality_counts.get('Photo', 0)}; "
                f"audio transcript segments: {modality_counts.get('Audio', 0)}."
            )

            add_paragraph(
                f"Investigator review status: "
                f"{review_counts.get('Approved', 0)} approved, "
                f"{review_counts.get('Pending', 0)} pending, "
                f"{review_counts.get('Rejected', 0)} rejected, "
                f"{review_counts.get('Flagged', 0)} flagged."
            )

            story.append(
                Paragraph(
                    "2. Evidence Sources",
                    styles["SectionHeading"]
                )
            )

            source_rows = [
                ["Evidence Type", "File Name", "Source ID"]
            ]

            for record in st.session_state.video_records.values():
                source_rows.append([
                    "CCTV Video",
                    record["name"],
                    record["id"]
                ])

            for record in st.session_state.photo_records.values():
                source_rows.append([
                    "Photograph",
                    record["name"],
                    record["id"]
                ])

            for record in st.session_state.audio_records.values():
                source_rows.append([
                    "Audio",
                    record["name"],
                    record["id"]
                ])

            add_table(
                source_rows,
                widths=[45 * mm, 150 * mm, 45 * mm]
            )

            story.append(
                Paragraph(
                    "3. Object Detection & Tracking Observations",
                    styles["SectionHeading"]
                )
            )

            cctv_photo_df = all_df[
                all_df["Modality"].isin(["CCTV", "Photo"])
            ]

            detection_table = [[
                "Type",
                "Source",
                "Timestamp",
                "Object",
                "Track ID",
                "Confidence",
                "Review"
            ]]

            for _, row in cctv_photo_df.iterrows():

                confidence_value = row.get("Confidence")

                confidence_text = (
                    f"{float(confidence_value):.3f}"
                    if pd.notna(confidence_value)
                    else "N/A"
                )

                detection_table.append([
                    str(row["Modality"]),
                    str(row["Source"])[:45],
                    str(row["Timestamp"]),
                    str(row["Object"]),
                    str(row["Track ID"]),
                    confidence_text,
                    str(row["Review Status"])
                ])

            if len(detection_table) == 1:
                detection_table.append(
                    ["No detection observations", "", "", "", "", "", ""]
                )

            add_table(
                detection_table,
                widths=[
                    23 * mm,
                    52 * mm,
                    28 * mm,
                    30 * mm,
                    48 * mm,
                    22 * mm,
                    25 * mm
                ]
            )

            story.append(
                Paragraph(
                    "4. Audio Transcripts & English Translations",
                    styles["SectionHeading"]
                )
            )

            transcript_table = [[
                "Source",
                "Timestamp",
                "Detected Language",
                "Original Transcript",
                "English Translation",
                "Review"
            ]]

            for _, row in all_df[
                all_df["Modality"] == "Audio"
            ].iterrows():

                transcript_table.append([
                    str(row["Source"])[:35],
                    str(row["Timestamp"]),
                    str(row.get("Detected Language", "Unknown")),
                    str(row.get("Original Transcript", "")),
                    str(row.get("English Translation", "")),
                    str(row["Review Status"])
                ])

            if len(transcript_table) == 1:
                transcript_table.append(
                    ["No audio transcript segments", "", "", "", "", ""]
                )

            add_table(
                transcript_table,
                widths=[
                    42 * mm,
                    33 * mm,
                    28 * mm,
                    65 * mm,
                    65 * mm,
                    20 * mm
                ],
                font_size=6
            )

            story.append(
                Paragraph(
                    "5. Investigator Review Decisions",
                    styles["SectionHeading"]
                )
            )

            review_table = [[
                "Observation ID",
                "Type",
                "Source",
                "Observation",
                "Decision",
                "Notes",
                "Reviewed At"
            ]]

            for _, row in all_df.iterrows():

                review_table.append([
                    str(row["Observation ID"]),
                    str(row["Modality"]),
                    str(row["Source"])[:35],
                    str(row["Observation"])[:100],
                    str(row["Review Status"]),
                    str(row.get("Review Notes", ""))[:80],
                    str(row.get("Reviewed At", ""))
                ])

            add_table(
                review_table,
                widths=[
                    27 * mm,
                    20 * mm,
                    43 * mm,
                    55 * mm,
                    24 * mm,
                    55 * mm,
                    30 * mm
                ],
                font_size=6
            )

            story.append(
                Paragraph(
                    "6. Object Tracking Summary",
                    styles["SectionHeading"]
                )
            )

            if not detection_df.empty:

                tracked = detection_df[
                    detection_df["Track ID"] != "Untracked"
                ].copy()

                if not tracked.empty:

                    summary = (
                        tracked.groupby(
                            ["Source", "Track ID", "Object"],
                            as_index=False
                        )
                        .agg(
                            first_seen=("Timestamp (seconds)", "min"),
                            last_seen=("Timestamp (seconds)", "max"),
                            observations=("Timestamp", "count"),
                            average_confidence=("Confidence", "mean")
                        )
                    )

                    tracking_table = [[
                        "Source",
                        "Track ID",
                        "Object",
                        "First Seen",
                        "Last Seen",
                        "Observations",
                        "Avg. Confidence"
                    ]]

                    for _, row in summary.iterrows():

                        tracking_table.append([
                            str(row["Source"])[:35],
                            str(row["Track ID"]),
                            str(row["Object"]),
                            format_timestamp(row["first_seen"]),
                            format_timestamp(row["last_seen"]),
                            str(row["observations"]),
                            f"{row['average_confidence']:.3f}"
                        ])

                    add_table(
                        tracking_table,
                        widths=[
                            50 * mm,
                            55 * mm,
                            30 * mm,
                            30 * mm,
                            30 * mm,
                            25 * mm,
                            30 * mm
                        ]
                    )

                else:

                    add_paragraph("No tracked CCTV objects available.")

            else:

                add_paragraph("No CCTV tracking summary available.")

            story.append(
                Paragraph(
                    "7. Evidence Graph Description",
                    styles["SectionHeading"]
                )
            )

            add_paragraph(
                "The application generates a relationship graph connecting "
                "evidence source files with observations, tracked objects, "
                "transcript segments and evidence images where available. "
                "The graph represents recorded data relationships and does "
                "not independently prove that separate observations refer "
                "to the same real-world event."
            )

            story.append(
                Paragraph(
                    "8. Methodology & Limitations",
                    styles["SectionHeading"]
                )
            )

            add_paragraph(
                "YOLOv8 Nano is used for object detection. ByteTrack is used "
                "for CCTV object tracking. Faster-Whisper generates speech "
                "transcriptions and English translations. Photo analysis "
                "uses object detection without temporal tracking. Photo "
                "timestamps are not inferred by this application. Audio "
                "timestamps are relative to the beginning of the audio file."
            )

            add_paragraph(
                "Detection confidence is a model score, not proof of identity "
                "or certainty. Tracking IDs may fragment or be inconsistent. "
                "Audio translation may omit nuance or contain errors. "
                "Human review is required before treating AI-generated "
                "observations as verified findings."
            )

            add_paragraph(
                "This report is an AI-assisted evidence organisation "
                "document, not a final forensic conclusion. Original source "
                "files should be preserved and independently examined."
            )

            story.append(Spacer(1, 12))

            add_paragraph(
                "End of Report"
            )

            document.build(story)

            st.session_state.last_report_path = str(report_path)

            st.success("PDF investigation report generated successfully.")

        except Exception as error:

            st.error(
                f"Could not generate PDF report: {error}"
            )

    report_path = st.session_state.last_report_path

    if report_path and Path(report_path).is_file():

        with open(report_path, "rb") as report_file:

            st.download_button(
                "⬇️ Download Investigation PDF Report",
                report_file.read(),
                file_name=Path(report_path).name,
                mime="application/pdf"
            )

else:

    st.info(
        "Upload and analyse at least one evidence source "
        "before generating a report."
    )


# =========================================================
# 22. FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Crime Scene & CCTV Investigation Assistant | "
    "AI-generated observations require human verification. "
    "Preserve original evidence and independently verify findings."
)
