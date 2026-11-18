"""YOLOv8 Object Detection and Tracking Engine.
Optimized for NVIDIA RTX 3050 (4 GB VRAM) on Windows 11 with CPU fallback.
Generates:
- Real-time bounding boxes and tracking IDs (ByteTrack)
- Keyframe evidence snapshots
- Full annotated MP4 video output for investigator review
- Structured observation records for SQLite persistence
"""

import hashlib
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.config import ROOT  # noqa: E402
OUTPUT_DIR = ROOT / "outputs"
EVIDENCE_FRAMES_DIR = OUTPUT_DIR / "evidence_frames"
ANNOTATED_VIDEOS_DIR = OUTPUT_DIR / "annotated_videos"
PHOTO_EVIDENCE_DIR = OUTPUT_DIR / "photo_evidence"

for d in [EVIDENCE_FRAMES_DIR, ANNOTATED_VIDEOS_DIR, PHOTO_EVIDENCE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_optimal_device() -> Tuple[Any, str]:
    """Select CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return 0, f"NVIDIA GPU ({device_name})"
    return "cpu", "CPU (Standard)"


_yolo_instance = None


def _weights_path() -> str:
    for p in (ROOT / "yolov8n.pt", ROOT / "models" / "yolov8n.pt"):
        if p.is_file():
            return str(p)
    return "yolov8n.pt"  # Ultralytics downloads it on first use


def load_model(fresh: bool = False) -> YOLO:
    """Load YOLOv8 Nano. The cached instance is shared for stateless photo detection.

    fresh=True returns a new instance. Tracking needs this: ByteTrack state lives on the
    model's predictor, so with a shared model track IDs leaked from one video (or camera)
    into the next run.
    """
    global _yolo_instance
    if fresh:
        return YOLO(_weights_path())
    if _yolo_instance is None:
        _yolo_instance = YOLO(_weights_path())
    return _yolo_instance


def class_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def format_timestamp(seconds: float) -> str:
    if seconds is None or np.isnan(seconds):
        return "00:00.00"
    minutes = int(seconds // 60)
    remaining = seconds % 60
    return f"{minutes:02d}:{remaining:05.2f}"


def run_cctv_detection_tracking(
    video_path: str,
    video_id: str,
    source_name: str,
    camera_id: str = "CAM-01",
    conf_threshold: float = 0.35,
    frame_interval_sec: float = 1.0,
    target_classes: Optional[List[str]] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
    """Process CCTV recording with YOLOv8 and ByteTrack.
    Returns:
    - detections: list of observation records
    - annotated_video_path: path to output annotated MP4 video
    - preview_images: list of {image, caption} dicts for UI display
    """
    device, device_desc = get_optimal_device()
    model = load_model(fresh=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or np.isnan(fps) or fps <= 0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_step = max(1, int(round(fps * frame_interval_sec)))

    # Output annotated video setup - H.264 Web-compatible MP4 encoder
    from services.media import media_processor
    out_video_name = f"{video_id}_annotated.mp4"
    out_video_path = ANNOTATED_VIDEOS_DIR / out_video_name
    out_writer = media_processor.WebVideoWriter(str(out_video_path), fps, width, height)

    detections: List[Dict[str, Any]] = []
    previews: List[Dict[str, Any]] = []
    frame_idx = 0

    try:
        with torch.inference_mode():
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Run tracking with ByteTrack
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device=device,
                    imgsz=640,
                    conf=conf_threshold,
                    verbose=False
                )

                result = results[0]
                annotated_frame = result.plot()
                out_writer.write(annotated_frame)

                # Process detections at specified interval
                is_sample_frame = (frame_idx % frame_step == 0)
                timestamp = frame_idx / fps

                if is_sample_frame and result.boxes is not None and len(result.boxes) > 0:
                    saved_frame_path = ""
                    frame_has_target = False

                    # Check boxes
                    for box_idx, box in enumerate(result.boxes):
                        class_id = int(box.cls[0])
                        obj_name = model.names[class_id]

                        if target_classes and obj_name not in target_classes:
                            continue

                        frame_has_target = True
                        score = float(box.conf[0])
                        coords = [round(float(v), 1) for v in box.xyxy[0].tolist()]

                        if box.id is not None and len(box.id) > 0:
                            raw_id = int(box.id[0].item())
                            raw_track_id = f"T{raw_id}"
                            track_id = f"{camera_id}-T{raw_id}-{class_slug(obj_name)}"
                        else:
                            raw_track_id = "Untracked"
                            track_id = "Untracked"

                        # Save keyframe image if not saved yet for this sample frame
                        if not saved_frame_path:
                            keyframe_file = EVIDENCE_FRAMES_DIR / f"frame_{video_id}_{frame_idx}.jpg"
                            cv2.imwrite(str(keyframe_file), annotated_frame)
                            # Unannotated copy: drawn boxes/labels must not leak into
                            # appearance analysis (colour, similarity) of the crops.
                            cv2.imwrite(str(keyframe_file.with_name(keyframe_file.stem + "_raw.jpg")), frame)
                            saved_frame_path = str(keyframe_file)

                        # Untracked boxes of the same class in one frame used to share an ID
                        # and overwrite each other; the box index keeps them distinct.
                        id_key = track_id if track_id != "Untracked" else f"untracked{box_idx}"
                        obs_id = hashlib.sha1(
                            f"{video_id}_{frame_idx}_{id_key}_{obj_name}".encode("utf-8")
                        ).hexdigest()[:12]

                        row = {
                            "Observation ID": obs_id,
                            "Modality": "CCTV",
                            "Source": source_name,
                            "Source ID": video_id,
                            "Camera ID": camera_id,
                            "Timestamp": format_timestamp(timestamp),
                            "Timestamp (seconds)": round(timestamp, 2),
                            "Segment End (seconds)": None,
                            "Object": obj_name,
                            "Raw Track ID": raw_track_id,
                            "Track ID": track_id,
                            "Confidence": round(score, 3),
                            "Bounding Box": str(coords),
                            "Observation": f"{obj_name} detected in CCTV footage ({camera_id}) with {score:.1%} confidence.",
                            "Original Transcript": "",
                            "English Translation": "",
                            "Detected Language": "",
                            "Language Probability": None,
                            "Evidence Path": saved_frame_path,
                            "Review Status": "Pending",
                            "Review Notes": "",
                            "Reviewed At": ""
                        }
                        detections.append(row)

                    if frame_has_target and len(previews) < 6:
                        rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        previews.append({
                            "image": rgb,
                            "caption": f"{camera_id} | {source_name} | {format_timestamp(timestamp)}"
                        })

                frame_idx += 1

                # Periodically update progress and clear CUDA cache to protect 4GB VRAM
                if frame_idx % 30 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if progress_callback and total_frames > 0:
                        pct = min(frame_idx / total_frames, 1.0)
                        progress_callback(pct, f"Processing frame {frame_idx}/{total_frames} ({pct:.0%})")

    finally:
        cap.release()
        out_writer.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if progress_callback:
        progress_callback(1.0, f"Analysis complete: {frame_idx} frames processed.")

    return detections, str(out_video_path), previews


def run_photo_detection(
    image_path: str,
    photo_id: str,
    source_name: str,
    camera_id: str = "SCENE-PHOTO",
    conf_threshold: float = 0.35,
    target_classes: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], str, np.ndarray]:
    """Detect objects in a still crime scene photograph."""
    device, _ = get_optimal_device()
    model = load_model()

    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image file: {image_path}")

    with torch.inference_mode():
        results = model.predict(
            source=image,
            device=device,
            imgsz=640,
            conf=conf_threshold,
            verbose=False
        )

    result = results[0]
    annotated = result.plot()

    out_annotated_path = PHOTO_EVIDENCE_DIR / f"annotated_{photo_id}.jpg"
    cv2.imwrite(str(out_annotated_path), annotated)

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        for idx, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            obj_name = model.names[class_id]

            if target_classes and obj_name not in target_classes:
                continue

            score = float(box.conf[0])
            coords = [round(float(v), 1) for v in box.xyxy[0].tolist()]

            obs_id = hashlib.sha1(
                f"{photo_id}_{idx}_{obj_name}".encode("utf-8")
            ).hexdigest()[:12]

            row = {
                "Observation ID": obs_id,
                "Modality": "Photo",
                "Source": source_name,
                "Source ID": photo_id,
                "Camera ID": camera_id,
                "Timestamp": "Scene Snapshot",
                "Timestamp (seconds)": None,
                "Segment End (seconds)": None,
                "Object": obj_name,
                "Raw Track ID": "Not applicable",
                "Track ID": "Not applicable",
                "Confidence": round(score, 3),
                "Bounding Box": str(coords),
                "Observation": f"{obj_name} identified in scene photograph with {score:.1%} confidence.",
                "Original Transcript": "",
                "English Translation": "",
                "Detected Language": "",
                "Language Probability": None,
                "Evidence Path": str(out_annotated_path),
                "Review Status": "Pending",
                "Review Notes": "",
                "Reviewed At": ""
            }
            detections.append(row)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return detections, str(out_annotated_path), rgb

