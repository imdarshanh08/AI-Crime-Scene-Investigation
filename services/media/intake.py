"""Evidence intake: store an uploaded file, hash it, validate it and register it.

The original upload is written once to uploads/ under its SHA-256 prefix and never
modified afterwards; all derived artefacts (frames, annotated video, WAV) go to
outputs/. Identical files (same SHA-256) are recognised and not registered twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.config import UPLOAD_DIR, get_logger
from services.media import media_processor
from storage import db_manager

log = get_logger("intake")

ID_PREFIX = {"CCTV": "CAM", "Photo": "PHOTO", "Audio": "AUDIO"}


@dataclass
class IntakeResult:
    status: str                 # "registered" | "duplicate" | "invalid" | "error"
    message: str
    source: Optional[Dict[str, Any]] = None


def register_upload(uploaded_file, modality: str) -> IntakeResult:
    """Persist and register one Streamlit UploadedFile as evidence of the given modality."""
    name = getattr(uploaded_file, "name", "upload")
    try:
        info = media_processor.save_uploaded_stream(uploaded_file, UPLOAD_DIR)
    except OSError as exc:
        log.error("could not store %s: %s", name, exc)
        return IntakeResult("error", f"{name}: could not be stored ({exc}).")

    existing = db_manager.get_source_by_id(info["id"])
    if existing:
        return IntakeResult("duplicate", f"{name} is already registered as {existing['camera_id']} "
                                         "(identical SHA-256).", existing)

    meta: Dict[str, Any] = {}
    if modality == "CCTV":
        meta = media_processor.get_video_metadata(info["path"])
        if not meta["is_valid"]:
            return IntakeResult("invalid", f"{name}: not a readable video ({meta['error']}).")
        meta = {k: meta[k] for k in ("duration_seconds", "fps", "total_frames", "width", "height")}

    device_id = db_manager.next_camera_id(ID_PREFIX.get(modality, "SRC"))
    db_manager.save_source_record(
        source_id=info["id"], source_name=name, file_path=info["path"], modality=modality,
        camera_id=device_id, file_size_bytes=info["size_bytes"], sha256_hash=info["sha256"], **meta,
    )
    db_manager.log_investigator_action(
        action=f"Registered {modality} evidence", entity_type=modality, entity_id=info["id"],
        details=f"File: {name}, ID: {device_id}, SHA-256: {info['sha256']}",
    )
    log.info("registered %s %s as %s", modality, name, device_id)
    return IntakeResult("registered", f"{name} registered as {device_id}.", db_manager.get_source_by_id(info["id"]))
