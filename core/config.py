"""Central configuration: filesystem layout, processing defaults and shared constants.

Every module resolves paths from here, so moving a module between packages can never
silently relocate the evidence store or databases.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Project root (the folder that contains app.py).
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Storage layout (unchanged from earlier versions so existing cases keep working)
# ---------------------------------------------------------------------------
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
EVIDENCE_FRAMES_DIR = OUTPUT_DIR / "evidence_frames"
ANNOTATED_VIDEOS_DIR = OUTPUT_DIR / "annotated_videos"
PHOTO_EVIDENCE_DIR = OUTPUT_DIR / "photo_evidence"
AUDIO_EVIDENCE_DIR = OUTPUT_DIR / "audio_evidence"
REPORTS_DIR = OUTPUT_DIR / "reports"
GRAPHS_DIR = OUTPUT_DIR / "evidence_graphs"
ARCHIVE_DIR = ROOT / "archived_videos"
DATABASE_DIR = ROOT / "database"
LOG_DIR = OUTPUT_DIR / "logs"

MASTER_DB_PATH = DATABASE_DIR / "investigation_master.db"
EVENTS_DB_PATH = OUTPUT_DIR / "event_reconstruction.db"
CROSS_CAMERA_DB_PATH = OUTPUT_DIR / "cross_camera_links.db"
CROSS_MODAL_DB_PATH = OUTPUT_DIR / "evidence_graph_relationships.db"

ALL_DIRS = [
    UPLOAD_DIR, OUTPUT_DIR, EVIDENCE_FRAMES_DIR, ANNOTATED_VIDEOS_DIR, PHOTO_EVIDENCE_DIR,
    AUDIO_EVIDENCE_DIR, REPORTS_DIR, GRAPHS_DIR, DATABASE_DIR, LOG_DIR,
]
for _d in ALL_DIRS:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Review workflow
# ---------------------------------------------------------------------------
REVIEW_STATUSES = ["Pending", "Approved", "Rejected", "Flagged"]

# ---------------------------------------------------------------------------
# Accepted upload types
# ---------------------------------------------------------------------------
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg", "aac"]

# ---------------------------------------------------------------------------
# Hardware-aware defaults (RTX 3050 Laptop, 4 GB VRAM, 16 GB RAM)
# ---------------------------------------------------------------------------
YOLO_IMAGE_SIZE = 640
# Whisper runs on CPU int8 so the 4 GB of VRAM stays free for YOLO.
WHISPER_DEVICE = "cpu"
WHISPER_DEFAULT_MODEL = "base"

# ---------------------------------------------------------------------------
# Evidence-graph rule thresholds (seconds). Each one appears in edge reasoning.
# ---------------------------------------------------------------------------
HEARD_DURING_WINDOW_S = 5.0        # speech <-> visual observation on a shared clock
TEMPORAL_OVERLAP_SLACK_S = 2.0     # two events count as overlapping within this slack
CROSS_CAMERA_MAX_GAP_S = 120.0     # max hand-off gap for CROSSED_CAMERA candidates
SIMULTANEITY_CONTRADICTION_S = 1.0  # same entity, two locations, within this window


def get_logger(name: str) -> logging.Logger:
    """Module logger writing to outputs/logs/app.log and stderr (configured once)."""
    root_logger = logging.getLogger("crime_ai")
    if not root_logger.handlers:
        root_logger.setLevel(os.environ.get("CRIME_AI_LOG_LEVEL", "INFO"))
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
    return root_logger.getChild(name)
