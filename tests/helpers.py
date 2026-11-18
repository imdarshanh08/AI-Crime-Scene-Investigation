"""Test fixtures: an isolated temporary case database and synthetic graph inputs.

Synthetic records exist only inside tests; the application never generates sample data.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from storage import db_manager  # noqa: E402


class TempDatabaseCase(unittest.TestCase):
    """Points the master DB at a fresh temporary file for each test."""

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="crime_ai_test_")
        self._orig = db_manager.MASTER_DB_PATH
        db_manager.MASTER_DB_PATH = Path(self._tmp) / "case.db"
        db_manager.init_database()

    def tearDown(self):
        db_manager.MASTER_DB_PATH = self._orig
        shutil.rmtree(self._tmp, ignore_errors=True)


def obs(oid, source_id, camera, t, obj="person", track="", modality="CCTV", conf=0.8, status="Pending",
        transcript="", end=None):
    return {
        "Observation ID": oid, "Modality": modality, "Source": f"{source_id}.mp4", "Source ID": source_id,
        "Camera ID": camera, "Timestamp": f"{t:.2f}", "Timestamp (seconds)": t, "Segment End (seconds)": end,
        "Object": obj, "Raw Track ID": track, "Track ID": track or ("Not applicable" if modality != "CCTV" else "Untracked"),
        "Confidence": conf, "Bounding Box": "", "Observation": f"{obj} at {t}", "Original Transcript": transcript,
        "English Translation": transcript, "Detected Language": "en" if transcript else "",
        "Language Probability": None, "Evidence Path": "", "Review Status": status, "Review Notes": "",
        "Reviewed At": "", "Created At": "",
    }


def source(sid, camera, modality="CCTV", start="", location=""):
    return {"source_id": sid, "source_name": f"{sid}.mp4", "camera_id": camera, "modality": modality,
            "recorded_start": start, "location": location, "file_path": "", "sha256_hash": sid * 4,
            "duration_seconds": 120.0, "upload_time": "2026-11-18 10:00:00"}


def frame(rows):
    return pd.DataFrame(rows)
