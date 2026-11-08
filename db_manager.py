"""Unified Database and Evidence Persistence Manager for AI Crime Scene Investigation Assistant.
Handles persistent storage of:
- Uploaded videos, photo evidence, and audio files
- Camera assignments and metadata
- YOLO object detections and ByteTrack tracking records
- Audio transcripts and translations (Faster-Whisper)
- Reconstructed events and cross-camera candidate links
- Multimodal graph relationships
- Investigator review actions (Human-in-the-loop audit trail)
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATABASE_DIR = ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
MASTER_DB_PATH = DATABASE_DIR / "investigation_master.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(MASTER_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database() -> None:
    """Initialize all master database tables."""
    with get_db_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS evidence_sources (
                source_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                modality TEXT NOT NULL,          -- 'CCTV', 'Photo', 'Audio'
                camera_id TEXT DEFAULT 'CAM-01', -- Assigned camera identifier
                file_size_bytes INTEGER DEFAULT 0,
                sha256_hash TEXT NOT NULL,
                duration_seconds REAL DEFAULT 0,
                fps REAL DEFAULT 0,
                total_frames INTEGER DEFAULT 0,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                upload_time TEXT NOT NULL,
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                modality TEXT NOT NULL,          -- 'CCTV', 'Photo', 'Audio'
                source_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                camera_id TEXT DEFAULT 'CAM-01',
                timestamp_str TEXT NOT NULL,
                timestamp_seconds REAL DEFAULT 0,
                segment_end_seconds REAL,
                object_class TEXT NOT NULL,
                raw_track_id TEXT DEFAULT 'Untracked',
                track_id TEXT DEFAULT 'Untracked',
                confidence REAL,
                bounding_box TEXT DEFAULT '',
                observation_text TEXT NOT NULL,
                original_transcript TEXT DEFAULT '',
                english_translation TEXT DEFAULT '',
                detected_language TEXT DEFAULT '',
                language_probability REAL,
                evidence_path TEXT DEFAULT '',
                review_status TEXT DEFAULT 'Pending', -- 'Pending', 'Approved', 'Rejected', 'Flagged'
                review_notes TEXT DEFAULT '',
                reviewed_at TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES evidence_sources(source_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS observation_reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                observation_id TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                reviewer_notes TEXT,
                reviewed_at TEXT NOT NULL,
                FOREIGN KEY (observation_id) REFERENCES observations(observation_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS graph_relationships (
                relationship_id TEXT PRIMARY KEY,
                source_observation TEXT NOT NULL,
                target_observation TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                explanation TEXT,
                confidence REAL DEFAULT 0.0,
                status TEXT DEFAULT 'Pending',
                origin TEXT DEFAULT 'AI Proposed',
                reviewer_notes TEXT DEFAULT '',
                reviewed_at TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                details TEXT,
                timestamp TEXT NOT NULL
            );
        """)
        conn.commit()


# -------------------------------------------------------------
# SOURCE MANAGEMENT
# -------------------------------------------------------------

def save_source_record(
    source_id: str,
    source_name: str,
    file_path: str,
    modality: str,
    camera_id: str = "CAM-01",
    file_size_bytes: int = 0,
    sha256_hash: str = "",
    duration_seconds: float = 0.0,
    fps: float = 0.0,
    total_frames: int = 0,
    width: int = 0,
    height: int = 0,
    notes: str = ""
) -> None:
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO evidence_sources (
                source_id, source_name, file_path, modality, camera_id,
                file_size_bytes, sha256_hash, duration_seconds, fps,
                total_frames, width, height, upload_time, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                source_name = excluded.source_name,
                file_path = excluded.file_path,
                camera_id = CASE WHEN excluded.camera_id != 'CAM-01' THEN excluded.camera_id ELSE evidence_sources.camera_id END,
                duration_seconds = excluded.duration_seconds,
                fps = excluded.fps,
                total_frames = excluded.total_frames,
                width = excluded.width,
                height = excluded.height
        """, (
            source_id, source_name, str(file_path), modality, camera_id,
            file_size_bytes, sha256_hash, duration_seconds, fps,
            total_frames, width, height, now_str, notes
        ))
        conn.commit()


def get_all_sources(modality: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        if modality:
            rows = conn.execute(
                "SELECT * FROM evidence_sources WHERE modality = ? ORDER BY upload_time DESC",
                (modality,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM evidence_sources ORDER BY upload_time DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def get_source_by_id(source_id: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM evidence_sources WHERE source_id = ?",
            (source_id,)
        ).fetchone()
        return dict(row) if row else None


def update_source_camera_id(source_id: str, camera_id: str) -> None:
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE evidence_sources SET camera_id = ? WHERE source_id = ?",
            (camera_id, source_id)
        )
        conn.execute(
            "UPDATE observations SET camera_id = ? WHERE source_id = ?",
            (camera_id, source_id)
        )
        conn.commit()


def delete_source_record(source_id: str) -> None:
    with get_db_connection() as conn:
        conn.execute("DELETE FROM evidence_sources WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM observations WHERE source_id = ?", (source_id,))
        conn.commit()


# -------------------------------------------------------------
# OBSERVATION MANAGEMENT
# -------------------------------------------------------------

def save_observation(obs_dict: Dict[str, Any]) -> None:
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO observations (
                observation_id, modality, source_id, source_name, camera_id,
                timestamp_str, timestamp_seconds, segment_end_seconds, object_class,
                raw_track_id, track_id, confidence, bounding_box, observation_text,
                original_transcript, english_translation, detected_language,
                language_probability, evidence_path, review_status, review_notes,
                reviewed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(observation_id) DO UPDATE SET
                confidence = excluded.confidence,
                bounding_box = excluded.bounding_box,
                evidence_path = excluded.evidence_path
        """, (
            obs_dict.get("Observation ID"),
            obs_dict.get("Modality", "CCTV"),
            obs_dict.get("Source ID"),
            obs_dict.get("Source", "Unknown"),
            obs_dict.get("Camera ID", "CAM-01"),
            obs_dict.get("Timestamp", "00:00.00"),
            obs_dict.get("Timestamp (seconds)", 0.0),
            obs_dict.get("Segment End (seconds)"),
            obs_dict.get("Object", "Unknown"),
            obs_dict.get("Raw Track ID", "Untracked"),
            obs_dict.get("Track ID", "Untracked"),
            obs_dict.get("Confidence"),
            obs_dict.get("Bounding Box", ""),
            obs_dict.get("Observation", ""),
            obs_dict.get("Original Transcript", ""),
            obs_dict.get("English Translation", ""),
            obs_dict.get("Detected Language", ""),
            obs_dict.get("Language Probability"),
            obs_dict.get("Evidence Path", ""),
            obs_dict.get("Review Status", "Pending"),
            obs_dict.get("Review Notes", ""),
            obs_dict.get("Reviewed At", ""),
            now_str
        ))
        conn.commit()


def save_observations_batch(observations: List[Dict[str, Any]]) -> None:
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with get_db_connection() as conn:
        for obs in observations:
            conn.execute("""
                INSERT INTO observations (
                    observation_id, modality, source_id, source_name, camera_id,
                    timestamp_str, timestamp_seconds, segment_end_seconds, object_class,
                    raw_track_id, track_id, confidence, bounding_box, observation_text,
                    original_transcript, english_translation, detected_language,
                    language_probability, evidence_path, review_status, review_notes,
                    reviewed_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(observation_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    bounding_box = excluded.bounding_box,
                    evidence_path = excluded.evidence_path
            """, (
                obs.get("Observation ID"),
                obs.get("Modality", "CCTV"),
                obs.get("Source ID"),
                obs.get("Source", "Unknown"),
                obs.get("Camera ID", "CAM-01"),
                obs.get("Timestamp", "00:00.00"),
                obs.get("Timestamp (seconds)", 0.0),
                obs.get("Segment End (seconds)"),
                obs.get("Object", "Unknown"),
                obs.get("Raw Track ID", "Untracked"),
                obs.get("Track ID", "Untracked"),
                obs.get("Confidence"),
                obs.get("Bounding Box", ""),
                obs.get("Observation", ""),
                obs.get("Original Transcript", ""),
                obs.get("English Translation", ""),
                obs.get("Detected Language", ""),
                obs.get("Language Probability"),
                obs.get("Evidence Path", ""),
                obs.get("Review Status", "Pending"),
                obs.get("Review Notes", ""),
                obs.get("Reviewed At", ""),
                now_str
            ))
        conn.commit()


def get_all_observations_df(modality: Optional[str] = None) -> pd.DataFrame:
    with get_db_connection() as conn:
        query = "SELECT * FROM observations"
        params = ()
        if modality:
            query += " WHERE modality = ?"
            params = (modality,)
        query += " ORDER BY timestamp_seconds ASC, observation_id ASC"
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return pd.DataFrame()

    # Map column names to UI standard
    rename_map = {
        "observation_id": "Observation ID",
        "modality": "Modality",
        "source_name": "Source",
        "source_id": "Source ID",
        "camera_id": "Camera ID",
        "timestamp_str": "Timestamp",
        "timestamp_seconds": "Timestamp (seconds)",
        "segment_end_seconds": "Segment End (seconds)",
        "object_class": "Object",
        "raw_track_id": "Raw Track ID",
        "track_id": "Track ID",
        "confidence": "Confidence",
        "bounding_box": "Bounding Box",
        "observation_text": "Observation",
        "original_transcript": "Original Transcript",
        "english_translation": "English Translation",
        "detected_language": "Detected Language",
        "language_probability": "Language Probability",
        "evidence_path": "Evidence Path",
        "review_status": "Review Status",
        "review_notes": "Review Notes",
        "reviewed_at": "Reviewed At",
        "created_at": "Created At"
    }
    return df.rename(columns=rename_map)


def update_observation_review(
    observation_id: str,
    new_status: str,
    reviewer_notes: str
) -> None:
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with get_db_connection() as conn:
        cur = conn.execute(
            "SELECT review_status FROM observations WHERE observation_id = ?",
            (observation_id,)
        )
        row = cur.fetchone()
        previous_status = row["review_status"] if row else "Pending"

        conn.execute("""
            UPDATE observations
            SET review_status = ?,
                review_notes = ?,
                reviewed_at = ?
            WHERE observation_id = ?
        """, (new_status, reviewer_notes, now_str if new_status != "Pending" else "", observation_id))

        conn.execute("""
            INSERT INTO observation_reviews (
                observation_id, previous_status, new_status, reviewer_notes, reviewed_at
            ) VALUES (?, ?, ?, ?, ?)
        """, (observation_id, previous_status, new_status, reviewer_notes, now_str))

        # Log action
        conn.execute("""
            INSERT INTO audit_logs (action, entity_type, entity_id, details, timestamp)
            VALUES (?, 'Observation', ?, ?, ?)
        """, (f"Review changed: {previous_status} -> {new_status}", observation_id, reviewer_notes, now_str))

        conn.commit()


def clear_all_observations() -> None:
    with get_db_connection() as conn:
        conn.execute("DELETE FROM observations")
        conn.execute("DELETE FROM observation_reviews")
        conn.commit()


def log_investigator_action(action: str, entity_type: str, entity_id: str, details: str = "") -> None:
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO audit_logs (action, entity_type, entity_id, details, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (action, entity_type, entity_id, details, now_str))
        conn.commit()


def get_audit_logs() -> pd.DataFrame:
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM audit_logs ORDER BY log_id DESC LIMIT 500",
            conn
        )
        return df


def record_graph_relationship(
    source_observation: str,
    target_observation: str,
    relationship_type: str,
    explanation: str = "",
    confidence: float = 1.0,
    status: str = "Pending",
    origin: str = "Investigator Linked",
    reviewer_notes: str = ""
) -> str:
    """Record an investigator or AI-proposed relationship between two evidence observations."""
    rel_id = hashlib.sha256(f"{source_observation}_{target_observation}_{relationship_type}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with get_db_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO graph_relationships (
                relationship_id, source_observation, target_observation, relationship_type,
                explanation, confidence, status, origin, reviewer_notes, reviewed_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rel_id, source_observation, target_observation, relationship_type,
            explanation, confidence, status, origin, reviewer_notes,
            now_str if status != "Pending" else "", now_str
        ))
        conn.execute("""
            INSERT INTO audit_logs (action, entity_type, entity_id, details, timestamp)
            VALUES (?, 'GraphRelationship', ?, ?, ?)
        """, (f"Relationship recorded: {relationship_type}", rel_id, f"{source_observation} -> {target_observation} ({status})", now_str))
        conn.commit()
    return rel_id


def get_graph_relationships() -> pd.DataFrame:
    """Retrieve all recorded evidence graph relationships."""
    with get_db_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM graph_relationships ORDER BY created_at DESC",
            conn
        )
        return df


# Initialize tables on import
init_database()

