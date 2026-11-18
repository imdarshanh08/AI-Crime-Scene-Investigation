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
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from core.config import CROSS_CAMERA_DB_PATH, MASTER_DB_PATH, REVIEW_STATUSES
from storage.sqlite_utils import connect as sqlite_connect


def _now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def get_db_connection():
    """Context-managed connection to the master case database (commits on success, always closes)."""
    return sqlite_connect(MASTER_DB_PATH)


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

            CREATE INDEX IF NOT EXISTS idx_obs_source ON observations(source_id);
            CREATE INDEX IF NOT EXISTS idx_obs_modality ON observations(modality);
            CREATE INDEX IF NOT EXISTS idx_obs_time ON observations(timestamp_seconds);
            CREATE INDEX IF NOT EXISTS idx_sources_modality ON evidence_sources(modality);

            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );
        """)
    _run_migrations()


# -------------------------------------------------------------
# SCHEMA MIGRATIONS (additive only: no column or row is ever dropped)
# -------------------------------------------------------------

def _column_names(conn: sqlite3.Connection, table: str) -> set:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _migration_2_source_metadata(conn: sqlite3.Connection) -> None:
    """Add recording-clock, location and processing-status metadata to sources.

    recorded_start replaces the camera "time anchors" that previously lived in the
    separate cross-camera database, so every module shares one timebase. Existing
    anchors are copied across; the old table is left untouched.
    """
    existing = _column_names(conn, "evidence_sources")
    additions = {
        "recorded_start": "TEXT DEFAULT ''",          # ISO local time the recording began (investigator-verified)
        "location": "TEXT DEFAULT ''",                # free-text site/area label, e.g. "North gate"
        "processing_status": "TEXT DEFAULT 'Registered'",
        "processed_at": "TEXT DEFAULT ''",
    }
    for column, ddl in additions.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE evidence_sources ADD COLUMN {column} {ddl}")

    # Sources that already have observations have been processed.
    conn.execute("""
        UPDATE evidence_sources SET processing_status = 'Processed'
        WHERE (processing_status IS NULL OR processing_status = 'Registered')
          AND source_id IN (SELECT DISTINCT source_id FROM observations)
    """)

    if CROSS_CAMERA_DB_PATH.is_file():
        legacy = sqlite3.connect(str(CROSS_CAMERA_DB_PATH))
        try:
            has_table = legacy.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='camera_time_anchors'"
            ).fetchone()
            rows = legacy.execute(
                "SELECT source_id, recording_start FROM camera_time_anchors"
            ).fetchall() if has_table else []
        finally:
            legacy.close()
        for source_id, recording_start in rows:
            conn.execute(
                "UPDATE evidence_sources SET recorded_start = ? WHERE source_id = ? AND COALESCE(recorded_start, '') = ''",
                (str(recording_start), str(source_id)),
            )


def _migration_3_relationship_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_rel_source ON graph_relationships(source_observation);
        CREATE INDEX IF NOT EXISTS idx_rel_target ON graph_relationships(target_observation);
        CREATE INDEX IF NOT EXISTS idx_obs_status ON observations(review_status);
    """)


MIGRATIONS = [
    (2, "Source metadata: recorded_start, location, processing status; import camera anchors", _migration_2_source_metadata),
    (3, "Relationship and review-status indexes", _migration_3_relationship_indexes),
]


def _run_migrations() -> None:
    with get_db_connection() as conn:
        if not conn.execute("SELECT 1 FROM schema_version WHERE version = 1").fetchone():
            conn.execute(
                "INSERT INTO schema_version (version, description, applied_at) VALUES (1, 'Baseline schema', ?)",
                (_now(),),
            )
        applied = {r["version"] for r in conn.execute("SELECT version FROM schema_version").fetchall()}
        for version, description, migrate in MIGRATIONS:
            if version in applied:
                continue
            migrate(conn)
            conn.execute(
                "INSERT INTO schema_version (version, description, applied_at) VALUES (?, ?, ?)",
                (version, description, _now()),
            )


def get_schema_version() -> int:
    with get_db_connection() as conn:
        row = conn.execute("SELECT MAX(version) AS v FROM schema_version").fetchone()
    return int(row["v"] or 0)


# -------------------------------------------------------------
# SOURCE MANAGEMENT
# -------------------------------------------------------------

def update_source_metadata(
    source_id: str,
    location: Optional[str] = None,
    recorded_start: Optional[str] = None,
) -> None:
    """Set investigator-supplied location and/or recording start (ISO 'YYYY-MM-DD HH:MM:SS').

    Pass an empty string to clear a value; None leaves it unchanged. Logged to the audit trail.
    """
    changes = []
    with get_db_connection() as conn:
        old = conn.execute(
            "SELECT location, recorded_start FROM evidence_sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        if old is None:
            raise ValueError(f"Unknown source {source_id}")
        if location is not None and location.strip() != (old["location"] or ""):
            conn.execute("UPDATE evidence_sources SET location = ? WHERE source_id = ?", (location.strip(), source_id))
            changes.append(f"location '{old['location'] or ''}' -> '{location.strip()}'")
        if recorded_start is not None:
            value = recorded_start.strip()
            if value:
                value = datetime.fromisoformat(value).isoformat(sep=" ", timespec="seconds")
            if value != (old["recorded_start"] or ""):
                conn.execute("UPDATE evidence_sources SET recorded_start = ? WHERE source_id = ?", (value, source_id))
                changes.append(f"recorded_start '{old['recorded_start'] or ''}' -> '{value}'")
        if changes:
            conn.execute("""
                INSERT INTO audit_logs (action, entity_type, entity_id, details, timestamp)
                VALUES ('Source metadata updated', 'EvidenceSource', ?, ?, ?)
            """, (source_id, "; ".join(changes), _now()))


def set_processing_status(source_id: str, status: str) -> None:
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE evidence_sources SET processing_status = ?, processed_at = ? WHERE source_id = ?",
            (status, _now() if status == "Processed" else "", source_id),
        )


def data_version() -> str:
    """Cheap fingerprint of case data; changes whenever evidence, reviews or links change.

    Used as a cache key so expensive derived structures (the evidence graph) are rebuilt
    only when the underlying data actually changed, not on every Streamlit rerun.
    """
    with get_db_connection() as conn:
        parts = [
            conn.execute("SELECT COUNT(*), COALESCE(MAX(upload_time),''), "
                         "COALESCE(GROUP_CONCAT(camera_id || location || recorded_start, '|'),'') FROM evidence_sources").fetchone(),
            conn.execute("SELECT COUNT(*), COALESCE(MAX(created_at),''), COALESCE(MAX(reviewed_at),'') FROM observations").fetchone(),
            conn.execute("SELECT COUNT(*), COALESCE(MAX(created_at),''), COALESCE(MAX(reviewed_at),'') FROM graph_relationships").fetchone(),
            conn.execute("SELECT COALESCE(MAX(log_id), 0) FROM audit_logs").fetchone(),
        ]
    return hashlib.sha1(repr([tuple(p) for p in parts]).encode()).hexdigest()[:16]

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


def next_camera_id(prefix: str = "CAM") -> str:
    """Return the next free camera ID (e.g. CAM-03), based on IDs already in the database.

    Counting session records was unreliable: several files uploaded together all got
    the same ID, and IDs were reused after a deletion.
    """
    with get_db_connection() as conn:
        rows = conn.execute("SELECT camera_id FROM evidence_sources").fetchall()
    used = set()
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    for r in rows:
        m = pattern.match(str(r["camera_id"] or ""))
        if m:
            used.add(int(m.group(1)))
    n = 1
    while n in used:
        n += 1
    return f"{prefix}-{n:02d}"


def update_source_camera_id(source_id: str, camera_id: str) -> None:
    camera_id = camera_id.strip()
    if not camera_id:
        raise ValueError("Camera ID cannot be empty.")
    with get_db_connection() as conn:
        old = conn.execute(
            "SELECT camera_id FROM evidence_sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        conn.execute(
            "UPDATE evidence_sources SET camera_id = ? WHERE source_id = ?",
            (camera_id, source_id)
        )
        conn.execute(
            "UPDATE observations SET camera_id = ? WHERE source_id = ?",
            (camera_id, source_id)
        )
        conn.execute("""
            INSERT INTO audit_logs (action, entity_type, entity_id, details, timestamp)
            VALUES ('Camera ID changed', 'EvidenceSource', ?, ?, ?)
        """, (source_id, f"{old['camera_id'] if old else '?'} -> {camera_id}", _now()))


def delete_source_record(source_id: str) -> None:
    """Remove a source and its observations from the case. The file on disk is kept."""
    with get_db_connection() as conn:
        src = conn.execute(
            "SELECT source_name, sha256_hash FROM evidence_sources WHERE source_id = ?", (source_id,)
        ).fetchone()
        obs_ids = [r["observation_id"] for r in conn.execute(
            "SELECT observation_id FROM observations WHERE source_id = ?", (source_id,)
        ).fetchall()]
        if obs_ids:
            marks = ",".join("?" * len(obs_ids))
            conn.execute(
                f"DELETE FROM graph_relationships WHERE source_observation IN ({marks}) OR target_observation IN ({marks})",
                obs_ids + obs_ids,
            )
            conn.execute(f"DELETE FROM observation_reviews WHERE observation_id IN ({marks})", obs_ids)
        conn.execute("DELETE FROM observations WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM evidence_sources WHERE source_id = ?", (source_id,))
        details = (
            f"{src['source_name']} (SHA-256 {src['sha256_hash'][:16]}...), "
            f"{len(obs_ids)} observation(s) removed; file retained on disk"
            if src else "Source not found"
        )
        conn.execute("""
            INSERT INTO audit_logs (action, entity_type, entity_id, details, timestamp)
            VALUES ('Evidence source removed', 'EvidenceSource', ?, ?, ?)
        """, (source_id, details, _now()))


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
    if new_status not in REVIEW_STATUSES:
        raise ValueError(f"Invalid review status: {new_status}")
    reviewer_notes = reviewer_notes or ""
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



def clear_all_observations() -> int:
    """Delete every observation (and links between them). Sources stay registered."""
    with get_db_connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        conn.execute("DELETE FROM observations")
        conn.execute("DELETE FROM observation_reviews")
        conn.execute("DELETE FROM graph_relationships")
        conn.execute("""
            INSERT INTO audit_logs (action, entity_type, entity_id, details, timestamp)
            VALUES ('All observations cleared', 'Observation', 'ALL', ?, ?)
        """, (f"{count} observation(s) deleted by investigator", _now()))
    return count


def log_investigator_action(action: str, entity_type: str, entity_id: str, details: str = "") -> None:
    now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO audit_logs (action, entity_type, entity_id, details, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (action, entity_type, entity_id, details, now_str))


def get_audit_logs(limit: Optional[int] = 500) -> pd.DataFrame:
    """Most recent audit entries first. Pass limit=None for the complete trail."""
    query = "SELECT * FROM audit_logs ORDER BY log_id DESC"
    params: tuple = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (int(limit),)
    with get_db_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


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

