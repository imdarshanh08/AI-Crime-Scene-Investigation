"""Persistent, explainable evidence graph foundation for a Streamlit investigation app.

Python 3.11+; runtime dependencies: standard library only.
NetworkX/PyVis can consume the dictionaries returned by graph_payload().

Important design boundary:
- AI observations and AI-proposed relationships remain unverified.
- Source-relative timestamps are retained. Cross-source time comparisons are
  made only when both sources have investigator-confirmed clock alignment.
- A matching object class alone is never treated as proof of object identity.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
VALID_SOURCE_TYPES = {"cctv", "photo", "audio"}
VALID_OBSERVATION_TYPES = {
    "object_detection",
    "track_observation",
    "photo_detection",
    "transcript_segment",
    "manual_note",
}
VALID_REVIEW_STATUSES = {"pending", "approved", "rejected", "flagged"}
OBJECT_OBSERVATION_TYPES = {"object_detection", "track_observation", "photo_detection"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except (TypeError, json.JSONDecodeError):
        return {"raw": value}


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def _stable_relationship_id(case_id: str, a: str, b: str, relation_type: str) -> str:
    left, right = sorted((a, b))
    raw = f"{case_id}|{left}|{right}|{relation_type}".encode("utf-8")
    return "rel_" + hashlib.sha256(raw).hexdigest()[:20]


class EvidenceStore:
    """SQLite-backed case, source, observation, relationship and event store."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
                    source_type TEXT NOT NULL CHECK(source_type IN ('cctv','photo','audio')),
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL DEFAULT '',
                    camera_label TEXT NOT NULL DEFAULT '',
                    location_label TEXT NOT NULL DEFAULT '',
                    clock_offset_seconds REAL NOT NULL DEFAULT 0,
                    clock_alignment_verified INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
                    source_id TEXT NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
                    observation_type TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    confidence REAL,
                    source_start_seconds REAL,
                    source_end_seconds REAL,
                    track_local_id TEXT NOT NULL DEFAULT '',
                    original_text TEXT NOT NULL DEFAULT '',
                    translated_text TEXT NOT NULL DEFAULT '',
                    attributes_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_observations_case ON observations(case_id);
                CREATE INDEX IF NOT EXISTS idx_observations_source ON observations(source_id);
                CREATE TABLE IF NOT EXISTS relationships (
                    relationship_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
                    source_observation_id TEXT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
                    target_observation_id TEXT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL,
                    origin TEXT NOT NULL CHECK(origin IN ('ai','investigator')),
                    status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected','flagged')),
                    rule_name TEXT NOT NULL DEFAULT '',
                    explanation TEXT NOT NULL DEFAULT '',
                    heuristic_score REAL,
                    factors_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewer TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT,
                    notes TEXT NOT NULL DEFAULT '',
                    UNIQUE(case_id, source_observation_id, target_observation_id, relation_type),
                    CHECK(source_observation_id <> target_observation_id)
                );
                CREATE INDEX IF NOT EXISTS idx_relationships_case_status
                    ON relationships(case_id, status);
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected','flagged')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewer TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT,
                    notes TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS event_members (
                    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
                    observation_id TEXT NOT NULL REFERENCES observations(observation_id) ON DELETE CASCADE,
                    added_by TEXT NOT NULL DEFAULT 'system',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(event_id, observation_id)
                );
                CREATE TABLE IF NOT EXISTS review_actions (
                    action_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
                    entity_type TEXT NOT NULL CHECK(entity_type IN ('relationship','event')),
                    entity_id TEXT NOT NULL,
                    old_status TEXT NOT NULL,
                    new_status TEXT NOT NULL,
                    reviewer TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
            current = db.execute("SELECT version FROM schema_info LIMIT 1").fetchone()
            if current is None:
                db.execute("INSERT INTO schema_info(version) VALUES(?)", (SCHEMA_VERSION,))
            elif current["version"] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Database schema version {current['version']} is not supported by "
                    f"this module (expected {SCHEMA_VERSION}). Back up the database before migration."
                )

    def create_case(self, title: str, description: str = "", case_id: str | None = None) -> str:
        case_id = case_id or "case_" + uuid.uuid4().hex[:16]
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO cases(case_id,title,description,created_at) VALUES(?,?,?,?)",
                (case_id, title.strip() or "Untitled case", description, utc_now()),
            )
        return case_id

    def add_source(
        self,
        *,
        case_id: str,
        source_id: str,
        source_type: str,
        file_name: str,
        file_path: str,
        sha256: str = "",
        camera_label: str = "",
        location_label: str = "",
        clock_offset_seconds: float = 0.0,
        clock_alignment_verified: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if source_type not in VALID_SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {sorted(VALID_SOURCE_TYPES)}")
        with self._connect() as db:
            db.execute(
                """INSERT INTO sources(
                    source_id,case_id,source_type,file_name,file_path,sha256,camera_label,
                    location_label,clock_offset_seconds,clock_alignment_verified,metadata_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(source_id) DO UPDATE SET
                    file_name=excluded.file_name,file_path=excluded.file_path,sha256=excluded.sha256,
                    camera_label=excluded.camera_label,location_label=excluded.location_label,
                    clock_offset_seconds=excluded.clock_offset_seconds,
                    clock_alignment_verified=excluded.clock_alignment_verified,
                    metadata_json=excluded.metadata_json""",
                (source_id, case_id, source_type, file_name, file_path, sha256, camera_label,
                 location_label, float(clock_offset_seconds), int(clock_alignment_verified),
                 _json(metadata), utc_now()),
            )
        return source_id

    def set_source_clock_alignment(
        self, source_id: str, offset_seconds: float, *, verified: bool
    ) -> None:
        """Map source-relative seconds to a shared case timeline: aligned = source + offset."""
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE sources SET clock_offset_seconds=?, clock_alignment_verified=? WHERE source_id=?",
                (float(offset_seconds), int(verified), source_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown source: {source_id}")

    def add_observation(
        self,
        *,
        case_id: str,
        source_id: str,
        observation_type: str,
        label: str = "",
        observation_id: str | None = None,
        confidence: float | None = None,
        source_start_seconds: float | None = None,
        source_end_seconds: float | None = None,
        track_local_id: str = "",
        original_text: str = "",
        translated_text: str = "",
        attributes: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        if observation_type not in VALID_OBSERVATION_TYPES:
            raise ValueError(f"Unsupported observation_type: {observation_type}")
        for name, value in (("source_start_seconds", source_start_seconds),
                            ("source_end_seconds", source_end_seconds)):
            if value is not None and float(value) < 0:
                raise ValueError(f"{name} cannot be negative")
        if (source_start_seconds is not None and source_end_seconds is not None
                and source_end_seconds < source_start_seconds):
            raise ValueError("source_end_seconds must be >= source_start_seconds")
        if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        observation_id = observation_id or "obs_" + uuid.uuid4().hex[:16]
        with self._connect() as db:
            source = db.execute(
                "SELECT case_id FROM sources WHERE source_id=?", (source_id,)
            ).fetchone()
            if source is None:
                raise KeyError(f"Unknown source: {source_id}")
            if source["case_id"] != case_id:
                raise ValueError("Observation and source must belong to the same case")
            db.execute(
                """INSERT OR IGNORE INTO observations(
                    observation_id,case_id,source_id,observation_type,label,confidence,
                    source_start_seconds,source_end_seconds,track_local_id,original_text,
                    translated_text,attributes_json,metadata_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (observation_id, case_id, source_id, observation_type, label,
                 None if confidence is None else float(confidence),
                 None if source_start_seconds is None else float(source_start_seconds),
                 None if source_end_seconds is None else float(source_end_seconds),
                 track_local_id, original_text, translated_text, _json(attributes), _json(metadata), utc_now()),
            )
        return observation_id

    def list_observations(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT o.*, s.source_type, s.file_name, s.camera_label, s.location_label,
                          s.clock_offset_seconds, s.clock_alignment_verified
                   FROM observations o JOIN sources s ON s.source_id=o.source_id
                   WHERE o.case_id=? ORDER BY o.created_at,o.observation_id""",
                (case_id,),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["attributes"] = _loads(item.pop("attributes_json"))
            item["metadata"] = _loads(item.pop("metadata_json"))
            item["clock_alignment_verified"] = bool(item["clock_alignment_verified"])
            results.append(item)
        return results

    @staticmethod
    def _transcript_mentions_object(transcript: str, label: str) -> tuple[bool, str | None]:
        text = _normalise(transcript)
        label = _normalise(label)
        synonyms = {
            "backpack": ("backpack", "rucksack", "bag"),
            "handbag": ("handbag", "purse", "bag"),
            "suitcase": ("suitcase", "luggage", "trolley bag"),
            "car": ("car", "vehicle", "automobile"),
            "motorcycle": ("motorcycle", "motorbike", "bike"),
        }
        terms = synonyms.get(label, (label,))
        for term in terms:
            if term and re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text):
                return True, term
        return False, None

    def _insert_relationship(
        self,
        *, case_id: str,
        observation_a: str,
        observation_b: str,
        relation_type: str,
        rule_name: str,
        explanation: str,
        factors: dict[str, Any],
        heuristic_score: float | None = None,
    ) -> tuple[str, bool]:
        left, right = sorted((observation_a, observation_b))
        relationship_id = _stable_relationship_id(case_id, left, right, relation_type)
        now = utc_now()
        with self._connect() as db:
            cursor = db.execute(
                """INSERT OR IGNORE INTO relationships(
                    relationship_id,case_id,source_observation_id,target_observation_id,
                    relation_type,origin,status,rule_name,explanation,heuristic_score,
                    factors_json,created_at,updated_at
                ) VALUES(?,?,?,?,?,'ai','pending',?,?,?,?,?,?)""",
                (relationship_id, case_id, left, right, relation_type, rule_name,
                 explanation, heuristic_score, _json(factors), now, now),
            )
            return relationship_id, cursor.rowcount == 1

    def propose_relationships(
        self,
        case_id: str,
        *,
        temporal_window_seconds: float = 120.0,
    ) -> list[dict[str, Any]]:
        """Create explainable candidate links. Scores are heuristics, not probabilities."""
        if temporal_window_seconds < 0:
            raise ValueError("temporal_window_seconds cannot be negative")
        observations = self.list_observations(case_id)
        created: list[dict[str, Any]] = []

        for index, a in enumerate(observations):
            for b in observations[index + 1:]:
                if a["source_id"] == b["source_id"]:
                    continue

                # Rule 1: possible visual match. Same class alone is deliberately insufficient.
                if (a["observation_type"] in OBJECT_OBSERVATION_TYPES
                        and b["observation_type"] in OBJECT_OBSERVATION_TYPES
                        and _normalise(a["label"]) == _normalise(b["label"])
                        and _normalise(a["label"])):
                    useful_keys = {"color", "shape", "pattern", "distinctive_features", "size", "material"}
                    attrs_a, attrs_b = a["attributes"], b["attributes"]
                    matched = {
                        key: attrs_a[key] for key in useful_keys
                        if key in attrs_a and key in attrs_b
                        and _normalise(attrs_a[key]) == _normalise(attrs_b[key])
                    }
                    if matched:
                        score = min(0.85, 0.40 + 0.10 * len(matched))
                        rid, was_created = self._insert_relationship(
                            case_id=case_id, observation_a=a["observation_id"],
                            observation_b=b["observation_id"], relation_type="POSSIBLE_VISUAL_MATCH",
                            rule_name="shared_object_attributes",
                            explanation=("The observations have the same detected class and share "
                                         f"these recorded appearance attributes: {', '.join(sorted(matched))}. "
                                         "Inspect the original frames before deciding whether they show "
                                         "the same physical object."),
                            factors={"class_match": True, "matching_attributes": matched,
                                     "comparison_limit": "Attribute agreement is not object identity."},
                            heuristic_score=score,
                        )
                        if was_created:
                            created.append({"relationship_id": rid, "type": "POSSIBLE_VISUAL_MATCH"})

                # Rule 2: time proximity only when both source clocks are verified/aligned.
                time_a, time_b = a["source_start_seconds"], b["source_start_seconds"]
                if (time_a is not None and time_b is not None
                        and a["clock_alignment_verified"] and b["clock_alignment_verified"]):
                    aligned_a = float(time_a) + float(a["clock_offset_seconds"])
                    aligned_b = float(time_b) + float(b["clock_offset_seconds"])
                    delta = abs(aligned_a - aligned_b)
                    if delta <= temporal_window_seconds:
                        rid, was_created = self._insert_relationship(
                            case_id=case_id, observation_a=a["observation_id"],
                            observation_b=b["observation_id"], relation_type="TEMPORALLY_NEAR",
                            rule_name="verified_clock_proximity",
                            explanation=(f"The observations are {delta:.1f} seconds apart on the "
                                         "configured shared case timeline. Temporal proximity alone "
                                         "does not establish that they belong to the same event."),
                            factors={"aligned_time_a_seconds": round(aligned_a, 3),
                                     "aligned_time_b_seconds": round(aligned_b, 3),
                                     "absolute_difference_seconds": round(delta, 3),
                                     "window_seconds": float(temporal_window_seconds),
                                     "clock_alignment_verified_for_both_sources": True},
                        )
                        if was_created:
                            created.append({"relationship_id": rid, "type": "TEMPORALLY_NEAR"})

                # Rule 3: transcript mentions an object category/synonym.
                transcript = None
                visual = None
                if a["observation_type"] == "transcript_segment" and b["observation_type"] in OBJECT_OBSERVATION_TYPES:
                    transcript, visual = a, b
                elif b["observation_type"] == "transcript_segment" and a["observation_type"] in OBJECT_OBSERVATION_TYPES:
                    transcript, visual = b, a
                if transcript and visual:
                    text_value = " ".join((transcript["original_text"], transcript["translated_text"]))
                    mentioned, matched_term = self._transcript_mentions_object(text_value, visual["label"])
                    if mentioned:
                        rid, was_created = self._insert_relationship(
                            case_id=case_id, observation_a=transcript["observation_id"],
                            observation_b=visual["observation_id"], relation_type="CONTEXTUALLY_RELATED",
                            rule_name="transcript_keyword_match",
                            explanation=(f"The transcript contains the term '{matched_term}', which may "
                                         f"refer to the detected category '{visual['label']}'. This is "
                                         "a keyword-level contextual suggestion, not proof of a connection."),
                            factors={"matched_term": matched_term,
                                     "original_transcript_preserved": bool(transcript["original_text"]),
                                     "translation_present": bool(transcript["translated_text"]),
                                     "semantic_model_used": False},
                        )
                        if was_created:
                            created.append({"relationship_id": rid, "type": "CONTEXTUALLY_RELATED"})
        return created

    def list_relationships(self, case_id: str, status: str | None = None) -> list[dict[str, Any]]:
        if status is not None and status not in VALID_REVIEW_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_REVIEW_STATUSES)}")
        sql = "SELECT * FROM relationships WHERE case_id=?"
        params: list[Any] = [case_id]
        if status:
            sql += " AND status=?"
            params.append(status)
        sql += " ORDER BY created_at,relationship_id"
        with self._connect() as db:
            rows = db.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["factors"] = _loads(item.pop("factors_json"))
            result.append(item)
        return result

    def review_relationship(
        self, relationship_id: str, decision: str, *, reviewer: str, notes: str = ""
    ) -> None:
        if decision not in VALID_REVIEW_STATUSES - {"pending"}:
            raise ValueError("decision must be approved, rejected or flagged")
        with self._connect() as db:
            row = db.execute(
                "SELECT case_id,status FROM relationships WHERE relationship_id=?",
                (relationship_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown relationship: {relationship_id}")
            now = utc_now()
            db.execute(
                """UPDATE relationships SET status=?,origin='investigator',reviewer=?,
                   reviewed_at=?,notes=?,updated_at=? WHERE relationship_id=?""",
                (decision, reviewer.strip() or "Investigator", now, notes, now, relationship_id),
            )
            db.execute(
                """INSERT INTO review_actions(action_id,case_id,entity_type,entity_id,
                   old_status,new_status,reviewer,notes,created_at)
                   VALUES(?,?, 'relationship', ?,?,?,?,?,?)""",
                ("act_" + uuid.uuid4().hex[:16], row["case_id"], relationship_id,
                 row["status"], decision, reviewer.strip() or "Investigator", notes, now),
            )

    def create_event(
        self,
        *,
        case_id: str,
        title: str,
        observation_ids: Iterable[str],
        description: str = "",
        added_by: str = "investigator",
    ) -> str:
        ids = list(dict.fromkeys(str(item) for item in observation_ids))
        if not ids:
            raise ValueError("An event must contain at least one observation")
        event_id = "evt_" + uuid.uuid4().hex[:16]
        now = utc_now()
        with self._connect() as db:
            placeholders = ",".join("?" for _ in ids)
            rows = db.execute(
                f"SELECT observation_id FROM observations WHERE case_id=? AND observation_id IN ({placeholders})",
                [case_id, *ids],
            ).fetchall()
            found = {row["observation_id"] for row in rows}
            missing = [item for item in ids if item not in found]
            if missing:
                raise ValueError(f"Observations do not exist in this case: {missing}")
            db.execute(
                "INSERT INTO events(event_id,case_id,title,description,status,created_at,updated_at) "
                "VALUES(?,?,?,?, 'pending',?,?)",
                (event_id, case_id, title.strip() or "Candidate event", description, now, now),
            )
            db.executemany(
                "INSERT INTO event_members(event_id,observation_id,added_by,created_at) VALUES(?,?,?,?)",
                [(event_id, observation_id, added_by, now) for observation_id in ids],
            )
        return event_id

    def review_event(
        self, event_id: str, decision: str, *, reviewer: str, notes: str = ""
    ) -> None:
        if decision not in VALID_REVIEW_STATUSES - {"pending"}:
            raise ValueError("decision must be approved, rejected or flagged")
        with self._connect() as db:
            row = db.execute("SELECT case_id,status FROM events WHERE event_id=?", (event_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown event: {event_id}")
            now = utc_now()
            db.execute(
                "UPDATE events SET status=?,reviewer=?,reviewed_at=?,notes=?,updated_at=? WHERE event_id=?",
                (decision, reviewer.strip() or "Investigator", now, notes, now, event_id),
            )
            db.execute(
                """INSERT INTO review_actions(action_id,case_id,entity_type,entity_id,
                   old_status,new_status,reviewer,notes,created_at)
                   VALUES(?,?, 'event', ?,?,?,?,?,?)""",
                ("act_" + uuid.uuid4().hex[:16], row["case_id"], event_id,
                 row["status"], decision, reviewer.strip() or "Investigator", notes, now),
            )

    def graph_payload(self, case_id: str) -> dict[str, list[dict[str, Any]]]:
        """Return renderer-neutral nodes/edges for NetworkX, PyVis or another UI."""
        with self._connect() as db:
            sources = db.execute("SELECT * FROM sources WHERE case_id=?", (case_id,)).fetchall()
            observations = db.execute(
                "SELECT * FROM observations WHERE case_id=?", (case_id,)
            ).fetchall()
            relationships = db.execute(
                "SELECT * FROM relationships WHERE case_id=?", (case_id,)
            ).fetchall()
            events = db.execute("SELECT * FROM events WHERE case_id=?", (case_id,)).fetchall()
            memberships = db.execute(
                """SELECT em.*, e.status AS event_status, e.title AS event_title
                   FROM event_members em JOIN events e ON e.event_id=em.event_id
                   WHERE e.case_id=?""", (case_id,)
            ).fetchall()

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        for row in sources:
            nodes.append({"id": row["source_id"], "label": row["file_name"],
                          "type": "source", "source_type": row["source_type"],
                          "camera_label": row["camera_label"], "location_label": row["location_label"]})
        for row in observations:
            nodes.append({"id": row["observation_id"], "label": row["label"] or row["observation_type"],
                          "type": "observation", "observation_type": row["observation_type"],
                          "source_id": row["source_id"], "source_start_seconds": row["source_start_seconds"],
                          "confidence": row["confidence"], "track_local_id": row["track_local_id"]})
            edges.append({"source": row["source_id"], "target": row["observation_id"],
                          "type": "CONTAINS", "status": "structural", "origin": "system"})
        for row in relationships:
            edges.append({"source": row["source_observation_id"],
                          "target": row["target_observation_id"],
                          "type": row["relation_type"], "status": row["status"],
                          "origin": row["origin"], "relationship_id": row["relationship_id"],
                          "explanation": row["explanation"],
                          "heuristic_score": row["heuristic_score"]})
        for row in events:
            nodes.append({"id": row["event_id"], "label": row["title"],
                          "type": "event", "status": row["status"]})
        for row in memberships:
            edges.append({"source": row["event_id"], "target": row["observation_id"],
                          "type": "HAS_OBSERVATION", "status": row["event_status"],
                          "origin": row["added_by"], "event_title": row["event_title"]})
        return {"nodes": nodes, "edges": edges}

    def review_history(self, case_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM review_actions WHERE case_id=? ORDER BY created_at,action_id",
                (case_id,),
            ).fetchall()
        return [dict(row) for row in rows]
