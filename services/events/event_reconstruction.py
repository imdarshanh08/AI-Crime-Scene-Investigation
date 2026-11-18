
import hashlib
from datetime import datetime

import pandas as pd


# =========================================================
# EVENT RECONSTRUCTION ENGINE v1.0
# =========================================================

from core.config import EVENTS_DB_PATH
from storage.sqlite_utils import connect as sqlite_connect

DATABASE_PATH = EVENTS_DB_PATH

MIN_OBSERVATIONS = 2
DEFAULT_MAX_GAP_SECONDS = 30


# =========================================================
# DATABASE
# =========================================================

def get_connection():
    return sqlite_connect(DATABASE_PATH)


def initialise_database():
    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS reconstructed_events (
                event_id TEXT PRIMARY KEY,
                event_title TEXT,
                event_type TEXT,
                source TEXT,
                source_id TEXT,
                modality TEXT,
                object_category TEXT,
                start_seconds REAL,
                end_seconds REAL,
                observation_count INTEGER,
                average_confidence REAL,
                event_explanation TEXT,
                review_status TEXT DEFAULT 'Pending',
                reviewer_notes TEXT DEFAULT '',
                created_at TEXT,
                reviewed_at TEXT,
                active INTEGER DEFAULT 1
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS event_observations (
                event_id TEXT,
                observation_id TEXT,
                PRIMARY KEY (event_id, observation_id),
                FOREIGN KEY (event_id)
                    REFERENCES reconstructed_events(event_id)
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS event_review_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT,
                previous_status TEXT,
                new_status TEXT,
                reviewer_notes TEXT,
                reviewed_at TEXT
            )
        """)

        connection.commit()


# =========================================================
# DATA NORMALISATION
# =========================================================

def normalise_observations(observations):

    if observations is None or observations.empty:
        return pd.DataFrame()

    df = observations.copy()

    column_aliases = {
        "observation_id": "Observation ID",
        "source": "Source",
        "source_id": "Source ID",
        "modality": "Modality",
        "timestamp_seconds": "Timestamp (seconds)",
        "object": "Object",
        "track_id": "Track ID",
        "raw_track_id": "Raw Track ID",
        "confidence": "Confidence",
        "observation": "Observation",
        "evidence_path": "Evidence Path",
    }

    for column in column_aliases.values():
        if column not in df.columns:
            df[column] = None

    df["Timestamp (seconds)"] = pd.to_numeric(
        df["Timestamp (seconds)"], errors="coerce"
    )

    df["Confidence"] = pd.to_numeric(
        df["Confidence"], errors="coerce"
    )

    df["Observation ID"] = df["Observation ID"].fillna("").astype(str)
    df["Source ID"] = df["Source ID"].fillna("").astype(str)
    df["Source"] = df["Source"].fillna("Unknown source").astype(str)
    df["Modality"] = df["Modality"].fillna("Unknown").astype(str)
    df["Object"] = df["Object"].fillna("unknown").astype(str)

    df["Track ID"] = df["Track ID"].fillna("").astype(str)

    # Use Raw Track ID only when Track ID is unavailable.
    missing_track = df["Track ID"].isin(["", "None", "nan"])
    df.loc[missing_track, "Track ID"] = (
        df.loc[missing_track, "Raw Track ID"].fillna("").astype(str)
    )

    df = df[
        (df["Observation ID"] != "")
        & (df["Timestamp (seconds)"].notna())
    ].copy()

    df = df.drop_duplicates(subset=["Observation ID"])

    return df


# =========================================================
# STABLE EVENT IDENTIFIERS
# =========================================================

def generate_event_id(observation_ids):

    combined = "|".join(sorted(observation_ids))

    digest = hashlib.sha256(
        combined.encode("utf-8")
    ).hexdigest()[:12]

    return f"EVT-{digest.upper()}"


# =========================================================
# EVENT GROUPING
# =========================================================

def identify_track_events(
    observations,
    max_gap_seconds=DEFAULT_MAX_GAP_SECONDS,
    min_observations=MIN_OBSERVATIONS
):

    df = normalise_observations(observations)

    if df.empty:
        return []

    # Only CCTV observations with a usable track ID.
    cctv = df[
        (df["Modality"].str.lower() == "cctv")
        & (~df["Track ID"].isin(["", "None", "nan"]))
    ].copy()

    if cctv.empty:
        return []

    candidates = []

    # Track IDs are only compared within their own source.
    grouped = cctv.groupby(
        ["Source ID", "Track ID"],
        dropna=False
    )

    for (source_id, track_id), track_rows in grouped:

        track_rows = track_rows.sort_values(
            "Timestamp (seconds)"
        )

        current_group = []

        for _, row in track_rows.iterrows():

            if not current_group:
                current_group.append(row)
                continue

            previous_time = float(
                current_group[-1]["Timestamp (seconds)"]
            )

            current_time = float(
                row["Timestamp (seconds)"]
            )

            time_gap = current_time - previous_time

            if 0 <= time_gap <= max_gap_seconds:
                current_group.append(row)

            else:
                if len(current_group) >= min_observations:
                    candidates.append(current_group)

                current_group = [row]

        if len(current_group) >= min_observations:
            candidates.append(current_group)

    return candidates


# =========================================================
# EVENT DESCRIPTION
# =========================================================

def describe_event(rows):

    df = pd.DataFrame(rows)

    source = str(df["Source"].iloc[0])
    source_id = str(df["Source ID"].iloc[0])
    track_id = str(df["Track ID"].iloc[0])

    object_category = (
        df["Object"]
        .mode()
        .iloc[0]
        if not df["Object"].mode().empty
        else "unknown"
    )

    start_time = float(df["Timestamp (seconds)"].min())
    end_time = float(df["Timestamp (seconds)"].max())

    confidence = df["Confidence"].dropna()

    average_confidence = (
        float(confidence.mean())
        if not confidence.empty
        else None
    )

    observation_ids = df["Observation ID"].tolist()

    event_id = generate_event_id(observation_ids)

    event_title = (
        f"{object_category.title()} tracking event"
    )

    explanation = (
        f"{len(df)} observations associated with track "
        f"{track_id} in source {source}. "
        f"The observations span {end_time - start_time:.2f} "
        f"seconds of this source's recorded timeline. "
        "This is an AI-generated candidate event requiring "
        "investigator verification."
    )

    return {
        "event_id": event_id,
        "event_title": event_title,
        "event_type": "TRACK_CONTINUITY",
        "source": source,
        "source_id": source_id,
        "modality": "CCTV",
        "object_category": str(object_category),
        "start_seconds": start_time,
        "end_seconds": end_time,
        "observation_count": len(df),
        "average_confidence": average_confidence,
        "event_explanation": explanation,
        "observation_ids": observation_ids
    }


# =========================================================
# RECONSTRUCTION AND STORAGE
# =========================================================

def reconstruct_events(
    observations,
    max_gap_seconds=DEFAULT_MAX_GAP_SECONDS,
    min_observations=MIN_OBSERVATIONS
):

    initialise_database()

    groups = identify_track_events(
        observations,
        max_gap_seconds=max_gap_seconds,
        min_observations=min_observations
    )

    event_records = []

    with get_connection() as connection:

        # Mark old candidates inactive instead of deleting them.
        connection.execute("""
            UPDATE reconstructed_events
            SET active = 0
        """)

        for rows in groups:

            event = describe_event(rows)

            connection.execute("""
                INSERT INTO reconstructed_events (
                    event_id,
                    event_title,
                    event_type,
                    source,
                    source_id,
                    modality,
                    object_category,
                    start_seconds,
                    end_seconds,
                    observation_count,
                    average_confidence,
                    event_explanation,
                    created_at,
                    active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)

                ON CONFLICT(event_id) DO UPDATE SET
                    event_title = excluded.event_title,
                    event_type = excluded.event_type,
                    source = excluded.source,
                    source_id = excluded.source_id,
                    modality = excluded.modality,
                    object_category = excluded.object_category,
                    start_seconds = excluded.start_seconds,
                    end_seconds = excluded.end_seconds,
                    observation_count = excluded.observation_count,
                    average_confidence = excluded.average_confidence,
                    event_explanation = excluded.event_explanation,
                    active = 1
            """, (
                event["event_id"],
                event["event_title"],
                event["event_type"],
                event["source"],
                event["source_id"],
                event["modality"],
                event["object_category"],
                event["start_seconds"],
                event["end_seconds"],
                event["observation_count"],
                event["average_confidence"],
                event["event_explanation"],
                datetime.now().isoformat(timespec="seconds")
            ))

            connection.execute("""
                DELETE FROM event_observations
                WHERE event_id = ?
            """, (event["event_id"],))

            for observation_id in event["observation_ids"]:

                connection.execute("""
                    INSERT OR IGNORE INTO event_observations (
                        event_id,
                        observation_id
                    )
                    VALUES (?, ?)
                """, (
                    event["event_id"],
                    observation_id
                ))

            event_records.append(event)

        connection.commit()

    return pd.DataFrame(event_records)


# =========================================================
# RETRIEVE EVENTS
# =========================================================

def get_reconstructed_events(active_only=True):

    initialise_database()

    query = """
        SELECT *
        FROM reconstructed_events
    """

    if active_only:
        query += " WHERE active = 1"

    query += " ORDER BY source_id, start_seconds"

    with get_connection() as connection:
        return pd.read_sql_query(query, connection)


def get_event_observations(event_id):

    initialise_database()

    query = """
        SELECT
            eo.event_id,
            eo.observation_id
        FROM event_observations eo
        WHERE eo.event_id = ?
    """

    with get_connection() as connection:
        return pd.read_sql_query(
            query,
            connection,
            params=(event_id,)
        )


def get_all_event_memberships(active_only=True):
    """Return (event_id, observation_id) pairs, used to wire events into the evidence graph."""

    initialise_database()

    query = """
        SELECT eo.event_id, eo.observation_id
        FROM event_observations eo
        JOIN reconstructed_events re ON re.event_id = eo.event_id
    """

    if active_only:
        query += " WHERE re.active = 1"

    with get_connection() as connection:
        return pd.read_sql_query(query, connection)


def deactivate_all_events():
    """Retire every active candidate event (history and reviews are kept)."""

    initialise_database()

    with get_connection() as connection:
        connection.execute("UPDATE reconstructed_events SET active = 0")
        connection.commit()


# =========================================================
# INVESTIGATOR REVIEW
# =========================================================

def review_event(
    event_id,
    new_status,
    reviewer_notes=""
):

    allowed_statuses = [
        "Pending",
        "Approved",
        "Rejected",
        "Flagged"
    ]

    if new_status not in allowed_statuses:
        raise ValueError("Invalid event review status.")

    initialise_database()

    reviewed_at = datetime.now().isoformat(
        timespec="seconds"
    )

    with get_connection() as connection:

        existing = connection.execute("""
            SELECT review_status
            FROM reconstructed_events
            WHERE event_id = ?
        """, (event_id,)).fetchone()

        if existing is None:
            raise ValueError("Event not found.")

        previous_status = existing["review_status"]

        connection.execute("""
            UPDATE reconstructed_events
            SET
                review_status = ?,
                reviewer_notes = ?,
                reviewed_at = ?
            WHERE event_id = ?
        """, (
            new_status,
            reviewer_notes,
            reviewed_at,
            event_id
        ))

        connection.execute("""
            INSERT INTO event_review_history (
                event_id,
                previous_status,
                new_status,
                reviewer_notes,
                reviewed_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            event_id,
            previous_status,
            new_status,
            reviewer_notes,
            reviewed_at
        ))

        connection.commit()


# =========================================================
# REVIEW HISTORY
# =========================================================

def get_event_review_history(event_id):

    initialise_database()

    query = """
        SELECT *
        FROM event_review_history
        WHERE event_id = ?
        ORDER BY reviewed_at DESC
    """

    with get_connection() as connection:
        return pd.read_sql_query(
            query,
            connection,
            params=(event_id,)
        )


# =========================================================
# SUMMARY
# =========================================================

def get_event_summary():

    events = get_reconstructed_events()

    if events.empty:
        return {
            "total_events": 0,
            "pending_events": 0,
            "approved_events": 0,
            "rejected_events": 0,
            "flagged_events": 0
        }

    return {
        "total_events": len(events),
        "pending_events": int(
            (events["review_status"] == "Pending").sum()
        ),
        "approved_events": int(
            (events["review_status"] == "Approved").sum()
        ),
        "rejected_events": int(
            (events["review_status"] == "Rejected").sum()
        ),
        "flagged_events": int(
            (events["review_status"] == "Flagged").sum()
        )
    }


# =========================================================
# MODULE TEST
# =========================================================

if __name__ == "__main__":

    initialise_database()

    print("=" * 55)
    print("EVENT RECONSTRUCTION ENGINE v1.0")
    print("=" * 55)

    print("Database:", DATABASE_PATH)
    print("Database initialised successfully.")

    print("\nCurrent stored events:")

    events = get_reconstructed_events()

    if events.empty:
        print("No events reconstructed yet.")
        print(
            "The engine is ready to receive observations "
            "from app.py."
        )
    else:
        print(
            events[
                [
                    "event_id",
                    "event_title",
                    "source",
                    "object_category",
                    "observation_count",
                    "review_status"
                ]
            ].to_string(index=False)
        )

    print("\nEngine initialisation completed.")
