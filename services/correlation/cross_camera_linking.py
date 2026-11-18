
import hashlib
from datetime import datetime, timedelta

import pandas as pd

from services.events.event_reconstruction import get_reconstructed_events


from core.config import CROSS_CAMERA_DB_PATH, REVIEW_STATUSES
from storage import db_manager
from storage.sqlite_utils import connect as sqlite_connect

DATABASE = CROSS_CAMERA_DB_PATH

DATE_FORMAT = "%d-%m-%Y %H:%M:%S"


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite_connect(DATABASE)


def initialise_database():
    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS camera_time_anchors (
                source_id TEXT PRIMARY KEY,
                source_name TEXT NOT NULL,
                recording_start TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_camera_links (
                link_id TEXT PRIMARY KEY,
                event_a TEXT NOT NULL,
                event_b TEXT NOT NULL,
                source_a TEXT NOT NULL,
                source_b TEXT NOT NULL,
                category TEXT NOT NULL,
                event_a_time TEXT NOT NULL,
                event_b_time TEXT NOT NULL,
                time_gap_seconds REAL NOT NULL,
                explanation TEXT NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'Pending',
                reviewer_notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_camera_review_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                link_id TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                reviewer_notes TEXT,
                reviewed_at TEXT NOT NULL
            )
        """)


# ============================================================
# CAMERA TIME ANCHORS
# ============================================================

def get_anchors():
    """Recording start times, read from the master evidence registry (single timebase)."""

    rows = [
        {
            "source_id": s["source_id"],
            "source_name": s["source_name"],
            "recording_start": s["recorded_start"],
        }
        for s in db_manager.get_all_sources("CCTV")
        if (s.get("recorded_start") or "").strip()
    ]

    return pd.DataFrame(rows, columns=["source_id", "source_name", "recording_start"])


def save_anchor(source_id, source_name, recording_start):
    """Parse a DD-MM-YYYY HH:MM:SS start time and store it on the evidence source."""

    parsed = datetime.strptime(
        recording_start.strip(),
        DATE_FORMAT
    )

    db_manager.update_source_metadata(
        source_id,
        recorded_start=parsed.isoformat(sep=" ", timespec="seconds"),
    )


# ============================================================
# EVENT LINKING
# ============================================================

def make_link_id(event_a, event_b):

    pair = "|".join(sorted([
        str(event_a),
        str(event_b)
    ]))

    digest = hashlib.sha256(
        pair.encode()
    ).hexdigest()[:12].upper()

    return f"LINK-{digest}"


def interval_gap(start_a, end_a, start_b, end_b):

    if end_a < start_b:
        return (start_b - end_a).total_seconds()

    if end_b < start_a:
        return (start_a - end_b).total_seconds()

    return 0.0


def generate_cross_camera_links(max_gap_seconds=60):

    events = get_reconstructed_events()

    if events is None or events.empty:
        return 0

    anchors_df = get_anchors()

    if anchors_df.empty:
        return 0

    anchors = {
        str(row["source_id"]): datetime.fromisoformat(
            str(row["recording_start"])
        )
        for _, row in anchors_df.iterrows()
    }

    prepared = []

    for event in events.to_dict("records"):

        source_id = str(event.get("source_id") or "")

        category = str(
            event.get("object_category") or ""
        ).strip().lower()

        if not source_id or source_id not in anchors:
            continue

        if not category or category in [
            "unknown", "none", "nan"
        ]:
            continue

        try:
            start_seconds = float(event["start_seconds"])
            end_seconds = float(event["end_seconds"])

        except (TypeError, ValueError, KeyError):
            continue

        if end_seconds < start_seconds:
            continue

        recording_start = anchors[source_id]

        # Convert source-relative timestamps into absolute times.
        # Uses standard Python datetime arithmetic.

        absolute_start = recording_start + timedelta(
            seconds=start_seconds
        )

        absolute_end = recording_start + timedelta(
            seconds=end_seconds
        )

        prepared.append({
            "event_id": str(event["event_id"]),
            "source_id": source_id,
            "source": str(event.get("source") or source_id),
            "category": category,
            "start": absolute_start,
            "end": absolute_end
        })

    proposals = []

    for index, event_a in enumerate(prepared):

        for event_b in prepared[index + 1:]:

            # Do not compare events from the same camera.
            if event_a["source_id"] == event_b["source_id"]:
                continue

            # Require matching detected categories.
            if event_a["category"] != event_b["category"]:
                continue

            gap = interval_gap(
                event_a["start"],
                event_a["end"],
                event_b["start"],
                event_b["end"]
            )

            if gap > max_gap_seconds:
                continue

            event_ids = sorted([
                event_a["event_id"],
                event_b["event_id"]
            ])

            link_id = make_link_id(*event_ids)

            proposals.append({
                "link_id": link_id,
                "event_a": event_a["event_id"],
                "event_b": event_b["event_id"],
                "source_a": event_a["source"],
                "source_b": event_b["source"],
                "category": event_a["category"],
                "event_a_time": event_a["start"].isoformat(sep=" "),
                "event_b_time": event_b["start"].isoformat(sep=" "),
                "time_gap_seconds": gap,

                "explanation": (
                    f"Two different CCTV sources contain events "
                    f"with the same detected category "
                    f"({event_a['category']}). Their event intervals "
                    f"are separated by {gap:.2f} seconds or less. "
                    f"This is a temporal and category-based candidate "
                    f"link, not proof of physical identity."
                )
            })

    now = datetime.now().isoformat(
        sep=" ",
        timespec="seconds"
    )

    with get_connection() as conn:

        for proposal in proposals:

            conn.execute("""
                INSERT INTO cross_camera_links (
                    link_id,
                    event_a,
                    event_b,
                    source_a,
                    source_b,
                    category,
                    event_a_time,
                    event_b_time,
                    time_gap_seconds,
                    explanation,
                    review_status,
                    reviewer_notes,
                    created_at,
                    updated_at
                )

                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    'Pending', '', ?, ?
                )

                ON CONFLICT(link_id) DO UPDATE SET
                    source_a = excluded.source_a,
                    source_b = excluded.source_b,
                    category = excluded.category,
                    event_a_time = excluded.event_a_time,
                    event_b_time = excluded.event_b_time,
                    time_gap_seconds = excluded.time_gap_seconds,
                    explanation = excluded.explanation,
                    updated_at = excluded.updated_at
            """, (
                proposal["link_id"],
                proposal["event_a"],
                proposal["event_b"],
                proposal["source_a"],
                proposal["source_b"],
                proposal["category"],
                proposal["event_a_time"],
                proposal["event_b_time"],
                proposal["time_gap_seconds"],
                proposal["explanation"],
                now,
                now
            ))

    return len(proposals)


# ============================================================
# RETRIEVE LINKS
# ============================================================

def get_links(active_events_only=True):

    initialise_database()

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT * FROM cross_camera_links
            ORDER BY time_gap_seconds ASC
        """).fetchall()

    links = pd.DataFrame([dict(row) for row in rows])

    # Links whose events were superseded by a later reconstruction are stale.
    if active_events_only and not links.empty:
        events = get_reconstructed_events(active_only=True)
        active_ids = set(events["event_id"].astype(str)) if not events.empty else set()
        links = links[
            links["event_a"].astype(str).isin(active_ids)
            & links["event_b"].astype(str).isin(active_ids)
        ]

    return links


# ============================================================
# INVESTIGATOR REVIEW
# ============================================================

def review_link(link_id, new_status, notes):

    if new_status not in REVIEW_STATUSES:
        raise ValueError("Invalid review status.")

    now = datetime.now().isoformat(
        sep=" ",
        timespec="seconds"
    )

    with get_connection() as conn:

        existing = conn.execute("""
            SELECT review_status
            FROM cross_camera_links
            WHERE link_id = ?
        """, (link_id,)).fetchone()

        if existing is None:
            raise ValueError("Link not found.")

        previous_status = existing["review_status"]

        conn.execute("""
            UPDATE cross_camera_links
            SET review_status = ?,
                reviewer_notes = ?,
                updated_at = ?
            WHERE link_id = ?
        """, (
            new_status,
            notes,
            now,
            link_id
        ))

        if previous_status != new_status or notes:

            conn.execute("""
                INSERT INTO cross_camera_review_history (
                    link_id,
                    previous_status,
                    new_status,
                    reviewer_notes,
                    reviewed_at
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                link_id,
                previous_status,
                new_status,
                notes,
                now
            ))


def get_link_history(link_id):

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT
                previous_status,
                new_status,
                reviewer_notes,
                reviewed_at

            FROM cross_camera_review_history

            WHERE link_id = ?

            ORDER BY history_id DESC
        """, (link_id,)).fetchall()

    return pd.DataFrame([dict(row) for row in rows])
