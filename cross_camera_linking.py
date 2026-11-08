
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from event_reconstruction import get_reconstructed_events


ROOT = Path(__file__).resolve().parent
DATABASE = ROOT / "outputs" / "cross_camera_links.db"

REVIEW_STATUSES = ["Pending", "Approved", "Rejected", "Flagged"]

DATE_FORMAT = "%d-%m-%Y %H:%M:%S"


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


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

    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM camera_time_anchors
            ORDER BY source_name
        """).fetchall()

    return pd.DataFrame([dict(row) for row in rows])


def save_anchor(source_id, source_name, recording_start):

    parsed = datetime.strptime(
        recording_start.strip(),
        DATE_FORMAT
    )

    with get_connection() as conn:

        conn.execute("""
            INSERT INTO camera_time_anchors (
                source_id,
                source_name,
                recording_start,
                updated_at
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(source_id) DO UPDATE SET
                source_name = excluded.source_name,
                recording_start = excluded.recording_start,
                updated_at = excluded.updated_at
        """, (
            source_id,
            source_name,
            parsed.isoformat(sep=" "),
            datetime.now().isoformat(sep=" ", timespec="seconds")
        ))


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

def get_links():

    with get_connection() as conn:

        rows = conn.execute("""
            SELECT * FROM cross_camera_links
            ORDER BY time_gap_seconds ASC
        """).fetchall()

    return pd.DataFrame([dict(row) for row in rows])


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


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

def render_cross_camera_linking():

    initialise_database()

    st.header("Cross-Camera Evidence Linking Engine")

    st.caption(
        "Find potentially related events across CCTV sources "
        "using registered recording times and detected categories."
    )

    st.warning(
        "Enter verified CCTV recording start times. "
        "Do not use WhatsApp receipt times, file creation times, "
        "or assumed timestamps as actual recording times."
    )

    events = get_reconstructed_events()

    if events is None or events.empty:

        st.info(
            "Generate candidate events in the Event Reconstruction "
            "Engine before using cross-camera linking."
        )

        return

    sources = (
        events[["source_id", "source"]]
        .drop_duplicates()
        .dropna(subset=["source_id"])
    )

    # --------------------------------------------------------
    # CAMERA REGISTRATION
    # --------------------------------------------------------

    st.subheader("1. Register CCTV recording times")

    source_options = {
        f"{row['source']} ({row['source_id']})":
        (str(row["source_id"]), str(row["source"]))
        for _, row in sources.iterrows()
    }

    selected_source = st.selectbox(
        "Select CCTV source",
        list(source_options.keys()),
        key="cross_camera_source"
    )

    source_id, source_name = source_options[selected_source]

    anchors = get_anchors()

    existing = (
        anchors[
            anchors["source_id"].astype(str) == source_id
        ]
        if not anchors.empty
        else pd.DataFrame()
    )

    existing_value = ""

    if not existing.empty:

        stored_time = datetime.fromisoformat(
            str(existing.iloc[0]["recording_start"])
        )

        existing_value = stored_time.strftime(DATE_FORMAT)

    st.caption(
        "Enter the actual date and time when this CCTV recording began."
    )

    recording_start = st.text_input(
        "Recording start time (DD-MM-YYYY HH:MM:SS)",
        value=existing_value,
        placeholder="06-11-2026 14:30:00",
        key=f"anchor_input_{source_id}"
    )

    st.caption(
        "Example: 06-11-2026 14:30:00 means "
        "6 November 2026 at 2:30 PM."
    )

    if st.button(
        "Save camera time anchor",
        use_container_width=True
    ):

        try:

            save_anchor(
                source_id,
                source_name,
                recording_start.strip()
            )

            st.success("Camera recording time saved.")

            st.rerun()

        except ValueError:

            st.error(
                "Invalid date/time. "
                "Use DD-MM-YYYY HH:MM:SS."
            )

    # --------------------------------------------------------
    # REGISTERED CAMERAS
    # --------------------------------------------------------

    st.markdown("**Registered camera time anchors**")

    anchors = get_anchors()

    if anchors.empty:

        st.info("No camera recording times registered yet.")

    else:

        display_anchors = anchors.copy()

        display_anchors["recording_start"] = (
            display_anchors["recording_start"].apply(
                lambda value: datetime.fromisoformat(
                    str(value)
                ).strftime(DATE_FORMAT)
            )
        )

        st.dataframe(
            display_anchors[
                [
                    "source_name",
                    "source_id",
                    "recording_start"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # --------------------------------------------------------
    # GENERATE PROPOSALS
    # --------------------------------------------------------

    st.subheader("2. Generate cross-camera proposals")

    max_gap = st.slider(
        "Maximum separation between event intervals (seconds)",
        min_value=5,
        max_value=600,
        value=60,
        step=5,
        key="cross_camera_max_gap"
    )

    if st.button(
        "Find potentially related events",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Comparing events across camera sources..."
        ):

            count = generate_cross_camera_links(max_gap)

        st.success(
            f"Cross-camera comparison completed. "
            f"{count} candidate links generated or refreshed."
        )

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # LINK DASHBOARD
    # --------------------------------------------------------

    st.subheader("3. Cross-camera link dashboard")

    links = get_links()

    if links.empty:

        st.info(
            "No cross-camera links yet. Register recording times "
            "for at least two sources, then generate proposals."
        )

        return

    total = len(links)

    pending = int(
        (links["review_status"] == "Pending").sum()
    )

    approved = int(
        (links["review_status"] == "Approved").sum()
    )

    rejected = int(
        (links["review_status"] == "Rejected").sum()
    )

    flagged = int(
        (links["review_status"] == "Flagged").sum()
    )

    metrics = st.columns(5)

    metrics[0].metric("Total links", total)
    metrics[1].metric("Pending", pending)
    metrics[2].metric("Approved", approved)
    metrics[3].metric("Rejected", rejected)
    metrics[4].metric("Flagged", flagged)

    st.dataframe(
        links[
            [
                "link_id",
                "source_a",
                "source_b",
                "category",
                "time_gap_seconds",
                "review_status"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # LINK INSPECTION
    # --------------------------------------------------------

    st.subheader("4. Inspect and verify a proposed link")

    link_options = {
        f"{row['link_id']} | "
        f"{row['source_a']} <-> {row['source_b']}":
        row["link_id"]

        for _, row in links.iterrows()
    }

    selected_label = st.selectbox(
        "Select a proposed link",
        list(link_options.keys()),
        key="selected_cross_camera_link"
    )

    selected_id = link_options[selected_label]

    selected = links[
        links["link_id"] == selected_id
    ].iloc[0]

    st.markdown("**Proposed relationship**")

    st.write(selected["explanation"])

    st.write({
        "Link ID": selected["link_id"],
        "Event A": selected["event_a"],
        "Event B": selected["event_b"],
        "Camera A": selected["source_a"],
        "Camera B": selected["source_b"],
        "Detected category": selected["category"],
        "Event A absolute start": selected["event_a_time"],
        "Event B absolute start": selected["event_b_time"],
        "Interval separation (seconds)": selected["time_gap_seconds"]
    })

    st.info(
        "A matching category and nearby time do not establish "
        "that the same physical person or object appears in both videos."
    )

    current_status = selected["review_status"]

    status = st.selectbox(
        "Investigator decision",
        REVIEW_STATUSES,
        index=REVIEW_STATUSES.index(current_status),
        key=f"cross_review_status_{selected_id}"
    )

    notes = st.text_area(
        "Investigator notes",
        value=selected["reviewer_notes"] or "",
        key=f"cross_review_notes_{selected_id}"
    )

    if st.button(
        "Save link verification decision",
        type="primary",
        use_container_width=True,
        key=f"save_cross_review_{selected_id}"
    ):

        try:

            review_link(
                selected_id,
                status,
                notes
            )

            st.success(
                "Link decision and review history saved."
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Could not save decision: {error}"
            )

    # --------------------------------------------------------
    # REVIEW HISTORY
    # --------------------------------------------------------

    st.markdown("**Link review history**")

    history = get_link_history(selected_id)

    if history.empty:

        st.info("No review history for this link yet.")

    else:

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Only investigator-reviewed links should be treated "
        "as verified relationships. Original evidence remains unchanged."
    )
