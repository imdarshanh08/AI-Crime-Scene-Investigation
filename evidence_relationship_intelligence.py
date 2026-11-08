
# ============================================================
# SECTION 27: EVIDENCE RELATIONSHIP INTELLIGENCE
# ============================================================

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 1. DATABASE LOCATIONS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

DATABASES = {
    "Evidence Graph": (
        OUTPUT_DIR / "evidence_graph_relationships.db"
    ),
    "Event Reconstruction": (
        OUTPUT_DIR / "event_reconstruction.db"
    ),
    "Cross-Camera Linking": (
        OUTPUT_DIR / "cross_camera_links.db"
    ),
}


# ------------------------------------------------------------
# 2. TABLE CLASSIFICATION
# ------------------------------------------------------------

def classify_table(table_name):
    name = table_name.lower()

    if "review_history" in name:
        return "Review History"

    if "event_observations" in name:
        return "Event Evidence"

    if "reconstructed_events" in name:
        return "Reconstructed Events"

    if "cross_camera_links" in name:
        return "Cross-Camera Relationships"

    if "graph_relationships" in name:
        return "Evidence Graph Relationships"

    if "camera_time_anchors" in name:
        return "Camera Time Anchors"

    return "Other Records"


# ------------------------------------------------------------
# 3. SAFE SQLITE DATABASE READER
# ------------------------------------------------------------

def read_database(database_path):
    path = Path(database_path)

    if not path.is_file():
        return [], [], f"Database not found: {path.name}"

    records = []
    tables_found = []

    try:
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro",
            uri=True,
            timeout=5,
        )

        connection.row_factory = sqlite3.Row

        try:
            tables = connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()

            for table_row in tables:
                table_name = table_row["name"]
                tables_found.append(table_name)

                # Table names come from sqlite_master.
                safe_table_name = table_name.replace('"', '""')

                rows = connection.execute(
                    f'SELECT * FROM "{safe_table_name}" LIMIT 10000'
                ).fetchall()

                for row in rows:
                    records.append({
                        "table": table_name,
                        "record_type": classify_table(table_name),
                        "data": dict(row),
                    })

        finally:
            connection.close()

        return records, tables_found, None

    except (sqlite3.Error, OSError, ValueError) as error:
        return [], tables_found, str(error)


# ------------------------------------------------------------
# 4. RECORD FIELD HELPERS
# ------------------------------------------------------------

def get_field(data, possible_names, default="Not available"):
    lookup = {
        str(key).lower(): value
        for key, value in data.items()
    }

    for name in possible_names:
        value = lookup.get(name.lower())

        if value is not None and str(value).strip():
            return value

    return default


def get_record_status(data):
    return str(
        get_field(
            data,
            [
                "review_status",
                "status",
                "review_state",
            ],
            default="Not specified",
        )
    )


def get_record_id(data):
    return str(
        get_field(
            data,
            [
                "relationship_id",
                "link_id",
                "event_id",
                "observation_id",
                "history_id",
                "id",
            ],
        )
    )


def get_record_summary(data, record_type):
    if record_type == "Evidence Graph Relationships":
        first = get_field(
            data,
            ["relationship_type", "relationship", "relation_type"],
        )
        return f"Evidence graph relationship: {first}"

    if record_type == "Cross-Camera Relationships":
        first = get_field(
            data,
            ["category", "relationship_type", "link_type"],
        )
        return f"Cross-camera candidate: {first}"

    if record_type == "Reconstructed Events":
        first = get_field(
            data,
            ["description", "event_description", "event_type", "title"],
        )
        return f"Reconstructed event: {first}"

    if record_type == "Event Evidence":
        event_id = get_field(data, ["event_id"])
        observation_id = get_field(
            data,
            ["observation_id", "evidence_id"],
        )
        return f"Event {event_id} contains observation {observation_id}"

    if record_type == "Camera Time Anchors":
        source = get_field(
            data,
            ["source_name", "source_id"],
        )
        return f"Camera recording time anchor: {source}"

    if record_type == "Review History":
        action = get_field(
            data,
            ["action", "review_status", "status", "decision"],
        )
        return f"Investigator review history: {action}"

    return "Stored database record"


def prepare_record(component, raw_record):
    data = raw_record["data"]
    record_type = raw_record["record_type"]

    searchable_text = " ".join(
        str(value)
        for value in data.values()
        if value is not None
    ).lower()

    return {
        "Component": component,
        "Record Type": record_type,
        "Record ID": get_record_id(data),
        "Status": get_record_status(data),
        "Summary": get_record_summary(data, record_type),
        "_search": searchable_text,
        "_details": data,
    }


# ------------------------------------------------------------
# 5. COLLECT ALL PERSISTED INVESTIGATION RECORDS
# ------------------------------------------------------------

def collect_relationship_records():
    all_records = []
    database_report = []

    for component, database_path in DATABASES.items():
        raw_records, tables, error = read_database(database_path)

        database_report.append({
            "Component": component,
            "Database": database_path.name,
            "Available": database_path.is_file(),
            "Tables": ", ".join(tables) if tables else "None found",
            "Error": error or "",
        })

        for raw_record in raw_records:
            all_records.append(
                prepare_record(component, raw_record)
            )

    return all_records, pd.DataFrame(database_report)


# ------------------------------------------------------------
# 6. SEARCH AND FILTER
# ------------------------------------------------------------

def filter_relationship_records(
    records,
    query="",
    selected_components=None,
    selected_statuses=None,
):
    filtered = list(records)

    if selected_components:
        filtered = [
            record for record in filtered
            if record["Component"] in selected_components
        ]

    if selected_statuses:
        filtered = [
            record for record in filtered
            if record["Status"].lower() in {
                status.lower() for status in selected_statuses
            }
        ]

    terms = [
        term.lower()
        for term in query.split()
        if term.strip()
    ]

    if terms:
        filtered = [
            record for record in filtered
            if all(
                term in record["_search"]
                or term in record["Summary"].lower()
                or term in record["Record ID"].lower()
                for term in terms
            )
        ]

    return filtered


# ------------------------------------------------------------
# 7. RELATIONSHIP DETAIL DISPLAY
# ------------------------------------------------------------

def render_record_details(record):
    st.write("**Component:**", record["Component"])
    st.write("**Record type:**", record["Record Type"])
    st.write("**Record ID:**", record["Record ID"])
    st.write("**Review status:**", record["Status"])

    st.divider()
    st.write("**Stored database fields**")

    details = record["_details"]

    for key, value in details.items():
        if value is None or str(value).strip() == "":
            continue

        st.write(f"**{key}:** {value}")


# ------------------------------------------------------------
# 8. MAIN STREAMLIT INTERFACE
# ------------------------------------------------------------

def render_evidence_relationship_intelligence():

    st.header(
        "🧠 Section 27: Evidence Relationship Intelligence"
    )

    st.write(
        "Explore persisted evidence relationships, reconstructed "
        "events, cross-camera candidate links and investigator "
        "review history from the project's existing databases."
    )

    st.info(
        "This dashboard reads existing SQLite records in read-only "
        "mode. It does not create, approve or modify evidence links."
    )

    records, database_report = collect_relationship_records()

    # --------------------------------------------------------
    # DATABASE STATUS
    # --------------------------------------------------------

    st.subheader("Connected Investigation Databases")

    st.dataframe(
        database_report,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    st.divider()
    st.subheader("Investigation Relationship Overview")

    total = len(records)

    graph_count = sum(
        r["Record Type"] == "Evidence Graph Relationships"
        for r in records
    )

    event_count = sum(
        r["Record Type"] == "Reconstructed Events"
        for r in records
    )

    camera_count = sum(
        r["Record Type"] == "Cross-Camera Relationships"
        for r in records
    )

    review_count = sum(
        r["Record Type"] == "Review History"
        for r in records
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total records", total)
    col2.metric("Graph links", graph_count)
    col3.metric("Events", event_count)
    col4.metric("Camera links", camera_count)
    col5.metric("Review history", review_count)

    if not records:
        st.warning(
            "No persisted investigation records were found. "
            "Check that the databases exist and contain records."
        )
        return

    # --------------------------------------------------------
    # SEARCH INTERFACE
    # --------------------------------------------------------

    st.divider()
    st.subheader("Search Investigation Relationships")

    query = st.text_input(
        "Search by event ID, object, source, observation ID or relationship",
        placeholder="Example: backpack, Camera 1, EVT-123...",
        key="relationship_intelligence_query",
    )

    components = sorted({
        record["Component"] for record in records
    })

    statuses = sorted({
        record["Status"] for record in records
    })

    selected_components = st.multiselect(
        "Investigation components",
        options=components,
        default=components,
        key="relationship_component_filter",
    )

    selected_statuses = st.multiselect(
        "Review status",
        options=statuses,
        default=statuses,
        key="relationship_status_filter",
    )

    filtered = filter_relationship_records(
        records,
        query=query,
        selected_components=selected_components,
        selected_statuses=selected_statuses,
    )

    st.write(f"**Matching records: {len(filtered)}**")

    if not filtered:
        st.warning("No matching investigation records found.")
        return

    # --------------------------------------------------------
    # RESULTS TABLE
    # --------------------------------------------------------

    overview = pd.DataFrame([
        {
            "Component": record["Component"],
            "Record Type": record["Record Type"],
            "Record ID": record["Record ID"],
            "Status": record["Status"],
            "Summary": record["Summary"],
        }
        for record in filtered
    ])

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # RECORD INSPECTION
    # --------------------------------------------------------

    st.divider()
    st.subheader("Inspect Relationship Records")

    for index, record in enumerate(filtered):
        heading = (
            f"{record['Record Type']} | "
            f"{record['Record ID']} | "
            f"{record['Status']}"
        )

        with st.expander(heading):
            render_record_details(record)

    # --------------------------------------------------------
    # INVESTIGATION SAFEGUARD
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "Interpretation safeguard: a stored candidate relationship "
        "is not proof of physical identity, causation or a verified "
        "incident. Review decisions and original evidence must be "
        "considered separately."
    )
