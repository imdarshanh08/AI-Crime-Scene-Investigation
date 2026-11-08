
# ============================================================
# SECTION 28: CROSS-MODAL EVIDENCE CORRELATION ENGINE
# ============================================================
#
# Extends the existing evidence graph database.
# Does not create a separate relationship database.
#
# Supported relationships:
#   1. CCTV_AUDIO_MENTION
#   2. CCTV_PHOTO_CONTEXT
#   3. AUDIO_PHOTO_MENTION
#
# All relationships are AI proposals requiring human review.
# No cross-file clock synchronization is assumed.
# ============================================================

import hashlib
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 1. DATABASE CONFIGURATION
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
GRAPH_DB_PATH = OUTPUT_DIR / "evidence_graph_relationships.db"

RELATIONSHIP_TYPES = [
    "CCTV_AUDIO_MENTION",
    "CCTV_PHOTO_CONTEXT",
    "AUDIO_PHOTO_MENTION",
]

REVIEW_STATUSES = [
    "Pending",
    "Approved",
    "Rejected",
    "Flagged",
]

MAX_PROPOSALS_PER_RUN = 500


# ------------------------------------------------------------
# 2. OBJECT VOCABULARY
# ------------------------------------------------------------

OBJECT_ALIASES = {
    "backpack": [
        "backpack",
        "back pack",
        "rucksack",
        "school bag",
    ],
    "handbag": [
        "handbag",
        "purse",
        "hand bag",
    ],
    "suitcase": [
        "suitcase",
        "luggage",
        "travel bag",
    ],
    "bottle": [
        "bottle",
        "water bottle",
    ],
    "cell phone": [
        "cell phone",
        "cellphone",
        "mobile phone",
        "mobile",
        "phone",
    ],
    "laptop": [
        "laptop",
        "notebook computer",
    ],
    "keyboard": [
        "keyboard",
    ],
    "mouse": [
        "computer mouse",
    ],
    "umbrella": [
        "umbrella",
    ],
    "book": [
        "book",
        "notebook",
    ],
    "chair": [
        "chair",
    ],
    "suit": [
        "suit",
    ],
    "person": [
        "person",
        "individual",
        "someone",
        "somebody",
        "man",
        "woman",
    ],
    "bicycle": [
        "bicycle",
        "bike",
        "cycle",
    ],
    "motorcycle": [
        "motorcycle",
        "motorbike",
    ],
    "car": [
        "car",
        "automobile",
        "vehicle",
    ],
    "bus": [
        "bus",
    ],
    "truck": [
        "truck",
        "lorry",
    ],
    "refrigerator": [
        "refrigerator",
        "fridge",
    ],
    "cup": [
        "cup",
    ],
    "bowl": [
        "bowl",
    ],
    "remote": [
        "remote",
        "remote control",
    ],
    "tv": [
        "tv",
        "television",
    ],
    "clock": [
        "clock",
    ],
    "scissors": [
        "scissors",
    ],
    "vase": [
        "vase",
    ],
    "sports ball": [
        "sports ball",
        "ball",
    ],
    "dog": [
        "dog",
    ],
    "cat": [
        "cat",
    ],
}


# ------------------------------------------------------------
# 3. DATABASE CONNECTION
# ------------------------------------------------------------

def get_connection():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        str(GRAPH_DB_PATH),
        timeout=15,
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialise_correlation_database():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS graph_relationships (
                relationship_id TEXT PRIMARY KEY,
                source_observation TEXT NOT NULL,
                target_observation TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                explanation TEXT,
                confidence REAL DEFAULT 0,
                status TEXT DEFAULT 'Pending',
                origin TEXT DEFAULT 'AI Proposed',
                reviewer_notes TEXT DEFAULT '',
                reviewed_at TEXT DEFAULT ''
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS graph_review_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                relationship_id TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                reviewer_notes TEXT,
                reviewed_at TEXT NOT NULL
            )
        """)


# ------------------------------------------------------------
# 4. TEXT NORMALISATION
# ------------------------------------------------------------

def normalise_text(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def contains_phrase(text, phrase):

    text = normalise_text(text)
    phrase = normalise_text(phrase)

    if not text or not phrase:
        return False

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )

    return re.search(pattern, text) is not None


def get_object_aliases(object_name):

    category = normalise_text(object_name)

    if not category:
        return []

    aliases = OBJECT_ALIASES.get(
        category,
        [category],
    )

    return list(dict.fromkeys(
        [category] + aliases
    ))


def transcript_mentions_object(
    transcript,
    object_name,
):

    aliases = get_object_aliases(object_name)

    matched = []

    for alias in aliases:

        if contains_phrase(transcript, alias):
            matched.append(alias)

    return matched


# ------------------------------------------------------------
# 5. OBSERVATION NORMALISATION
# ------------------------------------------------------------

def safe_value(row, key, default=""):

    value = row.get(key, default)

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    return value


def normalise_observations(observations):

    if observations is None:
        return pd.DataFrame()

    df = observations.copy()

    required_columns = [
        "Observation ID",
        "Modality",
        "Source",
        "Source ID",
        "Timestamp",
        "Timestamp (seconds)",
        "Object",
        "Observation",
        "Original Transcript",
        "English Translation",
        "Evidence Path",
    ]

    for column in required_columns:

        if column not in df.columns:
            df[column] = ""

    df["Observation ID"] = (
        df["Observation ID"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Modality"] = (
        df["Modality"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["Object"] = (
        df["Object"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["Observation ID"] != ""
    ].copy()

    df = df.drop_duplicates(
        subset=["Observation ID"],
        keep="last",
    )

    return df


# ------------------------------------------------------------
# 6. TRANSCRIPT EXTRACTION
# ------------------------------------------------------------

def get_transcript_text(row):

    original = normalise_text(
        safe_value(row, "Original Transcript")
    )

    translated = normalise_text(
        safe_value(row, "English Translation")
    )

    observation = normalise_text(
        safe_value(row, "Observation")
    )

    # Preserve both original and translated text.
    # Translation matching is not proof of the original speech.

    parts = []

    if original:
        parts.append(original)

    if translated and translated != original:
        parts.append(translated)

    if not parts and observation:
        parts.append(observation)

    return " ".join(parts)


# ------------------------------------------------------------
# 7. RELATIONSHIP ID
# ------------------------------------------------------------

def make_relationship_id(
    source,
    target,
    relationship_type,
):

    raw = (
        str(source)
        + "|"
        + str(target)
        + "|"
        + str(relationship_type)
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:20]


# ------------------------------------------------------------
# 8. CANDIDATE GENERATION
# ------------------------------------------------------------

def create_candidate(
    source_row,
    target_row,
    relationship_type,
    confidence,
    explanation,
):

    source_id = str(
        safe_value(source_row, "Observation ID")
    )

    target_id = str(
        safe_value(target_row, "Observation ID")
    )

    if not source_id or not target_id:
        return None

    if source_id == target_id:
        return None

    return {
        "source_observation": source_id,
        "target_observation": target_id,
        "relationship_type": relationship_type,
        "confidence": round(
            max(0.0, min(1.0, float(confidence))),
            3,
        ),
        "explanation": explanation,
        "status": "Pending",
        "origin": "AI Cross-Modal Correlation",
    }


def generate_cross_modal_candidates(
    observations,
    include_photo_context=True,
):

    df = normalise_observations(observations)

    candidates = []

    if df.empty:
        return candidates

    cctv = df[
        df["Modality"] == "cctv"
    ]

    photos = df[
        df["Modality"] == "photo"
    ]

    audio = df[
        df["Modality"] == "audio"
    ]

    # --------------------------------------------------------
    # A. CCTV TO AUDIO
    # --------------------------------------------------------

    for _, cctv_row in cctv.iterrows():

        object_name = normalise_text(
            safe_value(cctv_row, "Object")
        )

        if not object_name:
            continue

        if object_name == "speech":
            continue

        for _, audio_row in audio.iterrows():

            transcript = get_transcript_text(
                audio_row
            )

            if not transcript:
                continue

            matches = transcript_mentions_object(
                transcript,
                object_name,
            )

            if not matches:
                continue

            source_name = safe_value(
                cctv_row,
                "Source",
                "CCTV source",
            )

            audio_source = safe_value(
                audio_row,
                "Source",
                "Audio source",
            )

            matched_text = ", ".join(matches)

            explanation = (
                f"The CCTV observation records the object "
                f"category '{object_name}' in source "
                f"'{source_name}'. The audio transcript from "
                f"'{audio_source}' explicitly contains the "
                f"matching term or alias '{matched_text}'. "
                f"This is a semantic mention relationship, "
                f"not proof that the speech refers to the "
                f"particular detected object. No shared clock "
                f"or physical identity is assumed."
            )

            confidence = 0.65

            if len(matches) > 1:
                confidence = 0.72

            candidate = create_candidate(
                cctv_row,
                audio_row,
                "CCTV_AUDIO_MENTION",
                confidence,
                explanation,
            )

            if candidate:
                candidates.append(candidate)

    # --------------------------------------------------------
    # B. CCTV TO PHOTO
    # --------------------------------------------------------

    if include_photo_context:

        for _, cctv_row in cctv.iterrows():

            cctv_object = normalise_text(
                safe_value(cctv_row, "Object")
            )

            if not cctv_object:
                continue

            if cctv_object == "speech":
                continue

            for _, photo_row in photos.iterrows():

                photo_object = normalise_text(
                    safe_value(photo_row, "Object")
                )

                if not photo_object:
                    continue

                if photo_object != cctv_object:
                    continue

                cctv_source = safe_value(
                    cctv_row,
                    "Source",
                    "CCTV source",
                )

                photo_source = safe_value(
                    photo_row,
                    "Source",
                    "Photo source",
                )

                explanation = (
                    f"A CCTV observation and a photograph "
                    f"both contain the detected object category "
                    f"'{cctv_object}'. CCTV source: "
                    f"'{cctv_source}'. Photo source: "
                    f"'{photo_source}'. This is a same-category "
                    f"contextual relationship only. It does "
                    f"not establish that both observations show "
                    f"the same physical object, person or event. "
                    f"The photograph has no verified capture "
                    f"time in the available observation data."
                )

                candidate = create_candidate(
                    cctv_row,
                    photo_row,
                    "CCTV_PHOTO_CONTEXT",
                    0.35,
                    explanation,
                )

                if candidate:
                    candidates.append(candidate)

    # --------------------------------------------------------
    # C. AUDIO TO PHOTO
    # --------------------------------------------------------

    for _, audio_row in audio.iterrows():

        transcript = get_transcript_text(
            audio_row
        )

        if not transcript:
            continue

        for _, photo_row in photos.iterrows():

            object_name = normalise_text(
                safe_value(photo_row, "Object")
            )

            if not object_name:
                continue

            if object_name == "speech":
                continue

            matches = transcript_mentions_object(
                transcript,
                object_name,
            )

            if not matches:
                continue

            audio_source = safe_value(
                audio_row,
                "Source",
                "Audio source",
            )

            photo_source = safe_value(
                photo_row,
                "Source",
                "Photo source",
            )

            explanation = (
                f"The audio transcript from '{audio_source}' "
                f"explicitly mentions '{object_name}' using "
                f"the matching term or alias "
                f"'{', '.join(matches)}'. A photograph from "
                f"'{photo_source}' contains an object detection "
                f"of the same category. This is a semantic "
                f"cross-modal association, not proof that the "
                f"speech refers to the photographed object. "
                f"No capture-time alignment is assumed."
            )

            candidate = create_candidate(
                audio_row,
                photo_row,
                "AUDIO_PHOTO_MENTION",
                0.60,
                explanation,
            )

            if candidate:
                candidates.append(candidate)

    # --------------------------------------------------------
    # D. DEDUPLICATION AND LIMIT
    # --------------------------------------------------------

    unique = {}

    for candidate in candidates:

        relationship_id = make_relationship_id(
            candidate["source_observation"],
            candidate["target_observation"],
            candidate["relationship_type"],
        )

        unique[relationship_id] = candidate

    return list(unique.values())[
        :MAX_PROPOSALS_PER_RUN
    ]


# ------------------------------------------------------------
# 9. SAVE PROPOSALS TO EXISTING EVIDENCE GRAPH
# ------------------------------------------------------------

def save_cross_modal_candidates(candidates):

    inserted = 0

    with get_connection() as connection:

        for candidate in candidates:

            relationship_id = make_relationship_id(
                candidate["source_observation"],
                candidate["target_observation"],
                candidate["relationship_type"],
            )

            cursor = connection.execute("""
                INSERT OR IGNORE INTO graph_relationships (
                    relationship_id,
                    source_observation,
                    target_observation,
                    relationship_type,
                    explanation,
                    confidence,
                    status,
                    origin,
                    reviewer_notes,
                    reviewed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'Pending',
                        'AI Cross-Modal Correlation', '', '')
            """, (
                relationship_id,
                candidate["source_observation"],
                candidate["target_observation"],
                candidate["relationship_type"],
                candidate["explanation"],
                candidate["confidence"],
            ))

            inserted += cursor.rowcount

    return inserted


# ------------------------------------------------------------
# 10. LOAD CROSS-MODAL RELATIONSHIPS
# ------------------------------------------------------------

def get_cross_modal_relationships():

    placeholders = ",".join(
        "?" for _ in RELATIONSHIP_TYPES
    )

    with get_connection() as connection:

        rows = connection.execute(
            f"""
            SELECT *
            FROM graph_relationships
            WHERE relationship_type IN ({placeholders})
            ORDER BY
                CASE status
                    WHEN 'Pending' THEN 0
                    WHEN 'Flagged' THEN 1
                    WHEN 'Approved' THEN 2
                    ELSE 3
                END,
                relationship_id
            """,
            RELATIONSHIP_TYPES,
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ------------------------------------------------------------
# 11. INVESTIGATOR REVIEW
# ------------------------------------------------------------

def review_cross_modal_relationship(
    relationship_id,
    decision,
    reviewer_notes="",
):

    if decision not in REVIEW_STATUSES:
        raise ValueError("Invalid review decision.")

    reviewed_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with get_connection() as connection:

        record = connection.execute("""
            SELECT status
            FROM graph_relationships
            WHERE relationship_id = ?
              AND relationship_type IN (?, ?, ?)
        """, (
            relationship_id,
            *RELATIONSHIP_TYPES,
        )).fetchone()

        if not record:
            raise ValueError(
                "Cross-modal relationship not found."
            )

        previous_status = record["status"]

        if previous_status == decision:
            return False

        connection.execute("""
            UPDATE graph_relationships
            SET status = ?,
                reviewer_notes = ?,
                reviewed_at = ?
            WHERE relationship_id = ?
        """, (
            decision,
            str(reviewer_notes),
            reviewed_at,
            relationship_id,
        ))

        connection.execute("""
            INSERT INTO graph_review_history (
                relationship_id,
                previous_status,
                new_status,
                reviewer_notes,
                reviewed_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            relationship_id,
            previous_status,
            decision,
            str(reviewer_notes),
            reviewed_at,
        ))

    return True


def get_cross_modal_review_history(relationship_id):

    with get_connection() as connection:

        rows = connection.execute("""
            SELECT *
            FROM graph_review_history
            WHERE relationship_id = ?
            ORDER BY id DESC
        """, (relationship_id,)).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ------------------------------------------------------------
# 12. STREAMLIT INTERFACE
# ------------------------------------------------------------

def render_cross_modal_correlation(observations):

    initialise_correlation_database()

    st.header(
        "🔗 Section 28: Cross-Modal Evidence Correlation"
    )

    st.write(
        "Find explainable relationships between CCTV "
        "detections, audio transcripts and photographs."
    )

    st.warning(
        "These are AI-generated candidate relationships, "
        "not verified facts. Matching categories or words "
        "does not establish physical identity or prove "
        "that two observations belong to the same incident."
    )

    df = normalise_observations(observations)

    if df.empty:
        st.info(
            "No observations are available. Upload and analyse "
            "CCTV, audio or photographs first."
        )
        return

    counts = (
        df["Modality"]
        .value_counts()
        .to_dict()
    )

    cctv_count = counts.get("cctv", 0)
    audio_count = counts.get("audio", 0)
    photo_count = counts.get("photo", 0)

    col1, col2, col3 = st.columns(3)

    col1.metric("CCTV observations", cctv_count)
    col2.metric("Audio observations", audio_count)
    col3.metric("Photo observations", photo_count)

    st.divider()

    # --------------------------------------------------------
    # GENERATE CANDIDATES
    # --------------------------------------------------------

    st.subheader("Generate Cross-Modal Proposals")

    st.write(
        "The engine checks transcript mentions and matching "
        "object categories. It does not assume that separate "
        "recordings are synchronized."
    )

    include_photo_context = st.checkbox(
        "Include same-category CCTV and photo context proposals",
        value=True,
        help=(
            "These proposals have low confidence because "
            "same-category detections do not prove object identity."
        ),
    )

    if st.button(
        "Generate Cross-Modal Relationships",
        type="primary",
        key="generate_cross_modal_relationships",
    ):

        with st.spinner(
            "Analysing cross-modal evidence..."
        ):

            candidates = generate_cross_modal_candidates(
                df,
                include_photo_context=include_photo_context,
            )

            inserted = save_cross_modal_candidates(
                candidates
            )

        st.success(
            f"Generated {len(candidates)} candidate(s). "
            f"Added {inserted} new relationship(s) to the "
            f"existing Evidence Graph."
        )

        if len(candidates) >= MAX_PROPOSALS_PER_RUN:
            st.warning(
                "The proposal limit was reached. Review the "
                "current results before generating more."
            )

    # --------------------------------------------------------
    # RELATIONSHIP DASHBOARD
    # --------------------------------------------------------

    st.divider()
    st.subheader("Cross-Modal Relationship Dashboard")

    relationships = get_cross_modal_relationships()

    if not relationships:
        st.info(
            "No cross-modal relationships have been generated yet."
        )
        return

    relationship_df = pd.DataFrame(
        relationships
    )

    total = len(relationship_df)

    pending = int(
        (relationship_df["status"] == "Pending").sum()
    )

    approved = int(
        (relationship_df["status"] == "Approved").sum()
    )

    rejected = int(
        (relationship_df["status"] == "Rejected").sum()
    )

    flagged = int(
        (relationship_df["status"] == "Flagged").sum()
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Total", total)
    col2.metric("Pending", pending)
    col3.metric("Approved", approved)
    col4.metric("Rejected", rejected)
    col5.metric("Flagged", flagged)

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    selected_types = st.multiselect(
        "Filter relationship types",
        RELATIONSHIP_TYPES,
        default=RELATIONSHIP_TYPES,
        key="cross_modal_type_filter",
    )

    selected_statuses = st.multiselect(
        "Filter review status",
        REVIEW_STATUSES,
        default=REVIEW_STATUSES,
        key="cross_modal_status_filter",
    )

    visible = relationship_df[
        relationship_df["relationship_type"].isin(
            selected_types
        )
        & relationship_df["status"].isin(
            selected_statuses
        )
    ].copy()

    display_columns = [
        "relationship_id",
        "relationship_type",
        "source_observation",
        "target_observation",
        "confidence",
        "status",
        "origin",
    ]

    st.dataframe(
        visible[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    if visible.empty:
        st.info(
            "No relationships match the selected filters."
        )
        return

    # --------------------------------------------------------
    # RELATIONSHIP DETAILS
    # --------------------------------------------------------

    st.divider()
    st.subheader("Inspect Relationship")

    relationship_options = {}

    for _, row in visible.iterrows():

        label = (
            str(row["relationship_type"])
            + " | "
            + str(row["source_observation"])
            + " → "
            + str(row["target_observation"])
            + " | "
            + str(row["status"])
        )

        relationship_options[label] = str(
            row["relationship_id"]
        )

    selected_label = st.selectbox(
        "Select a relationship",
        list(relationship_options.keys()),
        key="cross_modal_selected_relationship",
    )

    selected_id = relationship_options[
        selected_label
    ]

    selected = relationship_df[
        relationship_df["relationship_id"].astype(str)
        == selected_id
    ].iloc[0]

    st.write(
        "**Relationship type:**",
        selected["relationship_type"],
    )

    st.write(
        "**Source observation:**",
        selected["source_observation"],
    )

    st.write(
        "**Target observation:**",
        selected["target_observation"],
    )

    st.write(
        "**Proposed confidence:**",
        f"{float(selected['confidence']):.3f}",
    )

    st.write(
        "**Current status:**",
        selected["status"],
    )

    st.markdown("**Explanation**")

    st.info(
        str(selected["explanation"])
    )

    # --------------------------------------------------------
    # REVIEW CONTROLS
    # --------------------------------------------------------

    st.divider()
    st.subheader("Investigator Verification")

    reviewer = st.text_input(
        "Investigator / reviewer",
        key="cross_modal_reviewer",
    )

    decision = st.selectbox(
        "Review decision",
        REVIEW_STATUSES,
        index=REVIEW_STATUSES.index(
            str(selected["status"])
        ),
        key="cross_modal_decision",
    )

    notes = st.text_area(
        "Investigator notes",
        key="cross_modal_review_notes",
    )

    if st.button(
        "Save Investigator Decision",
        type="primary",
        key="save_cross_modal_review",
    ):

        if not reviewer.strip():
            st.error(
                "Enter the investigator's name before saving."
            )

        else:

            complete_notes = (
                f"Reviewer: {reviewer.strip()}\n"
                + notes.strip()
            )

            try:

                changed = review_cross_modal_relationship(
                    selected_id,
                    decision,
                    complete_notes,
                )

                if changed:
                    st.success(
                        "Decision saved in the existing "
                        "Evidence Graph review history."
                    )
                else:
                    st.info(
                        "The relationship already has this status."
                    )

            except (ValueError, sqlite3.Error) as error:
                st.error(str(error))

    # --------------------------------------------------------
    # REVIEW HISTORY
    # --------------------------------------------------------

    st.divider()
    st.subheader("Relationship Review History")

    history = get_cross_modal_review_history(
        selected_id
    )

    if history:

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df[
                [
                    "previous_status",
                    "new_status",
                    "reviewer_notes",
                    "reviewed_at",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info(
            "No investigator review decisions recorded yet."
        )

    st.caption(
        "This module reuses the existing evidence graph. "
        "Review decisions are recorded in graph_review_history. "
        "Reviewer names are manually entered and are not "
        "authenticated identities."
    )
