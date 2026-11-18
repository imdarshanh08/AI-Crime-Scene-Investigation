
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
from datetime import datetime

import pandas as pd


# ------------------------------------------------------------
# 1. DATABASE CONFIGURATION
# ------------------------------------------------------------

from core.config import CROSS_MODAL_DB_PATH, OUTPUT_DIR
from storage.sqlite_utils import connect as sqlite_connect

GRAPH_DB_PATH = CROSS_MODAL_DB_PATH

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

    return sqlite_connect(GRAPH_DB_PATH, timeout=15)


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
