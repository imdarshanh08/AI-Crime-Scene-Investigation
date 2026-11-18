# ============================================================
# SECTION 26: EVIDENCE SEARCH & NATURAL LANGUAGE QUERY ENGINE
# ============================================================

import re

import pandas as pd


# ------------------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------------------

SEARCH_COLUMNS = [
    "Observation ID",
    "Modality",
    "Source",
    "Source ID",
    "Camera ID",
    "Timestamp",
    "Timestamp (seconds)",
    "Segment End (seconds)",
    "Object",
    "Track ID",
    "Raw Track ID",
    "Confidence",
    "Observation",
    "Original Transcript",
    "English Translation",
    "Review Status",
    "Review Notes",
    "Evidence Path",
]

STOP_WORDS = {
    "show", "find", "search", "everything", "related", "about",
    "what", "happened", "between", "from", "to", "the", "a",
    "an", "me", "all", "evidence", "records", "observations",
    "please", "give", "list", "display", "during", "in", "on",
    "at", "and", "or", "is", "was", "were", "with", "of",
    "this", "that", "there", "which", "who", "where",
    "detections", "detection", "camera", "cctv", "footage", "video",
    "photo", "audio", "transcripts", "transcript", "recordings",
    "recording", "images", "image", "sound", "seen", "appeared"
}

MODALITY_ALIASES = {
    "video": "CCTV",
    "camera": "CCTV",
    "cctv": "CCTV",
    "image": "Photo",
    "picture": "Photo",
    "photo": "Photo",
    "audio": "Audio",
    "sound": "Audio",
    "speech": "Audio",
    "voice": "Audio",
}


# ------------------------------------------------------------
# 2. DATA NORMALISATION
# ------------------------------------------------------------

def _normalise_observations(observations):
    if observations is None:
        return pd.DataFrame(columns=SEARCH_COLUMNS)

    df = observations.copy()

    for column in SEARCH_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df.copy()

    for column in [
        "Observation ID", "Modality", "Source", "Source ID",
        "Timestamp", "Object", "Track ID", "Raw Track ID",
        "Observation", "Original Transcript",
        "English Translation", "Review Status",
        "Review Notes", "Evidence Path",
    ]:
        df[column] = df[column].fillna("").astype(str)

    df["Timestamp (seconds)"] = pd.to_numeric(
        df["Timestamp (seconds)"], errors="coerce"
    )

    df["Segment End (seconds)"] = pd.to_numeric(
        df["Segment End (seconds)"], errors="coerce"
    )

    df["Confidence"] = pd.to_numeric(
        df["Confidence"], errors="coerce"
    )

    return df


# ------------------------------------------------------------
# 3. TIME PARSING
# ------------------------------------------------------------

def _time_to_seconds(value):
    """
    Converts:
    10:30       -> 630 seconds (MM:SS)
    01:10:30    -> 4230 seconds (HH:MM:SS)
    90          -> 90 seconds
    """

    value = str(value).strip()

    if not value:
        return None

    try:
        if ":" not in value:
            return float(value)

        parts = [float(part) for part in value.split(":")]

        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds

        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds

    except (ValueError, TypeError):
        return None

    return None


def _extract_time_range(query):
    """
    Recognises queries such as:
    between 10:30 and 10:45
    from 00:02:10 to 00:05:00
    """

    patterns = [
        r"(?:between|from)\s+(\d{1,2}:\d{2}(?::\d{2})?)"
        r"\s+(?:and|to)\s+(\d{1,2}:\d{2}(?::\d{2})?)"
    ]

    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)

        if match:
            start = _time_to_seconds(match.group(1))
            end = _time_to_seconds(match.group(2))

            if start is not None and end is not None:
                return min(start, end), max(start, end)

    return None


# ------------------------------------------------------------
# 4. QUERY INTERPRETATION
# ------------------------------------------------------------

def _interpret_query(query):
    lowered = query.lower()

    interpreted = {
        "modality": None,
        "time_range": _extract_time_range(lowered),
        "terms": [],
    }

    for alias, modality in MODALITY_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", lowered):
            interpreted["modality"] = modality
            break

    cleaned = re.sub(
        r"(?:between|from)\s+\d{1,2}:\d{2}(?::\d{2})?"
        r"\s+(?:and|to)\s+\d{1,2}:\d{2}(?::\d{2})?",
        " ",
        lowered,
        flags=re.IGNORECASE,
    )

    words = re.findall(r"[a-zA-Z0-9_-]+", cleaned)

    interpreted["terms"] = [
        word for word in words
        if word not in STOP_WORDS and len(word) > 1
    ]

    return interpreted


# ------------------------------------------------------------
# 5. SEARCH ENGINE
# ------------------------------------------------------------

def search_evidence(observations, query):
    df = _normalise_observations(observations)

    if df.empty:
        return df, _interpret_query(query)

    interpretation = _interpret_query(query)
    results = df.copy()

    # Modality filter
    if interpretation["modality"]:
        results = results[
            results["Modality"].str.lower()
            == interpretation["modality"].lower()
        ]

    # Time range filter.
    # Timestamps are evaluated relative to each source video.
    time_range = interpretation["time_range"]

    if time_range:
        start_seconds, end_seconds = time_range

        results = results[
            results["Timestamp (seconds)"].notna()
            & (
                results["Timestamp (seconds)"].between(
                    start_seconds, end_seconds
                )
                |
                results["Segment End (seconds)"].between(
                    start_seconds, end_seconds
                )
            )
        ]

    # Text matching
    searchable_columns = [
        "Observation ID",
        "Camera ID",
        "Source",
        "Source ID",
        "Timestamp",
        "Object",
        "Track ID",
        "Raw Track ID",
        "Observation",
        "Original Transcript",
        "English Translation",
        "Review Status",
        "Review Notes",
    ]

    if interpretation["terms"]:
        searchable_text = (
            results[searchable_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.lower()
        )

        # Require every meaningful search term to be present.
        mask = pd.Series(True, index=results.index)

        for term in interpretation["terms"]:
            mask &= searchable_text.str.contains(
                re.escape(term), regex=True, na=False
            )

        results = results[mask]

    results = results.sort_values(
        by=["Source", "Timestamp (seconds)"],
        na_position="last",
    )

    return results.reset_index(drop=True), interpretation
