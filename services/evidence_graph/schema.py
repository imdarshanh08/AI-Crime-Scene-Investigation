"""Vocabulary of the evidence graph: node types, relationship types and their meaning.

Keeping the definitions in one place lets the engine, the visualisation, the report
and the tests agree on exactly what each relationship asserts.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Node types
# ---------------------------------------------------------------------------
CAMERA = "Camera"
LOCATION = "Location"
VIDEO = "Video Recording"
AUDIO = "Audio Recording"
IMAGE = "Image"
PERSON = "Person Track"          # anonymous ByteTrack ID, never an identity
VEHICLE = "Vehicle"
BELONGING = "Personal Belonging"
OBJECT = "Object"
OBSERVATION = "Observation"
TRANSCRIPT = "Transcript Segment"
EVENT = "Event"

ENTITY_TYPES = (PERSON, VEHICLE, BELONGING, OBJECT)
SOURCE_TYPES = (VIDEO, AUDIO, IMAGE)
STRUCTURAL_TYPES = (CAMERA, LOCATION) + SOURCE_TYPES

NODE_TYPES = (CAMERA, LOCATION, VIDEO, AUDIO, IMAGE, PERSON, VEHICLE, BELONGING,
              OBJECT, OBSERVATION, TRANSCRIPT, EVENT)

# COCO class -> entity group
_VEHICLES = {"car", "motorcycle", "bicycle", "truck", "bus", "train", "boat", "airplane"}
_BELONGINGS = {"backpack", "handbag", "suitcase", "umbrella", "cell phone", "laptop", "tie",
               "wallet", "bottle", "book"}


def entity_type_for_class(object_class: str) -> str:
    cls = (object_class or "").strip().lower()
    if cls == "person":
        return PERSON
    if cls in _VEHICLES:
        return VEHICLE
    if cls in _BELONGINGS:
        return BELONGING
    return OBJECT


# ---------------------------------------------------------------------------
# Relationship types. Each entry: meaning shown in the UI and the report.
# ---------------------------------------------------------------------------
RELATIONS = {
    "CAPTURED_BY": "Recording was captured by this camera (evidence registry).",
    "LOCATED_NEAR": "Camera is installed at / near this investigator-entered location.",
    "DETECTED_IN": "Observation was produced from this recording, image or audio file.",
    "SAME_TRACK": "Observation belongs to this anonymous ByteTrack track (same source).",
    "APPEARED_AT": "Tracked entity was seen by this camera during the stated interval.",
    "APPEARED_BEFORE": "Source ended before target started on a comparable clock (inverse: APPEARED_AFTER).",
    "ASSOCIATED_WITH_EVENT": "Observation or track is a member of this candidate event.",
    "TEMPORALLY_OVERLAPS": "Intervals overlap on a comparable clock (shared source or verified anchors).",
    "CROSSED_CAMERA": "Same object class left one camera and appeared on another within the hand-off window. Candidate only.",
    "POSSIBLY_SAME_OBJECT": "Proposed association between appearances. Not proof of identity.",
    "HEARD_DURING": "Speech segment overlaps the visual evidence in time on a comparable clock.",
    "SUPPORTS": "Evidence that corroborates the target (e.g. speech naming an object visible at that moment).",
    "CONTRADICTS": "Evidence that conflicts with the target in time, location, class or review decision.",
    "DISTINCT_ENTITIES": "Investigator ruled these out as the same entity.",
}

# Relationships that carry an evidential claim (used for retrieval and reports).
EVIDENTIAL_RELATIONS = {
    "SAME_TRACK", "APPEARED_AT", "APPEARED_BEFORE", "ASSOCIATED_WITH_EVENT", "TEMPORALLY_OVERLAPS",
    "CROSSED_CAMERA", "POSSIBLY_SAME_OBJECT", "HEARD_DURING", "SUPPORTS", "CONTRADICTS",
    "DISTINCT_ENTITIES",
}

ORIGIN_RULE = "Derived rule"          # deterministic rule over stored data
ORIGIN_AI = "AI proposal"             # stored model / correlation proposal awaiting review
ORIGIN_INVESTIGATOR = "Investigator"  # recorded by a human reviewer
ORIGIN_REGISTRY = "Evidence registry"  # provenance facts (file came from camera, etc.)

# ---------------------------------------------------------------------------
# Visual style (restrained palette; shape encodes type, colour encodes family)
# ---------------------------------------------------------------------------
NODE_STYLE = {
    CAMERA: {"shape": "box", "color": "#64748b"},
    LOCATION: {"shape": "box", "color": "#475569"},
    VIDEO: {"shape": "database", "color": "#3b82f6"},
    AUDIO: {"shape": "database", "color": "#8b5cf6"},
    IMAGE: {"shape": "database", "color": "#0ea5e9"},
    PERSON: {"shape": "dot", "color": "#f59e0b"},
    VEHICLE: {"shape": "dot", "color": "#14b8a6"},
    BELONGING: {"shape": "dot", "color": "#ef4444"},
    OBJECT: {"shape": "dot", "color": "#a3a3a3"},
    OBSERVATION: {"shape": "dot", "color": "#94a3b8"},
    TRANSCRIPT: {"shape": "triangle", "color": "#a78bfa"},
    EVENT: {"shape": "diamond", "color": "#22c55e"},
}

EDGE_STYLE = {
    "CAPTURED_BY": {"color": "#475569", "dashes": False},
    "LOCATED_NEAR": {"color": "#475569", "dashes": False},
    "DETECTED_IN": {"color": "#334155", "dashes": False},
    "SAME_TRACK": {"color": "#a16207", "dashes": False},
    "APPEARED_AT": {"color": "#b45309", "dashes": False},
    "APPEARED_BEFORE": {"color": "#64748b", "dashes": [2, 4]},
    "ASSOCIATED_WITH_EVENT": {"color": "#16a34a", "dashes": False},
    "TEMPORALLY_OVERLAPS": {"color": "#0891b2", "dashes": [6, 4]},
    "CROSSED_CAMERA": {"color": "#38bdf8", "dashes": [6, 4]},
    "POSSIBLY_SAME_OBJECT": {"color": "#eab308", "dashes": [6, 4]},
    "HEARD_DURING": {"color": "#a78bfa", "dashes": [4, 4]},
    "SUPPORTS": {"color": "#22c55e", "dashes": False},
    "CONTRADICTS": {"color": "#ef4444", "dashes": False},
    "DISTINCT_ENTITIES": {"color": "#f87171", "dashes": [2, 2]},
}


def short_track(track_id: str) -> str:
    """'CAM-02-T14-person' -> 'T14' (falls back to the raw ID)."""
    import re

    m = re.search(r"(?:^|-)(T\d+)(?:-|$)", track_id or "")
    return m.group(1) if m else (track_id or "")
