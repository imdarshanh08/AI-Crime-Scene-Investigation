"""Investigator question answering over the evidence graph.

A question is parsed into explicit filters (object classes, approximate colour,
cameras/locations, time window, modality, speech keywords, intent). Retrieval then
combines:

1. metadata filtering   - class, camera, location, modality, review status
2. temporal filtering   - source-relative or wall-clock windows (anchored sources)
3. keyword matching     - transcripts and observation descriptions (alias-aware)
4. graph expansion      - evidential relationships from the matched items

The answer is a chronological timeline in which every row cites the stored
evidence it came from. Nothing is inferred beyond what the graph contains, and
each row is labelled as investigator-verified, AI finding or rejected.

Semantic search here is vocabulary-based (curated aliases), not a neural
embedding model; the interpretation shown to the user makes that explicit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from services.correlation.cross_modal_correlation import OBJECT_ALIASES
from services.evidence_graph import analysis as A
from services.evidence_graph import schema as S
from services.evidence_graph.appearance import COLOUR_NAMES
from services.evidence_graph.engine import EvidenceGraph

STOP_WORDS = {
    "show", "me", "everything", "all", "related", "relating", "to", "the", "a", "an", "what", "happened",
    "between", "and", "from", "which", "cameras", "camera", "recorded", "this", "that", "object", "evidence",
    "is", "are", "was", "were", "connected", "with", "of", "in", "on", "at", "any", "there", "observations",
    "observation", "audio", "video", "sequence", "events", "event", "involving", "supports", "support",
    "possible", "incident", "conflicting", "conflict", "conflicts", "contradictions", "contradicting", "find",
    "list", "give", "seen", "did", "do", "does", "who", "where", "when", "how", "by", "for", "about", "please",
    "am", "pm", "after", "before", "during", "between", "tell", "speech", "said", "heard", "cctv", "photo",
    "photos", "image", "images", "clips", "footage", "time", "times", "it", "its", "be", "into", "out", "up",
}

TIME_RE = r"(\d{1,2}:\d{2}(?::\d{2})?)\s*(am|pm|a\.m\.|p\.m\.)?"
RANGE_RE = re.compile(rf"(?:between|from)\s+{TIME_RE}\s+(?:and|to|-)\s+{TIME_RE}", re.I)

RELATED_HINTS = ("related", "connected", "linked", "associated", "everything", "support", "involving")
CONFLICT_HINTS = ("conflict", "contradict", "inconsistent", "disagree")
CAMERA_HINTS = ("which camera", "what camera", "which cameras", "cameras recorded", "camera recorded")


@dataclass
class QueryPlan:
    question: str
    classes: Set[str] = field(default_factory=set)
    colours: Set[str] = field(default_factory=set)
    cameras: Set[str] = field(default_factory=set)
    locations: Set[str] = field(default_factory=set)
    modalities: Set[str] = field(default_factory=set)
    keywords: List[str] = field(default_factory=list)
    node_ids: Set[str] = field(default_factory=set)
    rel_window: Optional[Tuple[float, float]] = None
    abs_window: Optional[Tuple[dtime, dtime]] = None
    wants_related: bool = False
    wants_conflicts: bool = False
    wants_cameras: bool = False
    notes: List[str] = field(default_factory=list)

    def describe(self) -> List[str]:
        out = []
        if self.classes:
            out.append("Object classes: " + ", ".join(sorted(self.classes)))
        if self.colours:
            out.append("Approximate colour: " + ", ".join(sorted(self.colours)))
        if self.cameras:
            out.append("Cameras: " + ", ".join(sorted(self.cameras)))
        if self.locations:
            out.append("Locations: " + ", ".join(sorted(self.locations)))
        if self.modalities:
            out.append("Modalities: " + ", ".join(sorted(self.modalities)))
        if self.rel_window:
            out.append(f"Time window: {self.rel_window[0]:.0f}s to {self.rel_window[1]:.0f}s from the start of each recording")
        if self.abs_window:
            out.append(f"Wall-clock window: {self.abs_window[0].strftime('%H:%M:%S')} to {self.abs_window[1].strftime('%H:%M:%S')}")
        if self.keywords:
            out.append("Keywords (transcripts, notes, descriptions): " + ", ".join(self.keywords))
        if self.node_ids:
            out.append("Referenced records: " + ", ".join(sorted(self.node_ids)))
        if self.wants_related:
            out.append("Including evidence connected through the graph")
        if self.wants_conflicts:
            out.append("Checking for contradictions")
        return out or ["No specific filters recognised; showing all timed evidence"]


def _parse_clock(text: str, meridiem: Optional[str]) -> Optional[dtime]:
    parts = [int(p) for p in text.split(":")]
    h, m = parts[0], parts[1]
    s = parts[2] if len(parts) > 2 else 0
    if meridiem:
        mer = meridiem.lower().replace(".", "")
        if not 1 <= h <= 12:
            return None
        h = (h % 12) + (12 if mer == "pm" else 0)
    if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
        return None
    return dtime(h, m, s)


def _to_seconds(text: str) -> float:
    parts = [float(p) for p in text.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_question(question: str, eg: EvidenceGraph) -> QueryPlan:
    G = eg.graph
    q = question.strip()
    low = q.lower()
    plan = QueryPlan(question=q)

    # Time window
    m = RANGE_RE.search(low)
    if m:
        t1, mer1, t2, mer2 = m.group(1), m.group(2), m.group(3), m.group(4)
        if mer1 or mer2:
            mer1 = mer1 or mer2
            mer2 = mer2 or mer1
            a, b = _parse_clock(t1, mer1), _parse_clock(t2, mer2)
            if a and b:
                plan.abs_window = (min(a, b), max(a, b))
                if not eg.timebase.anchors:
                    plan.notes.append("A wall-clock time was given, but no recording has a verified start time, "
                                      "so no evidence can be placed on the clock. Set recording start times in "
                                      "Settings > Evidence sources.")
        else:
            a, b = _to_seconds(t1), _to_seconds(t2)
            plan.rel_window = (min(a, b), max(a, b))
            plan.notes.append("Times without AM/PM are read as minutes:seconds from the start of each recording. "
                              "Add AM/PM to search by wall-clock time.")
        low_wo_time = RANGE_RE.sub(" ", low)
    else:
        low_wo_time = low

    # Known vocabulary from the case itself
    case_classes = {d.get("object_class", "").lower() for _, d in G.nodes(data=True) if d.get("object_class")}
    alias_to_class: Dict[str, str] = {}
    for cls, aliases in OBJECT_ALIASES.items():
        for alias in aliases + [cls]:
            alias_to_class[alias.lower()] = cls
    for cls in case_classes:
        alias_to_class.setdefault(cls, cls)
    for alias in sorted(alias_to_class, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}s?\b", low_wo_time):
            plan.classes.add(alias_to_class[alias])
            low_wo_time = re.sub(rf"\b{re.escape(alias)}s?\b", " ", low_wo_time)

    for colour in COLOUR_NAMES + ("gray",):
        if re.search(rf"\b{colour}\b", low_wo_time):
            plan.colours.add("grey" if colour == "gray" else colour)
            low_wo_time = re.sub(rf"\b{colour}\b", " ", low_wo_time)
    if plan.colours:
        plan.notes.append("The detector does not output colour. Colour is estimated from the pixels of the best "
                          "detection crop and depends on lighting and camera settings; treat it as approximate.")

    cameras = {d.get("camera_id", "") for _, d in G.nodes(data=True) if d["type"] == S.CAMERA}
    for cam in cameras:
        if cam and re.search(rf"\b{re.escape(cam.lower())}\b", low_wo_time):
            plan.cameras.add(cam)
            low_wo_time = low_wo_time.replace(cam.lower(), " ")
    for _, d in G.nodes(data=True):
        if d["type"] == S.LOCATION and d.get("label", "").lower() in low_wo_time:
            plan.locations.add(d["label"])

    if re.search(r"\b(audio|speech|said|heard|voice|transcript)", low):
        plan.modalities.add("Audio")
    if re.search(r"\b(photo|photograph|image|picture)s?\b", low):
        plan.modalities.add("Photo")
    if re.search(r"\b(video|cctv|footage)\b", low) and plan.modalities:
        plan.modalities.add("CCTV")

    for token in re.findall(r"\b(evt-[0-9a-f]{12}|[0-9a-f]{12})\b", low):
        for prefix in ("evt:", "obs:"):
            cand = prefix + (token.upper() if token.startswith("evt-") else token)
            if cand in G:
                plan.node_ids.add(cand)

    plan.wants_related = any(h in low for h in RELATED_HINTS) or bool(plan.node_ids)
    plan.wants_conflicts = any(h in low for h in CONFLICT_HINTS)
    plan.wants_cameras = any(h in low for h in CAMERA_HINTS)

    words = re.findall(r"[a-z][a-z'-]{2,}", low_wo_time)
    plan.keywords = [w for w in words if w not in STOP_WORDS and not w.startswith("evt")][:6]
    return plan


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
QUERYABLE = set(S.ENTITY_TYPES) | {S.OBSERVATION, S.TRANSCRIPT, S.EVENT}


def _matches(G: nx.MultiDiGraph, n: str, d: Dict[str, Any], plan: QueryPlan) -> bool:
    # Tracked observations are represented by their track node to keep timelines readable.
    if d["type"] == S.OBSERVATION and d.get("modality") == "CCTV" and d.get("track_id", "") not in ("", "Untracked"):
        return False
    if plan.classes and d.get("object_class", "").lower() not in plan.classes:
        return False
    if plan.colours:
        colour = d.get("dominant_colour")
        if colour is None and d["type"] == S.EVENT:
            colour = next((G.nodes[u].get("dominant_colour") for u, _, e in G.in_edges(n, data=True)
                           if e["relation"] == "ASSOCIATED_WITH_EVENT" and u.startswith("trk:")), None)
        if colour not in plan.colours:
            return False
    if plan.cameras and d.get("camera_id") not in plan.cameras:
        return False
    if plan.locations:
        cam = d.get("camera_id")
        loc = G.nodes.get(f"cam:{cam}", {}).get("location", "") if cam else ""
        if loc not in plan.locations:
            return False
    if plan.modalities:
        mod = "Audio" if d["type"] == S.TRANSCRIPT else d.get("modality") or ("CCTV" if d["type"] in S.ENTITY_TYPES + (S.EVENT,) else "")
        if mod not in plan.modalities:
            return False
    if plan.keywords and not plan.classes and not plan.colours:
        hay = " ".join(str(d.get(k, "")) for k in ("label", "text", "original_text", "description", "review_notes",
                                                    "object_class", "camera_id")).lower()
        if not any(k in hay for k in plan.keywords):
            return False
    return True


def _in_window(G, n: str, plan: QueryPlan) -> bool:
    d = G.nodes[n]
    if plan.rel_window:
        t0, t1 = d.get("t_start"), d.get("t_end")
        if t0 is None:
            return False
        t1 = t0 if t1 is None else t1
        return t0 <= plan.rel_window[1] and t1 >= plan.rel_window[0]
    if plan.abs_window:
        if not d.get("abs_start"):
            return False
        a0 = datetime.fromisoformat(d["abs_start"]).time()
        a1 = datetime.fromisoformat(d.get("abs_end") or d["abs_start"]).time()
        return a0 <= plan.abs_window[1] and a1 >= plan.abs_window[0]
    return True


def _evidence_label(status: str, origin_ai: bool = True) -> str:
    if status == "Approved":
        return "Investigator-verified"
    if status == "Rejected":
        return "Rejected by investigator"
    if status == "Flagged":
        return "Flagged for review"
    return "AI finding (unverified)" if origin_ai else "Unreviewed"


def _row(G, n: str, how: str) -> Dict[str, Any]:
    d = G.nodes[n]
    refs = []
    if d.get("observation_id"):
        refs.append(f"obs {d['observation_id']}")
    if d.get("best_observation"):
        refs.append(f"best obs {d['best_observation']}")
    if d.get("event_id"):
        refs.append(d["event_id"])
    src = G.nodes.get(f"src:{d.get('source_id', '')}", {})
    if src:
        refs.append(f"file {src.get('label', '')}")
    if d["type"] == S.TRANSCRIPT:
        what = f"Speech: \"{(d.get('text') or d.get('original_text') or '')[:160]}\""
    elif d["type"] == S.EVENT:
        what = f"{d.get('label')}: {d.get('observation_count', 0)} observations"
    elif n.startswith("trk:"):
        colour = f", approx. {d['dominant_colour']}" if d.get("dominant_colour") else ""
        what = f"{d.get('object_class')} (anonymous track {S.short_track(d.get('track_id', ''))}{colour}), {d.get('observation_count', 0)} sightings"
    else:
        what = d.get("description") or d.get("label", n)
    if d.get("abs_start"):
        when, sort_key = d["abs_start"], (0, d["abs_start"], "", 0.0)
    elif d.get("t_start") is not None:
        end = d.get("t_end")
        when = f"{d['t_start']:.1f}s" + (f" to {end:.1f}s" if end is not None and end - d["t_start"] >= 0.1 else "") + " (relative)"
        sort_key = (1, "", d.get("source_id", ""), float(d["t_start"]))
    else:
        when, sort_key = "Not timestamped", (2, "", d.get("source_id", ""), 0.0)
    conf = d.get("confidence")
    return {
        "node": n, "_sort": sort_key, "When": when, "Camera / source": d.get("camera_id") or src.get("label", "-"),
        "Type": d["type"], "What": what, "Confidence": f"{conf:.0%}" if isinstance(conf, (int, float)) else "-",
        "Evidence status": _evidence_label(d.get("status", "Pending")), "Found by": how, "References": "; ".join(refs),
    }


@dataclass
class QueryResult:
    plan: QueryPlan
    timeline: List[Dict[str, Any]]
    matched: Set[str]
    related: Set[str]
    contradictions: List[Dict[str, Any]]
    cameras: List[Dict[str, Any]]
    supporting_edges: List[Dict[str, Any]]


def answer_question(question: str, eg: EvidenceGraph, max_rows: int = 200) -> QueryResult:
    G = eg.graph
    plan = parse_question(question, eg)

    matched: Set[str] = set(plan.node_ids)
    has_filters = bool(plan.classes or plan.colours or plan.cameras or plan.locations or plan.keywords
                       or plan.rel_window or plan.abs_window or plan.node_ids or plan.modalities)
    if plan.wants_conflicts and not has_filters:
        # "Any conflicting observations?" -> only the items involved in a contradiction.
        matched |= {n for c in eg.contradictions for n in c["nodes"] if n in G}
    else:
        for n, d in G.nodes(data=True):
            if d["type"] in QUERYABLE and _matches(G, n, d, plan) and _in_window(G, n, plan):
                matched.add(n)

    related: Set[str] = set()
    via: Dict[str, str] = {}
    if plan.wants_related or plan.node_ids:
        for seed in list(matched):
            for _, m, e in list(G.out_edges(seed, data=True)) + [(m, seed, e) for m, _, e in G.in_edges(seed, data=True)]:
                other = m if m != seed else _
                if e["relation"] not in S.EVIDENTIAL_RELATIONS or other in matched:
                    continue
                od = G.nodes[other]
                if od["type"] not in QUERYABLE:
                    continue
                if od["type"] == S.OBSERVATION and od.get("modality") == "CCTV" and od.get("track_id", "") not in ("", "Untracked"):
                    continue
                if (plan.rel_window or plan.abs_window) and not _in_window(G, other, plan) and od.get("t_start") is not None:
                    continue
                related.add(other)
                via.setdefault(other, f"{e['relation']} from {G.nodes[seed].get('label', seed)}")

    rows = [_row(G, n, "Direct match") for n in matched] + [_row(G, n, via[n]) for n in related]
    rows.sort(key=lambda r: r["_sort"])
    rows = rows[:max_rows]

    scope = matched | related
    contradictions = [c for c in eg.contradictions if set(c["nodes"]) & scope] if scope else list(eg.contradictions)
    if plan.wants_conflicts and not has_filters:
        contradictions = list(eg.contradictions)

    cam_rows: Dict[str, Dict[str, Any]] = {}
    for n in scope:
        d = G.nodes[n]
        cam = d.get("camera_id")
        if not cam or d["type"] == S.TRANSCRIPT:
            continue
        r = cam_rows.setdefault(cam, {"Camera": cam, "Location": G.nodes.get(f"cam:{cam}", {}).get("location", "") or "-",
                                      "Items": 0, "First seen": None, "Last seen": None})
        r["Items"] += 1
        t = d.get("abs_start") or (f"{d['t_start']:.1f}s" if d.get("t_start") is not None else None)
        if t:
            r["First seen"] = t if r["First seen"] is None or str(t) < str(r["First seen"]) else r["First seen"]
            r["Last seen"] = t if r["Last seen"] is None or str(t) > str(r["Last seen"]) else r["Last seen"]

    supporting = []
    for u, v, e in G.edges(data=True):
        if e["relation"] in ("SUPPORTS", "HEARD_DURING", "CROSSED_CAMERA", "POSSIBLY_SAME_OBJECT", "TEMPORALLY_OVERLAPS") \
                and (u in scope and v in scope):
            supporting.append({"Relation": e["relation"], "From": G.nodes[u].get("label", u),
                               "To": G.nodes[v].get("label", v), "Confidence": f"{e['confidence']:.0%}",
                               "Origin": e["origin"], "Status": e["status"], "Reasoning": e["reasoning"]})

    if plan.colours and not matched:
        plan.notes.append("No tracked object had a matching estimated colour. Try the question without the colour "
                          "to see all objects of that class.")
    return QueryResult(plan, rows, matched, related, contradictions, sorted(cam_rows.values(), key=lambda r: r["Camera"]),
                       supporting)


def entity_sequence(eg: EvidenceGraph, node: str) -> List[Dict[str, Any]]:
    """Chronological sequence of everything directly evidenced about one entity or event."""
    G = eg.graph
    if node not in G:
        return []
    members = A.neighbourhood(G, node, depth=1, relations=S.EVIDENTIAL_RELATIONS) - {node}
    rows = [_row(G, n, "Linked") for n in members if G.nodes[n]["type"] in QUERYABLE]
    rows.append(_row(G, node, "Selected"))
    rows.sort(key=lambda r: r["_sort"])
    return rows
