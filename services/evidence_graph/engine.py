"""Heterogeneous temporal evidence graph.

The graph is *derived*: it is rebuilt from the stored case data (sources, AI
observations, events, stored proposals and investigator decisions) by explicit,
documented rules. Nothing is persisted here, so the graph can always be
re-derived and every edge can be explained.

Each edge carries:
    relation    one of schema.RELATIONS
    confidence  0..1 (rule-specific; documented in `reasoning`)
    reasoning   human-readable justification, including thresholds used
    origin      Evidence registry | Derived rule | AI proposal | Investigator
    status      review status for proposals/decisions; "System" for registry/rule facts
    provenance  list of source / observation / record IDs the edge was derived from
    basis       clock basis for temporal claims (shared-clock / anchored) or ""

Each node carries `type`, `label`, timing (`t_start`/`t_end` source-relative seconds,
`abs_start`/`abs_end` ISO strings when the source clock is anchored) and provenance.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

from core.config import (
    CROSS_CAMERA_MAX_GAP_S,
    HEARD_DURING_WINDOW_S,
    SIMULTANEITY_CONTRADICTION_S,
    TEMPORAL_OVERLAP_SLACK_S,
    get_logger,
)
from core.timebase import Timebase
from services.evidence_graph import appearance
from services.evidence_graph import schema as S

log = get_logger("evidence_graph")

EVENT_CHRONOLOGY_MAX_GAP_S = 300.0
IDENTITY_RELATIONS = {"POSSIBLY_SAME_OBJECT", "CROSSED_CAMERA"}

# Stored cross-modal proposal type -> graph relation
CROSS_MODAL_MAP = {
    "CCTV_AUDIO_MENTION": "SUPPORTS",
    "AUDIO_PHOTO_MENTION": "SUPPORTS",
    "CCTV_PHOTO_CONTEXT": "POSSIBLY_SAME_OBJECT",
}


def _investigator_relation(rel_type: str) -> Optional[str]:
    t = (rel_type or "").upper()
    if t.startswith("TRANSIT_PATH"):
        return "CROSSED_CAMERA"
    if t.startswith("SAME_ENTITY"):
        return "POSSIBLY_SAME_OBJECT"
    if t.startswith("CO_OCCURRENCE"):
        return "TEMPORALLY_OVERLAPS"
    if t.startswith("SUSPICIOUS_PROXIMITY"):
        return "LOCATED_NEAR"
    if t.startswith("DISTINCT_ENTITIES"):
        return "DISTINCT_ENTITIES"
    if t in S.RELATIONS:
        return t
    return None


# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------
@dataclass
class CaseData:
    sources: List[Dict[str, Any]]
    observations: pd.DataFrame
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_members: pd.DataFrame = field(default_factory=pd.DataFrame)
    cross_camera_links: pd.DataFrame = field(default_factory=pd.DataFrame)
    cross_modal: pd.DataFrame = field(default_factory=pd.DataFrame)
    investigator_links: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        h.update(repr(sorted((s["source_id"], s.get("camera_id"), s.get("location"), s.get("recorded_start"))
                             for s in self.sources)).encode())
        for df in (self.observations, self.events, self.event_members, self.cross_camera_links,
                   self.cross_modal, self.investigator_links):
            if df is not None and not df.empty:
                h.update(str(int(pd.util.hash_pandas_object(df.astype(str), index=False).sum())).encode())
            h.update(b"|")
        return h.hexdigest()[:16]


def load_case_data() -> CaseData:
    """Read everything the graph needs from the case databases."""
    from services.correlation import cross_camera_linking, cross_modal_correlation
    from services.events import event_reconstruction
    from storage import db_manager

    def _safe(fn, *args, **kwargs) -> pd.DataFrame:
        try:
            out = fn(*args, **kwargs)
            return out if isinstance(out, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:  # a missing auxiliary DB must not break the graph
            log.warning("graph input %s unavailable: %s", getattr(fn, "__name__", fn), exc)
            return pd.DataFrame()

    cross_modal_correlation.initialise_correlation_database()
    return CaseData(
        sources=db_manager.get_all_sources(),
        observations=db_manager.get_all_observations_df(),
        events=_safe(event_reconstruction.get_reconstructed_events, active_only=True),
        event_members=_safe(event_reconstruction.get_all_event_memberships, active_only=True),
        cross_camera_links=_safe(cross_camera_linking.get_links),
        cross_modal=_safe(cross_modal_correlation.get_cross_modal_relationships),
        investigator_links=_safe(db_manager.get_graph_relationships),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _num(value) -> Optional[float]:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(v) else v


def _txt(value, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    return s if s else default


def _iso(dt) -> str:
    return dt.isoformat(sep=" ", timespec="seconds") if dt is not None else ""


def _is_tracked(track_id: str) -> bool:
    return track_id not in ("", "Untracked", "Not applicable", "nan", "None")


class _Builder:
    def __init__(self, data: CaseData):
        self.data = data
        self.G = nx.MultiDiGraph()
        self.tb = Timebase.from_sources(data.sources)
        self.sources_by_id = {str(s["source_id"]): s for s in data.sources}
        self.contradictions: List[Dict[str, Any]] = []

    # -- primitives ------------------------------------------------------
    def node(self, node_id: str, **attrs) -> str:
        if node_id in self.G:
            self.G.nodes[node_id].update({k: v for k, v in attrs.items() if v not in (None, "")})
        else:
            self.G.add_node(node_id, **attrs)
        return node_id

    def edge(self, u: str, v: str, relation: str, *, confidence: float, reasoning: str, origin: str,
             status: str = "System", provenance: Iterable[str] = (), basis: str = "", **extra) -> None:
        if u == v or u not in self.G or v not in self.G:
            return
        key = f"{relation}:{extra.pop('key_suffix', '')}"
        if self.G.has_edge(u, v, key=key):
            return
        self.G.add_edge(
            u, v, key=key, relation=relation, confidence=round(float(max(0.0, min(1.0, confidence))), 3),
            reasoning=reasoning, origin=origin, status=status, provenance=list(provenance), basis=basis, **extra,
        )

    def timing(self, source_id: str, t_start: Optional[float], t_end: Optional[float]) -> Dict[str, Any]:
        abs_s = self.tb.to_absolute(source_id, t_start)
        abs_e = self.tb.to_absolute(source_id, t_end if t_end is not None else t_start)
        return {
            "t_start": t_start, "t_end": t_end if t_end is not None else t_start,
            "abs_start": _iso(abs_s), "abs_end": _iso(abs_e),
            "clock": "anchored" if abs_s is not None else ("source-relative" if t_start is not None else "none"),
        }

    # -- R1: evidence registry -------------------------------------------
    def add_registry(self) -> None:
        for src in self.data.sources:
            sid = str(src["source_id"])
            modality = src.get("modality", "CCTV")
            ntype = {"CCTV": S.VIDEO, "Audio": S.AUDIO, "Photo": S.IMAGE}.get(modality, S.VIDEO)
            src_node = self.node(
                f"src:{sid}", type=ntype, label=_txt(src.get("source_name"), sid)[:40],
                source_id=sid, camera_id=_txt(src.get("camera_id")), modality=modality,
                sha256=_txt(src.get("sha256_hash")), file_path=_txt(src.get("file_path")),
                uploaded=_txt(src.get("upload_time")), recorded_start=_txt(src.get("recorded_start")),
                location=_txt(src.get("location")), duration=_num(src.get("duration_seconds")),
                **self.timing(sid, 0.0, _num(src.get("duration_seconds")) or 0.0),
            )
            if modality == "CCTV":
                cam_id = _txt(src.get("camera_id"), "UNASSIGNED")
                cam = self.node(f"cam:{cam_id}", type=S.CAMERA, label=cam_id, camera_id=cam_id)
                self.edge(src_node, cam, "CAPTURED_BY", confidence=1.0, origin=S.ORIGIN_REGISTRY,
                          reasoning=f"Evidence registry assigns this recording to camera {cam_id}.",
                          provenance=[sid])
                loc = _txt(src.get("location"))
                if loc:
                    self.G.nodes[cam]["location"] = loc
                    loc_node = self.node(f"loc:{loc.lower()}", type=S.LOCATION, label=loc)
                    self.edge(cam, loc_node, "LOCATED_NEAR", confidence=1.0, origin=S.ORIGIN_REGISTRY,
                              reasoning=f"Investigator recorded camera {cam_id} location as '{loc}'.",
                              provenance=[sid])

    # -- R2/R3: observations, transcripts and tracks ----------------------
    def add_observations(self) -> None:
        df = self.data.observations
        if df is None or df.empty:
            return
        tracks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in df.to_dict("records"):
            oid = _txt(row.get("Observation ID"))
            if not oid:
                continue
            sid = _txt(row.get("Source ID"))
            modality = _txt(row.get("Modality"), "CCTV")
            obj = _txt(row.get("Object"), "unknown")
            t0 = _num(row.get("Timestamp (seconds)"))
            t1 = _num(row.get("Segment End (seconds)"))
            is_speech = modality == "Audio"
            attrs = dict(
                type=S.TRANSCRIPT if is_speech else S.OBSERVATION,
                label=(f"Speech {_txt(row.get('Timestamp'))}" if is_speech else f"{obj} {_txt(row.get('Timestamp'))}"),
                observation_id=oid, source_id=sid, modality=modality, object_class=obj,
                category="Speech" if is_speech else S.entity_type_for_class(obj),
                camera_id=_txt(row.get("Camera ID")), timestamp=_txt(row.get("Timestamp")),
                confidence=_num(row.get("Confidence")), status=_txt(row.get("Review Status"), "Pending"),
                review_notes=_txt(row.get("Review Notes")), evidence_path=_txt(row.get("Evidence Path")),
                bbox=_txt(row.get("Bounding Box")), track_id=_txt(row.get("Track ID")),
                text=_txt(row.get("English Translation")) or _txt(row.get("Original Transcript")),
                original_text=_txt(row.get("Original Transcript")), language=_txt(row.get("Detected Language")),
                description=_txt(row.get("Observation")),
                **self.timing(sid, t0, t1),
            )
            obs = self.node(f"obs:{oid}", **attrs)
            src_node = f"src:{sid}"
            if src_node in self.G:
                conf = _num(row.get("Confidence"))
                self.edge(obs, src_node, "DETECTED_IN", confidence=1.0, origin=S.ORIGIN_REGISTRY,
                          reasoning=("Speech segment transcribed from this audio track." if is_speech else
                                     f"YOLO detected '{obj}'" + (f" with {conf:.0%} confidence" if conf is not None else "")
                                     + " in this source."),
                          provenance=[oid, sid])
            track_id = attrs["track_id"]
            if modality == "CCTV" and _is_tracked(track_id):
                tracks[f"trk:{sid}:{track_id}"].append(attrs)

        for trk_id, members in tracks.items():
            classes = Counter(m["object_class"] for m in members)
            obj = classes.most_common(1)[0][0]
            sid = members[0]["source_id"]
            times = [m["t_start"] for m in members if m["t_start"] is not None]
            confs = [m["confidence"] for m in members if m["confidence"] is not None]
            best = max(members, key=lambda m: m["confidence"] or 0.0)
            statuses = Counter(m["status"] for m in members)
            t0, t1 = (min(times), max(times)) if times else (None, None)
            raw_track = members[0]["track_id"]
            node_attrs = dict(
                type=S.entity_type_for_class(obj), label=f"{obj} {S.short_track(raw_track)}",
                object_class=obj, category=S.entity_type_for_class(obj), source_id=sid, track_id=raw_track,
                camera_id=members[0]["camera_id"], observation_count=len(members),
                confidence=round(sum(confs) / len(confs), 3) if confs else None,
                evidence_path=best["evidence_path"], bbox=best["bbox"], best_observation=best["observation_id"],
                status_summary=dict(statuses), status=("Approved" if statuses.get("Approved") else
                                                       "Rejected" if statuses.get("Rejected") == len(members) else "Pending"),
                **self.timing(sid, t0, t1),
            )
            crop = appearance.load_crop(best["evidence_path"], best["bbox"])
            if crop is not None:
                colour, share = appearance.dominant_colour(crop)
                node_attrs.update(dominant_colour=colour, dominant_colour_share=round(share, 2))
            trk = self.node(trk_id, **node_attrs)
            for m in members:
                self.edge(f"obs:{m['observation_id']}", trk, "SAME_TRACK", confidence=0.9, origin=S.ORIGIN_RULE,
                          reasoning=f"ByteTrack assigned track {raw_track} within source {sid}. "
                                    "Tracker continuity is algorithmic and can switch IDs under occlusion.",
                          provenance=[m["observation_id"]])
            cam_node = f"cam:{members[0]['camera_id']}"
            if cam_node in self.G:
                span = f"{t0:.1f}s-{t1:.1f}s" if t0 is not None else "unknown time"
                self.edge(trk, cam_node, "APPEARED_AT", confidence=0.9, origin=S.ORIGIN_RULE,
                          reasoning=f"{len(members)} sampled detection(s) of this track on {members[0]['camera_id']} ({span}).",
                          provenance=[m["observation_id"] for m in members], t_start=t0, t_end=t1)

    # -- R4: events --------------------------------------------------------
    def add_events(self) -> None:
        ev = self.data.events
        if ev is None or ev.empty:
            return
        members = defaultdict(list)
        if self.data.event_members is not None and not self.data.event_members.empty:
            for r in self.data.event_members.to_dict("records"):
                members[str(r["event_id"])].append(str(r["observation_id"]))
        for r in ev.to_dict("records"):
            eid = str(r["event_id"])
            sid = _txt(r.get("source_id"))
            t0, t1 = _num(r.get("start_seconds")), _num(r.get("end_seconds"))
            src = self.sources_by_id.get(sid, {})
            evt = self.node(
                f"evt:{eid}", type=S.EVENT, label=_txt(r.get("event_title"), "Event")[:32], event_id=eid,
                source_id=sid, camera_id=_txt(src.get("camera_id")), object_class=_txt(r.get("object_category")),
                category=S.entity_type_for_class(_txt(r.get("object_category"))),
                confidence=_num(r.get("average_confidence")), status=_txt(r.get("review_status"), "Pending"),
                description=_txt(r.get("event_explanation")), observation_count=int(_num(r.get("observation_count")) or 0),
                **self.timing(sid, t0, t1),
            )
            linked_tracks = set()
            for oid in members.get(eid, []):
                obs = f"obs:{oid}"
                if obs not in self.G:
                    continue
                self.edge(obs, evt, "ASSOCIATED_WITH_EVENT", confidence=0.9, origin=S.ORIGIN_RULE,
                          reasoning="Observation was grouped into this candidate event by the event "
                                    "reconstruction engine (same track, gaps within the configured limit).",
                          provenance=[oid, eid])
                trk = self.G.nodes[obs].get("track_id")
                if trk and _is_tracked(trk):
                    linked_tracks.add(f"trk:{sid}:{trk}")
            for trk in linked_tracks:
                self.edge(trk, evt, "ASSOCIATED_WITH_EVENT", confidence=0.9, origin=S.ORIGIN_RULE,
                          reasoning="Tracked entity is the subject of this candidate event.", provenance=[eid])

    # -- R5: chronology & overlap between events ----------------------------
    def add_event_chronology(self) -> None:
        events = [n for n, d in self.G.nodes(data=True) if d["type"] == S.EVENT and d.get("t_start") is not None]
        for a in events:
            da = self.G.nodes[a]
            nearest: Optional[Tuple[float, str, str]] = None
            for b in events:
                if a == b:
                    continue
                db = self.G.nodes[b]
                cmp = self.tb.compare_intervals(da["source_id"], da["t_start"], da["t_end"],
                                                db["source_id"], db["t_start"], db["t_end"])
                if cmp is None:
                    continue
                order, secs, basis = cmp
                if order == "overlaps" or (order == "a_before_b" and secs <= TEMPORAL_OVERLAP_SLACK_S):
                    if a < b:  # one undirected fact, stored once
                        self.edge(a, b, "TEMPORALLY_OVERLAPS", confidence=0.95 if basis == "shared-clock" else 0.8,
                                  origin=S.ORIGIN_RULE, basis=basis, provenance=[da["event_id"], db["event_id"]],
                                  reasoning=f"Event intervals overlap by {max(secs, 0):.1f}s ({basis}; slack "
                                            f"{TEMPORAL_OVERLAP_SLACK_S:.0f}s).")
                elif order == "a_before_b" and secs <= EVENT_CHRONOLOGY_MAX_GAP_S:
                    if nearest is None or secs < nearest[0]:
                        nearest = (secs, b, basis)
            if nearest:
                secs, b, basis = nearest
                self.edge(a, b, "APPEARED_BEFORE", confidence=0.95 if basis == "shared-clock" else 0.8,
                          origin=S.ORIGIN_RULE, basis=basis, gap_seconds=round(secs, 2),
                          provenance=[da["event_id"], self.G.nodes[b]["event_id"]],
                          reasoning=f"Next comparable event begins {secs:.1f}s after this one ends ({basis}). "
                                    f"Only the nearest successor within {EVENT_CHRONOLOGY_MAX_GAP_S:.0f}s is linked.")

    # -- R6: cross-camera association between tracks -------------------------
    def add_cross_camera(self) -> None:
        tracks = [n for n, d in self.G.nodes(data=True)
                  if d["type"] in S.ENTITY_TYPES and n.startswith("trk:") and d.get("t_start") is not None]
        for a in tracks:
            da = self.G.nodes[a]
            best_per_camera: Dict[str, Tuple[float, str, str]] = {}
            for b in tracks:
                db = self.G.nodes[b]
                if a == b or da["camera_id"] == db["camera_id"] or da["object_class"] != db["object_class"]:
                    continue
                cmp = self.tb.compare_intervals(da["source_id"], da["t_start"], da["t_end"],
                                                db["source_id"], db["t_start"], db["t_end"])
                if cmp is None:
                    continue
                order, secs, basis = cmp
                if order == "a_before_b" and secs <= CROSS_CAMERA_MAX_GAP_S:
                    prev = best_per_camera.get(db["camera_id"])
                    if prev is None or secs < prev[0]:
                        best_per_camera[db["camera_id"]] = (secs, b, basis)
            for secs, b, basis in best_per_camera.values():
                db = self.G.nodes[b]
                timing_score = 1.0 - secs / CROSS_CAMERA_MAX_GAP_S
                crop_a = appearance.load_crop(da.get("evidence_path"), da.get("bbox"))
                crop_b = appearance.load_crop(db.get("evidence_path"), db.get("bbox"))
                if crop_a is not None and crop_b is not None:
                    sim = appearance.histogram_similarity(crop_a, crop_b)
                    conf = 0.15 + 0.35 * timing_score + 0.35 * sim
                    visual = f"colour-histogram similarity {sim:.2f}"
                else:
                    sim = None
                    conf = 0.15 + 0.3 * timing_score
                    visual = "no usable crops, appearance not compared"
                self.edge(a, b, "CROSSED_CAMERA", confidence=min(conf, 0.85), origin=S.ORIGIN_RULE, basis=basis,
                          gap_seconds=round(secs, 2), appearance_similarity=sim,
                          provenance=[da.get("best_observation", ""), db.get("best_observation", "")],
                          reasoning=f"Same class '{da['object_class']}' left {da['camera_id']} and appeared on "
                                    f"{db['camera_id']} {secs:.1f}s later ({basis}); {visual}. Candidate hand-off "
                                    "only: class, timing and colour cannot establish identity.")

    # -- R7/R8: audio <-> visual ---------------------------------------------
    def add_audio_visual(self) -> None:
        from services.correlation.cross_modal_correlation import transcript_mentions_object

        transcripts = [n for n, d in self.G.nodes(data=True) if d["type"] == S.TRANSCRIPT and d.get("t_start") is not None]
        visuals = [n for n, d in self.G.nodes(data=True)
                   if d.get("t_start") is not None and (
                       (n.startswith("trk:")) or
                       (d["type"] == S.OBSERVATION and d.get("modality") == "CCTV" and not _is_tracked(d.get("track_id", ""))))]
        for t in transcripts:
            dt = self.G.nodes[t]
            text = f"{dt.get('original_text', '')} {dt.get('text', '')}"
            for v in visuals:
                dv = self.G.nodes[v]
                cmp = self.tb.compare_intervals(dt["source_id"], dt["t_start"] - HEARD_DURING_WINDOW_S,
                                                dt["t_end"] + HEARD_DURING_WINDOW_S,
                                                dv["source_id"], dv["t_start"], dv["t_end"])
                if cmp is None or cmp[0] != "overlaps":
                    continue
                basis = cmp[2]
                base = 0.7 if basis == "shared-clock" else 0.5
                self.edge(t, v, "HEARD_DURING", confidence=base, origin=S.ORIGIN_RULE, basis=basis,
                          provenance=[dt["observation_id"], dv.get("best_observation") or dv.get("observation_id", "")],
                          reasoning=f"Speech at {dt.get('timestamp')} overlaps the visual evidence within "
                                    f"±{HEARD_DURING_WINDOW_S:.0f}s ({basis}).")
                matches = transcript_mentions_object(text, dv.get("object_class", "")) if text.strip() else []
                if matches:
                    self.edge(t, v, "SUPPORTS", confidence=base + 0.1, origin=S.ORIGIN_RULE, basis=basis,
                              matched_terms=matches,
                              provenance=[dt["observation_id"], dv.get("best_observation") or dv.get("observation_id", "")],
                              reasoning=f"Speech mentions '{', '.join(matches)}' while a '{dv.get('object_class')}' "
                                        f"is visible ({basis}). The speaker may still refer to a different object.")

    # -- R9/R10: stored proposals and investigator decisions -----------------
    def add_stored_relationships(self) -> None:
        cm = self.data.cross_modal
        if cm is not None and not cm.empty:
            for r in cm.to_dict("records"):
                rel = CROSS_MODAL_MAP.get(_txt(r.get("relationship_type")))
                status = _txt(r.get("status"), "Pending")
                if rel is None or status == "Rejected":
                    continue
                self.edge(f"obs:{_txt(r.get('source_observation'))}", f"obs:{_txt(r.get('target_observation'))}", rel,
                          confidence=_num(r.get("confidence")) or 0.3, origin=S.ORIGIN_AI, status=status,
                          key_suffix=_txt(r.get("relationship_id")),
                          reasoning=_txt(r.get("explanation"), "Stored cross-modal proposal."),
                          provenance=[_txt(r.get("relationship_id"))], record_type=_txt(r.get("relationship_type")))

        links = self.data.cross_camera_links
        if links is not None and not links.empty:
            for r in links.to_dict("records"):
                status = _txt(r.get("review_status"), "Pending")
                if status == "Rejected":
                    continue
                self.edge(f"evt:{_txt(r.get('event_a'))}", f"evt:{_txt(r.get('event_b'))}", "POSSIBLY_SAME_OBJECT",
                          confidence=0.5, origin=S.ORIGIN_AI, status=status, basis="anchored",
                          key_suffix=_txt(r.get("link_id")), provenance=[_txt(r.get("link_id"))],
                          gap_seconds=_num(r.get("time_gap_seconds")),
                          reasoning=_txt(r.get("explanation"), "Cross-camera link proposal."))

        inv = self.data.investigator_links
        if inv is not None and not inv.empty:
            for r in inv.to_dict("records"):
                rel = _investigator_relation(_txt(r.get("relationship_type")))
                status = _txt(r.get("status"), "Pending")
                if rel is None or status == "Rejected":
                    continue
                self.edge(f"obs:{_txt(r.get('source_observation'))}", f"obs:{_txt(r.get('target_observation'))}", rel,
                          confidence=_num(r.get("confidence")) or 0.5, origin=S.ORIGIN_INVESTIGATOR, status=status,
                          key_suffix=_txt(r.get("relationship_id")), provenance=[_txt(r.get("relationship_id"))],
                          record_type=_txt(r.get("relationship_type")),
                          reasoning=_txt(r.get("explanation"), "Investigator-recorded relationship.")
                          + (f" Notes: {_txt(r.get('reviewer_notes'))}" if _txt(r.get("reviewer_notes")) else ""))

    # -- contradiction detection ---------------------------------------------
    def _entity_view(self, n: str) -> Dict[str, Any]:
        d = self.G.nodes[n]
        loc = ""
        cam = d.get("camera_id", "")
        if cam and f"cam:{cam}" in self.G:
            loc = self.G.nodes[f"cam:{cam}"].get("location", "")
        return {"source_id": d.get("source_id", ""), "t_start": d.get("t_start"), "t_end": d.get("t_end"),
                "camera_id": cam, "location": loc, "object_class": d.get("object_class", ""),
                "category": d.get("category", "")}

    def _contradict(self, a: str, b: str, kind: str, reasoning: str, provenance: List[str], severity: str) -> None:
        self.contradictions.append({"kind": kind, "nodes": [a, b], "reasoning": reasoning,
                                    "severity": severity, "provenance": provenance})
        self.edge(a, b, "CONTRADICTS", confidence=1.0 if severity == "high" else 0.7, origin=S.ORIGIN_RULE,
                  key_suffix=kind, reasoning=f"[{kind}] {reasoning}", provenance=provenance)

    def detect_contradictions(self) -> None:
        identity_edges = [(u, v, d) for u, v, d in self.G.edges(data=True) if d["relation"] in IDENTITY_RELATIONS]
        distinct_pairs = {frozenset((u, v)) for u, v, d in self.G.edges(data=True) if d["relation"] == "DISTINCT_ENTITIES"}

        for u, v, d in identity_edges:
            a, b = self._entity_view(u), self._entity_view(v)
            prov = list(d.get("provenance", []))
            # C1: one entity in two different places at the same time.
            if a["location"] and b["location"] and a["location"].lower() != b["location"].lower() \
                    and a["t_start"] is not None and b["t_start"] is not None:
                cmp = self.tb.compare_intervals(a["source_id"], a["t_start"], a["t_end"] or a["t_start"],
                                                b["source_id"], b["t_start"], b["t_end"] or b["t_start"])
                if cmp and (cmp[0] == "overlaps" or cmp[1] <= SIMULTANEITY_CONTRADICTION_S):
                    self._contradict(u, v, "location-simultaneity",
                                     f"Linked as the same object, but seen at '{a['location']}' and '{b['location']}' "
                                     f"at the same time ({cmp[2]}).", prov, "high")
            # C2: identity link between different entity categories.
            if a["category"] and b["category"] and a["category"] != b["category"] \
                    and "Speech" not in (a["category"], b["category"]):
                self._contradict(u, v, "class-mismatch",
                                 f"Linked as the same object, but classified as '{a['object_class']}' and "
                                 f"'{b['object_class']}'.", prov, "medium")
            # C3: investigator decided both "same" and "distinct" for one pair.
            if frozenset((u, v)) in distinct_pairs:
                self._contradict(u, v, "decision-conflict",
                                 "This pair is recorded both as a possible same object and as distinct entities.",
                                 prov, "high")
            # C5: claimed transit A->B, but B happened first.
            if d["relation"] == "CROSSED_CAMERA" and d.get("origin") == S.ORIGIN_INVESTIGATOR \
                    and a["t_start"] is not None and b["t_start"] is not None:
                cmp = self.tb.compare_intervals(a["source_id"], a["t_start"], a["t_end"] or a["t_start"],
                                                b["source_id"], b["t_start"], b["t_end"] or b["t_start"])
                if cmp and cmp[0] == "b_before_a":
                    self._contradict(u, v, "temporal-order",
                                     f"Recorded as a transit from the first to the second sighting, but the second "
                                     f"occurred {cmp[1]:.1f}s earlier ({cmp[2]}).", prov, "high")

        # C4: event decision contradicts the decisions on its member observations.
        for n, d in self.G.nodes(data=True):
            if d["type"] != S.EVENT:
                continue
            member_status = [self.G.nodes[u].get("status") for u, _, e in self.G.in_edges(n, data=True)
                             if e["relation"] == "ASSOCIATED_WITH_EVENT" and self.G.nodes[u]["type"] == S.OBSERVATION]
            if not member_status:
                continue
            rejected = sum(s == "Rejected" for s in member_status)
            approved = sum(s == "Approved" for s in member_status)
            if d.get("status") == "Approved" and rejected * 2 >= len(member_status):
                bad = next(u for u, _, e in self.G.in_edges(n, data=True)
                           if e["relation"] == "ASSOCIATED_WITH_EVENT" and self.G.nodes[u].get("status") == "Rejected")
                self._contradict(bad, n, "review-conflict",
                                 f"Event is Approved, but {rejected} of {len(member_status)} member observations "
                                 "were Rejected.", [d.get("event_id", "")], "medium")
            elif d.get("status") == "Rejected" and approved:
                good = next(u for u, _, e in self.G.in_edges(n, data=True)
                            if e["relation"] == "ASSOCIATED_WITH_EVENT" and self.G.nodes[u].get("status") == "Approved")
                self._contradict(good, n, "review-conflict",
                                 f"Event is Rejected, but {approved} member observation(s) were Approved.",
                                 [d.get("event_id", "")], "low")

    def build(self) -> "EvidenceGraph":
        self.add_registry()
        self.add_observations()
        self.add_events()
        self.add_event_chronology()
        self.add_cross_camera()
        self.add_audio_visual()
        self.add_stored_relationships()
        self.detect_contradictions()
        return EvidenceGraph(self.G, self.contradictions, self.tb)


@dataclass
class EvidenceGraph:
    graph: nx.MultiDiGraph
    contradictions: List[Dict[str, Any]]
    timebase: Timebase

    def summary(self) -> Dict[str, Any]:
        node_types = Counter(d["type"] for _, d in self.graph.nodes(data=True))
        relations = Counter(d["relation"] for _, _, d in self.graph.edges(data=True))
        origins = Counter(d["origin"] for _, _, d in self.graph.edges(data=True))
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": dict(node_types),
            "relations": dict(relations),
            "origins": dict(origins),
            "contradictions": len(self.contradictions),
            "anchored_sources": len(self.timebase.anchors),
        }

    def edges_table(self) -> pd.DataFrame:
        rows = []
        for u, v, d in self.graph.edges(data=True):
            rows.append({
                "Relation": d["relation"], "From": self.graph.nodes[u].get("label", u),
                "To": self.graph.nodes[v].get("label", v), "Confidence": d["confidence"], "Origin": d["origin"],
                "Status": d["status"], "Clock": d.get("basis", ""), "Reasoning": d["reasoning"],
                "Provenance": ", ".join(str(p) for p in d.get("provenance", []) if p), "from_id": u, "to_id": v,
            })
        return pd.DataFrame(rows)


def build_evidence_graph(data: CaseData) -> EvidenceGraph:
    return _Builder(data).build()
