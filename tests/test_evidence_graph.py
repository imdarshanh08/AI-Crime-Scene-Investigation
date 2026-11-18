import unittest
from datetime import datetime

import pandas as pd

from tests.helpers import frame, obs, source
from core.timebase import Timebase
from services.evidence_graph import analysis as A
from services.evidence_graph import schema as S
from services.evidence_graph.engine import CaseData, build_evidence_graph


def relations(eg, rel):
    return [(u, v, d) for u, v, d in eg.graph.edges(data=True) if d["relation"] == rel]


class TimebaseTests(unittest.TestCase):
    def setUp(self):
        self.tb = Timebase({"A": datetime(2026, 1, 1, 10, 0, 0), "B": datetime(2026, 1, 1, 10, 0, 30)})

    def test_same_source_is_comparable_without_anchor(self):
        self.assertEqual(self.tb.comparable_delta("X", 5, "X", 8), (3.0, "shared-clock"))

    def test_unanchored_sources_are_not_comparable(self):
        self.assertIsNone(self.tb.comparable_delta("A", 5, "X", 8))

    def test_anchored_interval_ordering(self):
        # A: 10:00:10-10:00:20, B: 10:00:35-10:00:40 -> A before B by 15s
        self.assertEqual(self.tb.compare_intervals("A", 10, 20, "B", 5, 10), ("a_before_b", 15.0, "anchored"))
        self.assertEqual(self.tb.compare_intervals("B", 5, 10, "A", 10, 20), ("b_before_a", 15.0, "anchored"))
        order, overlap, _ = self.tb.compare_intervals("A", 30, 40, "B", 0, 5)
        self.assertEqual((order, overlap), ("overlaps", 5.0))


def scenario(location_b="Lobby", investigator=None, events=None, members=None):
    sources = [source("srcA", "CAM-01", start="2026-11-06 14:30:00", location="Gate"),
               source("srcB", "CAM-02", start="2026-11-06 14:31:00", location=location_b),
               source("srcC", "CAM-03")]  # no verified start time
    rows = [
        obs("a1", "srcA", "CAM-01", 10.0, obj="backpack", track="CAM-01-T1-backpack"),
        obs("a2", "srcA", "CAM-01", 20.0, obj="backpack", track="CAM-01-T1-backpack"),
        # 30 s after a2 on the wall clock: 14:31:00 + (-10) is impossible, use 14:31:00 + 0..5
        obs("b1", "srcB", "CAM-02", 0.0, obj="backpack", track="CAM-02-T7-backpack"),
        obs("b2", "srcB", "CAM-02", 5.0, obj="backpack", track="CAM-02-T7-backpack"),
        obs("c1", "srcC", "CAM-03", 12.0, obj="backpack", track="CAM-03-T2-backpack"),
        # speech extracted from CAM-01's own audio track (same source -> shared clock)
        obs("s1", "srcA", "CAM-01", 14.0, obj="Speech", modality="Audio", end=16.0,
            transcript="he left the backpack by the door"),
    ]
    return CaseData(sources=sources, observations=frame(rows),
                    events=events if events is not None else pd.DataFrame(),
                    event_members=members if members is not None else pd.DataFrame(),
                    investigator_links=investigator if investigator is not None else pd.DataFrame())


class GraphEngineTests(unittest.TestCase):
    def test_node_types_and_provenance(self):
        eg = build_evidence_graph(scenario())
        G = eg.graph
        self.assertEqual(G.nodes["cam:CAM-01"]["type"], S.CAMERA)
        self.assertEqual(G.nodes["trk:srcA:CAM-01-T1-backpack"]["type"], S.BELONGING)
        self.assertEqual(G.nodes["obs:s1"]["type"], S.TRANSCRIPT)
        self.assertEqual(G.nodes["loc:gate"]["type"], S.LOCATION)
        for u, v, d in G.edges(data=True):
            for key in ("relation", "confidence", "reasoning", "origin", "provenance"):
                self.assertIn(key, d)
            self.assertTrue(d["reasoning"])

    def test_cross_camera_only_on_comparable_clocks(self):
        eg = build_evidence_graph(scenario())
        crossed = {(u, v) for u, v, _ in relations(eg, "CROSSED_CAMERA")}
        self.assertIn(("trk:srcA:CAM-01-T1-backpack", "trk:srcB:CAM-02-T7-backpack"), crossed)
        # CAM-03 has no verified start time: it must never be linked in time.
        self.assertFalse(any("srcC" in u or "srcC" in v for u, v in crossed))
        gap = next(d["gap_seconds"] for u, v, d in relations(eg, "CROSSED_CAMERA"))
        self.assertAlmostEqual(gap, 40.0)  # A ends 14:30:20, B starts 14:31:00
        self.assertTrue(all(d["confidence"] < 0.9 for _, _, d in relations(eg, "CROSSED_CAMERA")))

    def test_speech_heard_during_and_supports(self):
        eg = build_evidence_graph(scenario())
        target = "trk:srcA:CAM-01-T1-backpack"
        self.assertTrue(any(u == "obs:s1" and v == target for u, v, _ in relations(eg, "HEARD_DURING")))
        support = [d for u, v, d in relations(eg, "SUPPORTS") if u == "obs:s1" and v == target]
        self.assertEqual(len(support), 1)
        self.assertIn("backpack", support[0]["matched_terms"])
        self.assertEqual(support[0]["basis"], "shared-clock")
        # No temporal link to the unanchored CAM-03 track.
        self.assertFalse(any(v.startswith("trk:srcC") for _, v, _ in relations(eg, "HEARD_DURING")))

    def test_location_contradiction(self):
        inv = pd.DataFrame([{"relationship_id": "r1", "source_observation": "a2", "target_observation": "b1",
                             "relationship_type": "SAME_ENTITY_CANDIDATE", "status": "Approved", "confidence": 0.9,
                             "explanation": "", "reviewer_notes": ""}])
        # Shift CAM-02 so b1 (14:30:20) coincides with a2 at a different location.
        data = scenario(investigator=inv)
        data.sources[1]["recorded_start"] = "2026-11-06 14:30:20"
        eg = build_evidence_graph(data)
        kinds = {c["kind"] for c in eg.contradictions}
        self.assertIn("location-simultaneity", kinds)
        self.assertTrue(relations(eg, "CONTRADICTS"))

    def test_no_contradiction_when_same_location(self):
        inv = pd.DataFrame([{"relationship_id": "r1", "source_observation": "a2", "target_observation": "b1",
                             "relationship_type": "SAME_ENTITY_CANDIDATE", "status": "Approved", "confidence": 0.9,
                             "explanation": "", "reviewer_notes": ""}])
        data = scenario(location_b="Gate", investigator=inv)
        data.sources[1]["recorded_start"] = "2026-11-06 14:30:20"
        self.assertEqual(build_evidence_graph(data).contradictions, [])

    def test_rejected_links_are_excluded(self):
        inv = pd.DataFrame([{"relationship_id": "r1", "source_observation": "a1", "target_observation": "c1",
                             "relationship_type": "SAME_ENTITY_CANDIDATE", "status": "Rejected", "confidence": 0.9,
                             "explanation": "", "reviewer_notes": ""}])
        eg = build_evidence_graph(scenario(investigator=inv))
        self.assertFalse(any(d["origin"] == S.ORIGIN_INVESTIGATOR for _, _, d in eg.graph.edges(data=True)))

    def test_events_membership_and_review_conflict(self):
        events = pd.DataFrame([{"event_id": "EVT-1", "event_title": "Backpack tracking event", "source_id": "srcA",
                                "start_seconds": 10.0, "end_seconds": 20.0, "object_category": "backpack",
                                "average_confidence": 0.8, "review_status": "Approved", "observation_count": 2,
                                "event_explanation": ""}])
        members = pd.DataFrame([{"event_id": "EVT-1", "observation_id": "a1"},
                                {"event_id": "EVT-1", "observation_id": "a2"}])
        data = scenario(events=events, members=members)
        data.observations.loc[data.observations["Observation ID"].isin(["a1", "a2"]), "Review Status"] = "Rejected"
        eg = build_evidence_graph(data)
        self.assertEqual(len([1 for u, v, _ in relations(eg, "ASSOCIATED_WITH_EVENT") if v == "evt:EVT-1"]), 3)
        self.assertIn("review-conflict", {c["kind"] for c in eg.contradictions})


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.eg = build_evidence_graph(scenario())
        self.G = self.eg.graph

    def test_neighbourhood_depth(self):
        trk = "trk:srcA:CAM-01-T1-backpack"
        one = A.neighbourhood(self.G, trk, 1)
        self.assertIn("obs:a1", one)
        self.assertNotIn("obs:b1", one)
        self.assertIn("obs:b1", A.neighbourhood(self.G, trk, 2))

    def test_evidence_path_uses_evidential_links(self):
        hops = A.evidence_path(self.G, "obs:a1", "obs:b2", include_structural=False)
        self.assertTrue(hops)
        self.assertIn("CROSSED_CAMERA", [h["relation"] for h in hops])
        self.assertEqual(A.evidence_path(self.G, "obs:a1", "obs:c1", include_structural=False), [])

    def test_time_windows(self):
        rel = A.nodes_in_window(self.G, start=0, end=6)
        self.assertIn("obs:b1", rel)
        self.assertNotIn("obs:a2", rel)
        wall = A.nodes_in_window(self.G, abs_start=datetime(2026, 11, 6, 14, 30, 15),
                                 abs_end=datetime(2026, 11, 6, 14, 30, 25))
        self.assertIn("obs:a2", wall)
        self.assertNotIn("obs:c1", wall)  # unanchored

    def test_communities_and_key_evidence_are_deterministic(self):
        self.assertEqual(A.evidence_communities(self.G, min_size=2), A.evidence_communities(self.G, min_size=2))
        key = A.key_evidence(self.G)
        self.assertTrue(all(k["type"] not in S.STRUCTURAL_TYPES for k in key))


if __name__ == "__main__":
    unittest.main()
