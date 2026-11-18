import os
import tempfile
import unittest
from datetime import time

import pandas as pd

from tests.test_evidence_graph import scenario
from services.evidence_graph.engine import build_evidence_graph
from services.reconstruction.query_engine import answer_question, parse_question
from services.reports import report_generator


class QueryTests(unittest.TestCase):
    def setUp(self):
        self.eg = build_evidence_graph(scenario())

    def test_parse_wall_clock_and_relative_windows(self):
        p = parse_question("What happened between 2:30 PM and 2:31 PM?", self.eg)
        self.assertEqual(p.abs_window, (time(14, 30), time(14, 31)))
        p = parse_question("what happened between 00:05 and 00:15", self.eg)
        self.assertEqual(p.rel_window, (5.0, 15.0))
        self.assertTrue(any("minutes:seconds" in n for n in p.notes))

    def test_object_aliases_and_camera(self):
        p = parse_question("Which cameras recorded the rucksack on CAM-02?", self.eg)
        self.assertEqual(p.classes, {"backpack"})
        self.assertEqual(p.cameras, {"CAM-02"})
        self.assertTrue(p.wants_cameras)

    def test_colour_is_reported_as_approximate(self):
        r = answer_question("Show me everything related to the red backpack", self.eg)
        self.assertIn("red", r.plan.colours)
        self.assertTrue(any("does not output colour" in n for n in r.plan.notes))
        # Synthetic observations have no image crops, so no colour can be verified: nothing is invented.
        self.assertEqual(r.timeline, [])

    def test_wall_clock_query_excludes_unanchored_sources(self):
        r = answer_question("What happened between 2:30 PM and 2:30:30 PM", self.eg)
        cams = {row["Camera / source"] for row in r.timeline}
        self.assertIn("CAM-01", cams)
        self.assertNotIn("CAM-03", cams)
        self.assertEqual([row["_sort"] for row in r.timeline], sorted(row["_sort"] for row in r.timeline))

    def test_related_expansion_cites_relationship(self):
        r = answer_question("Show everything related to the backpack on CAM-01", self.eg)
        found_by = {row["Found by"] for row in r.timeline}
        self.assertIn("Direct match", found_by)
        self.assertTrue(any(f.startswith(("CROSSED_CAMERA", "SUPPORTS", "HEARD_DURING")) for f in found_by))
        self.assertTrue(all(row["Evidence status"] for row in r.timeline))

    def test_conflict_question_without_conflicts(self):
        r = answer_question("Are there any conflicting observations?", self.eg)
        self.assertEqual(r.contradictions, [])
        self.assertEqual(r.timeline, [])


class ReportTests(unittest.TestCase):
    def test_pdf_with_hostile_text_and_graph(self):
        eg = build_evidence_graph(scenario())
        data = scenario()
        with tempfile.TemporaryDirectory() as tmp:
            orig = report_generator.REPORTS_DIR
            report_generator.REPORTS_DIR = type(orig)(tmp)
            try:
                path = report_generator.generate_pdf_report(
                    case_title="Theft & <assault>", case_id="CR/2026:01", investigator_name="O'Neil & Co",
                    sources=data.sources, observations_df=data.observations, events_df=pd.DataFrame(),
                    audit_logs_df=pd.DataFrame([{"timestamp": "t", "action": "a <b>", "entity_type": "e",
                                                 "entity_id": "1", "details": "x & y"}]),
                    notes="line 1 <\nline 2 &", evidence_graph=eg)
                self.assertTrue(os.path.getsize(path) > 2000)
                self.assertNotIn("/", os.path.basename(path).replace("Forensic_Report_", ""))
            finally:
                report_generator.REPORTS_DIR = orig


if __name__ == "__main__":
    unittest.main()
