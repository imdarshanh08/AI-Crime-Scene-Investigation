import sqlite3
import unittest

from tests.helpers import TempDatabaseCase, obs
from storage import db_manager


class StorageTests(TempDatabaseCase):
    def _register(self, sid="aaaaaaaaaaaa", camera=None):
        db_manager.save_source_record(sid, f"{sid}.mp4", f"/tmp/{sid}.mp4", "CCTV",
                                      camera_id=camera or db_manager.next_camera_id(), sha256_hash="ab" * 32)

    def test_writes_are_committed(self):
        """Regression: a refactor once dropped the commit, silently discarding every write."""
        self._register()
        raw = sqlite3.connect(str(db_manager.MASTER_DB_PATH))
        try:
            self.assertEqual(raw.execute("SELECT COUNT(*) FROM evidence_sources").fetchone()[0], 1)
        finally:
            raw.close()

    def test_failed_transaction_rolls_back(self):
        with self.assertRaises(RuntimeError):
            with db_manager.get_db_connection() as conn:
                conn.execute("INSERT INTO audit_logs (action, entity_type, entity_id, details, timestamp) "
                             "VALUES ('x', 'y', 'z', '', 'now')")
                raise RuntimeError("boom")
        self.assertEqual(len(db_manager.get_audit_logs(None)), 0)

    def test_migrations_applied_and_idempotent(self):
        self.assertEqual(db_manager.get_schema_version(), 3)
        db_manager.init_database()  # second run must be a no-op
        self.assertEqual(db_manager.get_schema_version(), 3)
        self._register()
        src = db_manager.get_all_sources()[0]
        for col in ("recorded_start", "location", "processing_status"):
            self.assertIn(col, src)

    def test_camera_ids_are_unique_and_fill_gaps(self):
        self._register("aaaaaaaaaaaa")
        self._register("bbbbbbbbbbbb")
        self.assertEqual(sorted(s["camera_id"] for s in db_manager.get_all_sources()), ["CAM-01", "CAM-02"])
        db_manager.delete_source_record("aaaaaaaaaaaa")
        self.assertEqual(db_manager.next_camera_id(), "CAM-01")
        self.assertEqual(db_manager.next_camera_id("AUDIO"), "AUDIO-01")

    def test_review_keeps_ai_output_and_audits(self):
        self._register()
        db_manager.save_observations_batch([obs("o1", "aaaaaaaaaaaa", "CAM-01", 1.0, conf=0.61)])
        db_manager.update_observation_review("o1", "Approved", "checked frame")
        row = db_manager.get_all_observations_df().iloc[0]
        self.assertEqual(row["Review Status"], "Approved")
        self.assertAlmostEqual(row["Confidence"], 0.61)  # AI output untouched
        self.assertTrue(any("Pending -> Approved" in a for a in db_manager.get_audit_logs(None)["action"]))
        with self.assertRaises(ValueError):
            db_manager.update_observation_review("o1", "Maybe", "")

    def test_source_metadata_validation_and_audit(self):
        self._register()
        db_manager.update_source_metadata("aaaaaaaaaaaa", location="North gate", recorded_start="2026-11-06 14:30:00")
        src = db_manager.get_source_by_id("aaaaaaaaaaaa")
        self.assertEqual((src["location"], src["recorded_start"]), ("North gate", "2026-11-06 14:30:00"))
        with self.assertRaises(ValueError):
            db_manager.update_source_metadata("aaaaaaaaaaaa", recorded_start="06/11/2026 2pm")
        self.assertTrue(any(a == "Source metadata updated" for a in db_manager.get_audit_logs(None)["action"]))

    def test_delete_removes_observations_links_and_is_audited(self):
        self._register()
        db_manager.save_observations_batch([obs("o1", "aaaaaaaaaaaa", "CAM-01", 1.0),
                                            obs("o2", "aaaaaaaaaaaa", "CAM-01", 2.0)])
        db_manager.record_graph_relationship("o1", "o2", "SAME_ENTITY_CANDIDATE")
        db_manager.delete_source_record("aaaaaaaaaaaa")
        self.assertTrue(db_manager.get_all_observations_df().empty)
        self.assertTrue(db_manager.get_graph_relationships().empty)
        self.assertIn("Evidence source removed", set(db_manager.get_audit_logs(None)["action"]))


if __name__ == "__main__":
    unittest.main()
