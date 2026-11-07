"""Create a small, clearly labelled synthetic dataset to test the graph core.

Run from the project virtual environment:
    python demo_evidence_graph.py
It creates demo_evidence.db in the current working directory. It does not touch app.py.
"""
from pathlib import Path
from evidence_graph_core import EvidenceStore

store = EvidenceStore(Path("demo_evidence.db"))
case_id = store.create_case(
    "Synthetic red-backpack investigation",
    "Demonstration data only; all names and times are fictional.",
    case_id="case_red_backpack_demo",
)

store.add_source(case_id=case_id, source_id="src_cctv_01", source_type="cctv",
                 file_name="CCTV_01_demo.mp4", file_path="demo/CCTV_01_demo.mp4",
                 camera_label="Camera 01", location_label="Demo entrance",
                 clock_offset_seconds=0, clock_alignment_verified=True)
store.add_source(case_id=case_id, source_id="src_cctv_02", source_type="cctv",
                 file_name="CCTV_02_demo.mp4", file_path="demo/CCTV_02_demo.mp4",
                 camera_label="Camera 02", location_label="Demo corridor",
                 clock_offset_seconds=0, clock_alignment_verified=True)
store.add_source(case_id=case_id, source_id="src_audio_01", source_type="audio",
                 file_name="Audio_01_demo.wav", file_path="demo/Audio_01_demo.wav",
                 clock_offset_seconds=0, clock_alignment_verified=True)
store.add_source(case_id=case_id, source_id="src_photo_01", source_type="photo",
                 file_name="Photo_01_demo.jpg", file_path="demo/Photo_01_demo.jpg")

obs_1 = store.add_observation(
    case_id=case_id, source_id="src_cctv_01", observation_id="obs_cctv01_bag",
    observation_type="object_detection", label="backpack", confidence=0.91,
    source_start_seconds=1880.0, track_local_id="CCTV01_track_7",
    attributes={"color": "red", "pattern": "plain"},
    metadata={"frame_number": 47000, "note": "Synthetic demo observation"},
)
obs_2 = store.add_observation(
    case_id=case_id, source_id="src_cctv_02", observation_id="obs_cctv02_bag",
    observation_type="object_detection", label="backpack", confidence=0.87,
    source_start_seconds=2050.0, track_local_id="CCTV02_track_3",
    attributes={"color": "red", "pattern": "plain"},
    metadata={"frame_number": 51250, "note": "Synthetic demo observation"},
)
obs_3 = store.add_observation(
    case_id=case_id, source_id="src_audio_01", observation_id="obs_audio_bag",
    observation_type="transcript_segment", label="speech segment",
    source_start_seconds=2055.0, source_end_seconds=2060.0,
    original_text="Please move the red backpack near the entrance.",
    translated_text="", metadata={"language": "en", "synthetic": True},
)
obs_4 = store.add_observation(
    case_id=case_id, source_id="src_photo_01", observation_id="obs_photo_bag",
    observation_type="photo_detection", label="backpack", confidence=0.83,
    source_start_seconds=None,
    attributes={"color": "red", "pattern": "plain"},
    metadata={"note": "Photo time is unknown; no timeline alignment assumed."},
)

new_links = store.propose_relationships(case_id, temporal_window_seconds=120)
print(f"Case: {case_id}")
print(f"Observations: {len(store.list_observations(case_id))}")
print(f"New candidate relationships: {len(new_links)}")
for link in new_links:
    print(f"  {link['type']}: {link['relationship_id']}")
print("Graph payload:", {k: len(v) for k, v in store.graph_payload(case_id).items()})
print("\nThis database contains synthetic demo data only. It does not analyse real files.")
