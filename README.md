# AI Crime Scene & CCTV Investigation Assistant

Multimodal evidence analysis for CCTV footage, scene photographs and audio: detection and
tracking, speech transcription, a temporal evidence graph, question-driven incident
reconstruction, investigator review and PDF reporting.

AI outputs are investigative leads. Track IDs are anonymous, colour is approximate, and no
relationship in the application asserts identity. Every finding stays *Pending* until an
investigator approves it, and the AI output itself is never edited.

## Run

```powershell
cd C:\CrimeInvestigationAI
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # only needed on a new machine
streamlit run app.py
```

Tests (standard library only, no extra packages):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

## Workflow

1. **Evidence upload**: files are stored unmodified and fingerprinted (SHA-256) at intake,
   then validated and given a unique source ID (`CAM-01`, `PHOTO-01`, `AUDIO-01`).
2. **Settings > Evidence sources**: enter each recording's *verified* start time and
   location. This puts recordings on one clock; without it, recordings are never compared
   in time.
3. **CCTV / Audio / Image analysis**: YOLOv8n with ByteTrack, Faster-Whisper (CPU int8),
   and photo detection.
4. **Event timeline**: group tracked observations into candidate events and review them.
5. **Evidence graph** and **Incident reconstruction**: explore relationships, ask questions,
   and review cross-camera and cross-modal proposals.
6. **Investigator review**, then **Investigation reports**.

## Architecture

```
app.py                      entry point: theme + sidebar navigation
core/
  config.py                 paths, thresholds, logging (outputs/logs/app.log)
  timebase.py               when two timestamps are comparable (shared clock / anchors)
storage/
  db_manager.py             master case database, versioned additive migrations, audit trail
  sqlite_utils.py           connection helper (commit/rollback, always closes)
services/
  media/                    intake (hash, validate, register) and media utilities
  detection/                YOLOv8 + ByteTrack (fresh tracker per video)
  audio/                    Faster-Whisper transcription + time-aligned translation
  events/                   candidate event reconstruction
  correlation/              cross-camera and cross-modal proposals
  evidence_graph/           schema, engine, analysis, appearance features, rendering
  reconstruction/           question parsing and graph-based retrieval
  search/                   keyword search
  reports/                  ReportLab PDF dossier
ui/
  theme.py, components.py   design system
  state.py                  session state and cached evidence graph
  pages/                    one module per navigation page
  views/                    reusable page sections
tests/                      unittest suite with isolated temporary databases
tools/diagnostics/          standalone hardware and upload checks
```

## Evidence graph

A heterogeneous, temporal `networkx.MultiDiGraph`. It is derived from stored data on demand
and rebuilt only when the case data changes, so every edge is reproducible.

**Node types:** Camera, Location, Video Recording, Audio Recording, Image, Person Track,
Vehicle, Personal Belonging, Object, Observation, Transcript Segment, Event.

**Every relationship carries** a confidence, its reasoning (with the thresholds used), its
origin (*Evidence registry*, *Derived rule*, *AI proposal* or *Investigator*), a review
status, its clock basis (*shared-clock* or *anchored*) and the IDs of the records it was
derived from.

| Relationship | Created when |
|---|---|
| CAPTURED_BY, LOCATED_NEAR, DETECTED_IN | registry facts (file to camera, camera to location, observation to file) |
| SAME_TRACK, APPEARED_AT | ByteTrack assigned the same ID within one source |
| ASSOCIATED_WITH_EVENT | observation or track belongs to a reconstructed event |
| TEMPORALLY_OVERLAPS, APPEARED_BEFORE | event intervals overlap, or one is the nearest successor (≤300 s), **on a comparable clock** |
| CROSSED_CAMERA | same class, different cameras, hand-off ≤120 s on anchored clocks; confidence combines timing with colour-histogram similarity of the crops |
| HEARD_DURING, SUPPORTS | speech overlaps visual evidence (±5 s, comparable clock); SUPPORTS also requires the speech to name that object class |
| POSSIBLY_SAME_OBJECT | stored cross-camera or photo proposals, and investigator links |
| CONTRADICTS | location simultaneity, class mismatch, conflicting decisions, impossible transit order, event vs member review conflicts |

**Analysis:** k-hop traversal, strongest evidence path (Dijkstra on 1/confidence), Louvain
communities, betweenness/degree centrality (cameras and files excluded), wall-clock and
relative time windows, and chronological ordering.

## Database and migration

Existing case databases upgrade automatically on first start. The migrations are additive:

* v2 adds `recorded_start`, `location`, `processing_status` and `processed_at` to
  `evidence_sources`, and imports camera time anchors from `outputs/cross_camera_links.db`
  (the old table is left in place).
* v3 adds indexes for relationships and review status.

Back up `database/` and `outputs/*.db` before upgrading if the case matters.

## Hardware notes (RTX 3050 Laptop, 4 GB VRAM)

YOLO runs on CUDA when available, otherwise on the CPU. Whisper always runs on the CPU (int8),
so VRAM stays free for detection. Detection crops, thumbnails and H.264 conversions are
cached, and the evidence graph is rebuilt only when the data fingerprint changes.

## Known limitations

* Colour is estimated from crop pixels (HSV bins) and is lighting-dependent.
  Keyframes recorded before this version have boxes drawn on them; the crop is inset to
  reduce that effect.
* Question answering uses a curated vocabulary (object aliases, colours, cameras, times),
  not a neural language model.
* Cross-camera association uses class, timing and colour histograms only, with no re-identification model.
* Photographs carry no verified capture time and are therefore not placed on the timeline.
