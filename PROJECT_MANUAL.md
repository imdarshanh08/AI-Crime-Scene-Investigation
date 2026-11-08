# 🔍 AI Crime Scene & CCTV Investigation Assistant (TRACE-AI)
**Multimodal AI for Evidence Linking and Incident Reconstruction**

---

## 🚀 Quick Start (Windows 11)

### 1. Launch the Application
Open PowerShell in `C:\CrimeInvestigationAI` and run:
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 🛠️ System Architecture & Hardware Optimization

- **Host Machine**: ASUS TUF A15 (AMD Ryzen 7, 16 GB RAM, Windows 11)
- **GPU Acceleration**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Computer Vision Pipeline**: YOLOv8 Nano (`yolov8n.pt`) with ByteTrack tracking
  - Memory-safe inference with periodic `torch.cuda.empty_cache()`
  - Native **H.264 (yuv420p)** encoding via PyAV for instant HTML5 browser playback
- **Audio Intelligence Pipeline**: Faster-Whisper (`base` model) on CPU `int8`
  - Preserves 100% of GPU VRAM exclusively for vision tasks
  - Native PyAV extraction of audio tracks from CCTV video containers
- **Persistence Layer**: SQLite WAL (`database/investigation_master.db`)
  - Survives application restarts, browser reloads, and system reboots
- **Knowledge Graph**: NetworkX + PyVis interactive HTML visualization
- **Reporting Engine**: ReportLab forensic PDF dossier generator

---

## 📑 The 13 Forensic Workflow Sections

| # | Section / Tab | Functionality |
|---|---|---|
| **0** | **🎬 TRACE-AI Workstation** | **Dual-Pane Live Investigation Dashboard**: 2×2 Multi-Camera CCTV Grid on the left + Live Interactive Evidence Graph, Incident Flow, and Verification Action Bar (`[CONFIRMED] [NEEDS REVIEW] [INCORRECT]`) on the right. |
| **1** | **📊 Dashboard** | Case metrics overview, digital chain of custody with SHA-256 integrity hashes, modality breakdown, investigator audit logs, and **Live Cryptographic Evidence Integrity Audit Tool** (recomputes streaming SHA-256 directly from disk to detect tampering or bit rot). |
| **2** | **📹 CCTV Upload** | Multi-video streaming upload (MP4, AVI, MOV, MKV), OpenCV metadata extraction (duration, FPS, resolution), unique Camera ID assignment (`CAM-01`, `CAM-02`), and thumbnail generation. |
| **3** | **🤖 Video Detection** | YOLOv8 + ByteTrack multi-object tracking, confidence/interval sliders, class filters, interactive H.264 annotated video playback, and keyframe snapshots. |
| **4** | **📷 Photo Evidence** | Crime scene photograph upload, object/evidence detection, and annotated visual evidence archives. |
| **5** | **🎙️ Audio Intelligence** | Dual-input audio engine: extracts audio from CCTV recordings or accepts standalone 911/dispatch recordings. Generates timestamped original transcripts and English translations. |
| **6** | **🗄️ Evidence Repository** | Consolidated, filterable database of all observations across all modalities with CSV export. |
| **7** | **⏱️ Event Reconstruction** | **Dual Subtabs**: (1) Candidate Event Generator grouping consecutive sightings by time proximity and tracking IDs; (2) **Unified Chronological Evidence Timeline** with linked image snapshots and audio playback. |
| **8** | **🌐 Cross-Camera Linking** | **Triple Subtabs**: (1) Cross-Camera Sighting Correlation with camera time anchors; (2) **Audio-Visual Cross-Modal Correlation** linking speech keywords to visual object sightings; (3) **Side-by-Side CCTV Sighting Comparison & Forensic Linker** (renders zoomed subject crops, full context frames, Δt transit time calculation, and one-click knowledge graph relationship recording). |
| **9** | **🕸️ Evidence Graph** | **Advanced Multimodal Knowledge Graph** featuring: (1) Circular nodes embedding actual detection crop photographs; (2) Live floating glassmorphism forensic HUD inspector; (3) 3 Layout Perspectives (Force-Directed, Left-to-Right Timeline, Hierarchy Tree); (4) Spatio-temporal transit trajectories with $\Delta t$; (5) Real-time integration of investigator-confirmed links from the database; (6) NetworkX Centrality Analytics (Key Evidence Hub & Critical Bridge); (7) Standalone HTML and GraphML exports. |
| **10** | **🔍 Investigation Search** | Natural language and keyword search engine supporting queries like *"Show all detections from CAM-03"* or *"Find speech mentioning backpack"*. |
| **11** | **🧑‍⚖️ Verification Centre** | Dedicated human-in-the-loop review workstation for approving, rejecting, or flagging AI findings with immutable audit logging and **Formal Audit Trail CSV Export**. |
| **12** | **📄 Reports & Export** | Generates formal, publication-ready Forensic Investigation PDF Dossiers with **Section 3: CCTV Video Analytical Summaries & Incident Activity Synopses**, chain of custody hashes, observations table, event timeline, and legal disclaimers. |

---

## 🎯 Review & Demonstration Walkthrough

When presenting this project to an evaluator or review panel:

1. **Step 1: Open Tab 0 (`🎬 TRACE-AI Workstation`)**
   - Highlight the split-screen design matching real-world police command centers.
   - Point out the **Multi-Camera Matrix** on the left with camera feeds and target tracking tags.
   - Show the **Live Evidence Graph** on the right with circular image nodes and the **Chronological Incident Flow** list underneath.
   - Demonstrate clicking **`[✅ CONFIRMED]`** on an observation to show instant human-in-the-loop verification.

2. **Step 2: Open Tab 1 (`📊 Dashboard`)**
   - Click **`🔐 Run Cryptographic Evidence Integrity Audit`**.
   - Show the live streaming SHA-256 calculation across all files on disk, verifying 100% hash consistency with legal chain of custody requirements.

3. **Step 3: Open Tab 8 (`🌐 Cross-Camera Linking`) -> Subtab 3 (`👥 Side-by-Side Sighting Comparison`)**
   - Select Sighting A from Camera 1 and Sighting B from Camera 2.
   - Show the simultaneous rendering of the zoomed subject crop and full frame context.
   - Point out the automatic $\Delta t$ calculation (e.g. $+12.50\text{s}$ transit time) and transit plausibility check.
   - Click **`💾 Record Forensic Link in Knowledge Graph`** to establish a formal correlation.

4. **Step 4: Open Tab 9 (`🕸️ Evidence Graph`)**
   - Hover or click on the linked sightings to display the glassmorphism HUD card with detection details.
   - Switch layout to **Timeline (Left-to-Right)** to display chronological sequence.
   - Show the investigator-confirmed relationship line dynamically rendered in emerald green.
   - Click **Download Interactive HTML Graph** for court-ready offline evidence inspection.

5. **Step 5: Open Tab 10 (`🔍 Investigation Search`)**
   - Type: `"Show all detections from CAM-05"` or `"backpack"` to demonstrate intent parsing, camera filtering, and evidence retrieval.

6. **Step 6: Open Tab 11 (`🧑‍⚖️ Verification Centre`)**
   - Show the interactive table editor for human-in-the-loop approval, rejection, and investigator notes.
   - Click **`📥 Download Formal Forensic Audit Trail (CSV)`** to export the timestamped chain of custody.

7. **Step 7: Open Tab 12 (`📄 Reports & Export`)**
   - Enter your case reference number and click **Generate Forensic PDF Report**.
   - Download and open the generated PDF to show the formal chain of custody, SHA-256 hashes, and verified evidence register.

