"""Forensic Investigation PDF Report Generator using ReportLab.
Generates structured, professional case reports with:
- Formal case header and investigator metadata
- Digital evidence chain of custody & SHA-256 integrity hashes
- Video sources and camera details
- Verified multimodal observations (CCTV, Photos, Audio)
- Chronological incident reconstruction timeline
- Human-in-the-loop review decisions & audit trail
- Standard forensic investigative disclaimers
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from xml.sax.saxutils import escape as _xml_escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from core.config import ROOT  # noqa: E402
OUTPUT_DIR = ROOT / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OBSERVATION_REGISTER_LIMIT = 60
AUDIT_TRAIL_LIMIT = 40


def esc(value: Any) -> str:
    """Escape text for ReportLab Paragraph markup.

    Paragraph parses an XML-like mini-language, so a raw '&' or '<' in a case title,
    note, filename or transcript made doc.build() raise and no PDF was produced.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return _xml_escape(str(value))


def _fmt_mmss(seconds: Any) -> str:
    try:
        sec = float(seconds)
    except (TypeError, ValueError):
        return "--:--"
    if pd.isna(sec):
        return "--:--"
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def _safe_filename_part(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._")
    return cleaned[:60] or "CASE"


def build_video_analytical_summaries(
    sources: Optional[List[Dict[str, Any]]],
    observations_df: Optional[pd.DataFrame]
) -> List[Dict[str, Any]]:
    """Synthesize deep technical and forensic activity summaries for each CCTV video source."""
    summaries = []
    if not sources:
        return summaries

    for s in sources:
        # This section describes video feeds; photos and audio have their own registers.
        if s.get("modality", "CCTV") != "CCTV":
            continue
        cam_id = s.get("camera_id", "CAM-01")
        sid = s.get("source_id", "")
        name = s.get("source_name", "Footage")
        mod = s.get("modality", "CCTV")
        dur = float(s.get("duration_seconds", 0.0) or 0.0)
        fps = float(s.get("fps", 0.0) or 0.0)
        w = int(s.get("width", 0) or 0)
        h = int(s.get("height", 0) or 0)
        size_mb = float(s.get("file_size_bytes", 0) or 0) / (1024 * 1024)
        hash_val = s.get("sha256_hash", "N/A")

        # Find matching observations
        src_obs = pd.DataFrame()
        if observations_df is not None and not observations_df.empty:
            if "Source ID" in observations_df.columns:
                m1 = observations_df[observations_df["Source ID"] == sid]
                if not m1.empty:
                    src_obs = m1
            if src_obs.empty and "Camera ID" in observations_df.columns:
                m2 = observations_df[observations_df["Camera ID"] == cam_id]
                if not m2.empty:
                    src_obs = m2

        total_obs = len(src_obs)
        class_counts = {}
        unique_tracks = set()
        min_sec, max_sec = None, None
        status_map = {"Approved": 0, "Pending": 0, "Rejected": 0, "Flagged": 0}

        if not src_obs.empty:
            if "Object" in src_obs.columns:
                class_counts = src_obs["Object"].value_counts().to_dict()
            if "Track ID" in src_obs.columns:
                unique_tracks = {t for t in src_obs["Track ID"].dropna().unique() if t not in ("Untracked", "", "nan", "None")}
            if "Timestamp (seconds)" in src_obs.columns:
                v_secs = src_obs["Timestamp (seconds)"].dropna()
                if not v_secs.empty:
                    min_sec = float(v_secs.min())
                    max_sec = float(v_secs.max())
            if "Review Status" in src_obs.columns:
                for st_k, cnt in src_obs["Review Status"].value_counts().items():
                    if st_k in status_map:
                        status_map[st_k] = int(cnt)

        class_summary = ", ".join([f"{c}x {k}" for k, c in class_counts.items()]) if class_counts else "No visual objects detected"
        tracks_summary = f"{len(unique_tracks)} tracking trajectory(s)" if unique_tracks else "Single-frame detections"

        if min_sec is not None and max_sec is not None:
            m_a, s_a = int(min_sec // 60), min_sec % 60
            m_b, s_b = int(max_sec // 60), max_sec % 60
            active_window = f"{m_a:02d}:{s_a:04.1f} to {m_b:02d}:{s_b:04.1f}"
        else:
            active_window = f"00:00.0 to {int(dur // 60):02d}:{dur % 60:04.1f}" if dur > 0 else "N/A"

        res_str = f"{w}x{h}" if w and h else "Standard Definition"
        fps_str = f"{fps:.1f} FPS" if fps > 0 else "Variable FPS"
        dur_str = f"{dur:.1f}s" if dur > 0 else "Static Media"

        narrative = (
            f"<b>Forensic Activity Synopsis:</b> CCTV sensor <b>{esc(cam_id)}</b> recorded {dur_str} of footage ({res_str} @ {fps_str}, {size_mb:.1f} MB). "
            f"Automated YOLOv8 inference generated <b>{total_obs} candidate sightings</b> across the active window <b>{active_window}</b>. "
            f"Classified target breakdown: <i>{esc(class_summary)}</i> ({tracks_summary}). "
            f"Investigator verification audit status: <b>{status_map['Approved']} Approved</b>, "
            f"{status_map['Pending']} Pending, {status_map['Rejected']} Rejected, {status_map['Flagged']} Flagged."
        )

        summaries.append({
            "camera_id": cam_id,
            "source_id": sid,
            "source_name": name,
            "modality": mod,
            "duration_str": dur_str,
            "resolution_str": res_str,
            "fps_str": fps_str,
            "size_mb_str": f"{size_mb:.1f} MB",
            "sha256": hash_val,
            "total_detections": total_obs,
            "class_summary": class_summary,
            "tracks_summary": tracks_summary,
            "active_window": active_window,
            "status_map": status_map,
            "narrative": narrative
        })

    return summaries


def generate_pdf_report(
    case_title: str = "Investigation Case Report",
    case_id: str = "CASE-2026-001",
    investigator_name: str = "Lead Investigator",
    sources: Optional[List[Dict[str, Any]]] = None,
    observations_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
    audit_logs_df: Optional[pd.DataFrame] = None,
    notes: str = "",
    evidence_graph=None,
) -> str:
    """Build and save PDF investigation report. Returns file path string."""
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Case IDs like "CR/2026:01" contain characters Windows forbids in filenames.
    report_filename = f"Forensic_Report_{_safe_filename_part(case_id)}_{timestamp_str}.pdf"
    report_path = REPORTS_DIR / report_filename

    doc = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569")
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white
    )

    table_body_style = ParagraphStyle(
        "TableBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )

    badge_approved = ParagraphStyle(
        "BadgeApproved",
        parent=table_body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#15803d")
    )

    badge_pending = ParagraphStyle(
        "BadgePending",
        parent=table_body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#b45309")
    )

    badge_rejected = ParagraphStyle(
        "BadgeRejected",
        parent=table_body_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#b91c1c")
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b")
    )

    story = []

    # 1. Header & Metadata
    story.append(Paragraph("AI CRIME SCENE & CCTV INVESTIGATION ASSISTANT", title_style))
    story.append(Paragraph("Forensic Evidence Linking & Incident Reconstruction Dossier", subtitle_style))
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#0284c7"), spaceAfter=10))

    meta_data = [
        [
            Paragraph(f"<b>Case Title:</b> {esc(case_title)}", body_style),
            Paragraph(f"<b>Case ID:</b> {esc(case_id)}", body_style)
        ],
        [
            Paragraph(f"<b>Investigating Officer:</b> {esc(investigator_name)}", body_style),
            Paragraph(f"<b>Date/Time:</b> {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}", body_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[90 * mm, 84 * mm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 5 * mm))

    # 2. Executive Summary
    story.append(Paragraph("1. Executive Summary & Scope", section_heading))
    total_sources = len(sources) if sources else 0
    total_obs = len(observations_df) if observations_df is not None and not observations_df.empty else 0

    approved_count = 0
    if observations_df is not None and not observations_df.empty and "Review Status" in observations_df.columns:
        approved_count = (observations_df["Review Status"] == "Approved").sum()

    summary_text = (
        f"This investigative dossier compiles multimodal AI-assisted findings extracted from {total_sources} "
        f"digital evidence source(s). A total of {total_obs} candidate observations were recorded across CCTV recordings, "
        f"photographic evidence, and audio recordings. Of these, {approved_count} item(s) have been verified and approved "
        f"by the investigating officer. All remaining items are maintained with their original provenance and audit history."
    )
    if notes and notes.strip():
        notes_markup = esc(notes.strip()).replace("\n", "<br/>")
        summary_text += f"<br/><br/><b>Investigator Case Notes:</b> {notes_markup}"
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 4 * mm))

    # 3. Evidence Sources & Chain of Custody
    story.append(Paragraph("2. Evidence Sources & Chain of Custody (SHA-256)", section_heading))
    if sources:
        src_table_data = [[
            Paragraph("Camera / Source", table_header_style),
            Paragraph("Modality", table_header_style),
            Paragraph("Duration / Size", table_header_style),
            Paragraph("SHA-256 Integrity Hash", table_header_style)
        ]]
        for s in sources:
            name = s.get("source_name", "Unknown")
            cam = s.get("camera_id", "CAM-01")
            mod = s.get("modality", "CCTV")
            dur = float(s.get("duration_seconds", 0) or 0)
            sz = float(s.get("file_size_bytes", 0) or 0) / (1024 * 1024)
            h = str(s.get("sha256_hash") or "N/A")

            dur_str = f"{dur:.1f}s ({sz:.1f} MB)" if dur > 0 else f"{sz:.1f} MB"

            # The full hash is printed: a truncated hash cannot be used to verify a file.
            src_table_data.append([
                Paragraph(f"<b>{esc(cam)}</b><br/>{esc(name)}", table_body_style),
                Paragraph(esc(mod), table_body_style),
                Paragraph(dur_str, table_body_style),
                Paragraph(f"<font face='Courier' size=6>{esc(h[:32])}<br/>{esc(h[32:])}</font>", table_body_style)
            ])

        src_table = Table(src_table_data, colWidths=[54 * mm, 22 * mm, 30 * mm, 68 * mm])
        src_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(src_table)
    else:
        story.append(Paragraph("No evidence sources recorded in dossier.", body_style))
    story.append(Spacer(1, 4 * mm))

    # 3. CCTV Video Analytical Summary & Incident Activity Synopsis
    story.append(Paragraph("3. CCTV Video Analytical Summary & Incident Activity Synopsis", section_heading))
    video_summaries = build_video_analytical_summaries(sources, observations_df)

    if video_summaries:
        for vid in video_summaries:
            cam = vid["camera_id"]
            name = vid["source_name"]
            vid_header = Paragraph(f"<b>Forensic Sensor: {esc(cam)}</b> - <i>{esc(name)}</i>", table_header_style)
            hash_disp = f"{vid['sha256'][:16]}...{vid['sha256'][-8:]}" if len(vid['sha256']) > 24 else vid['sha256']

            col1_content = [
                Paragraph(f"<b>Modality:</b> {esc(vid['modality'])}", table_body_style),
                Paragraph(f"<b>Duration / Specs:</b> {vid['duration_str']} | {vid['resolution_str']} @ {vid['fps_str']}", table_body_style),
                Paragraph(f"<b>File Size:</b> {vid['size_mb_str']}", table_body_style),
                Paragraph(f"<b>SHA-256 Hash:</b> <font face='Courier' size=6.5>{esc(hash_disp)}</font>", table_body_style),
            ]

            col2_content = [
                Paragraph(f"<b>Total Sightings:</b> {vid['total_detections']} detections", table_body_style),
                Paragraph(f"<b>Observed Classes:</b> {esc(vid['class_summary'])}", table_body_style),
                Paragraph(f"<b>Trajectories:</b> {vid['tracks_summary']}", table_body_style),
                Paragraph(f"<b>Active Window:</b> {vid['active_window']}", table_body_style),
            ]

            card_table = Table(
                [
                    [vid_header, ""],
                    [col1_content, col2_content],
                    [Paragraph(vid["narrative"], body_style), ""]
                ],
                colWidths=[87 * mm, 87 * mm]
            )
            card_table.setStyle(TableStyle([
                ("SPAN", (0, 0), (1, 0)),
                ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#0f172a")),
                ("SPAN", (0, 2), (1, 2)),
                ("BACKGROUND", (0, 1), (1, 1), colors.HexColor("#f8fafc")),
                ("BACKGROUND", (0, 2), (1, 2), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
                ("LINEBELOW", (0, 0), (1, 0), 1, colors.HexColor("#0284c7")),
                ("LINEBELOW", (0, 1), (1, 1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(card_table)
            story.append(Spacer(1, 3.5 * mm))
    else:
        story.append(Paragraph("No CCTV video analytics available.", body_style))
    story.append(Spacer(1, 4 * mm))

    # 4. Verified Observations
    story.append(Paragraph("4. Observation Register & Verification Status", section_heading))
    if observations_df is not None and not observations_df.empty:
        obs_table_data = [[
            Paragraph("Obs ID", table_header_style),
            Paragraph("Cam / Time", table_header_style),
            Paragraph("Object / Modality", table_header_style),
            Paragraph("Observation Details", table_header_style),
            Paragraph("Status", table_header_style)
        ]]

        # Reviewed evidence first (Approved, Flagged, Rejected), then pending items in time order.
        # Previously the first 30 rows were taken silently, so approved evidence could be omitted.
        status_rank = {"Approved": 0, "Flagged": 1, "Rejected": 2, "Pending": 3}
        ranked = observations_df.copy()
        ranked["_rank"] = ranked.get("Review Status", pd.Series(index=ranked.index, dtype=object)).map(status_rank).fillna(4)
        if "Timestamp (seconds)" in ranked.columns:
            ranked["_t"] = pd.to_numeric(ranked["Timestamp (seconds)"], errors="coerce")
            ranked = ranked.sort_values(["_rank", "_t"], na_position="last")
        else:
            ranked = ranked.sort_values("_rank")
        top_obs = ranked.head(OBSERVATION_REGISTER_LIMIT)
        if len(observations_df) > len(top_obs):
            story.append(Paragraph(
                f"Showing {len(top_obs)} of {len(observations_df)} observations (reviewed items first). "
                "The complete register is available as a CSV export from the application.",
                disclaimer_style
            ))
            story.append(Spacer(1, 2 * mm))
        for _, r in top_obs.iterrows():
            oid = str(r.get("Observation ID", ""))[:8]
            cam = str(r.get("Camera ID", "CAM-01"))
            t_str = str(r.get("Timestamp", "00:00.00"))
            obj = str(r.get("Object", "Unknown"))
            mod = str(r.get("Modality", "CCTV"))
            detail = str(r.get("Observation", ""))[:120]
            status = str(r.get("Review Status", "Pending"))

            if status == "Approved":
                st_badge = Paragraph(esc(status), badge_approved)
            elif status == "Rejected":
                st_badge = Paragraph(esc(status), badge_rejected)
            else:
                st_badge = Paragraph(esc(status), badge_pending)

            obs_table_data.append([
                Paragraph(f"<font face='Courier' size=7>{esc(oid)}</font>", table_body_style),
                Paragraph(f"<b>{esc(cam)}</b><br/>{esc(t_str)}", table_body_style),
                Paragraph(f"<b>{esc(obj)}</b><br/>({esc(mod)})", table_body_style),
                Paragraph(esc(detail), table_body_style),
                st_badge
            ])

        obs_table = Table(obs_table_data, colWidths=[20 * mm, 30 * mm, 34 * mm, 65 * mm, 25 * mm])
        obs_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(obs_table)
    else:
        story.append(Paragraph("No observations recorded.", body_style))
    story.append(Spacer(1, 5 * mm))

    # 5. Reconstructed Events Timeline
    if events_df is not None and not events_df.empty:
        story.append(Paragraph("5. Reconstructed Events Timeline", section_heading))
        ev_table_data = [[
            Paragraph("Event Title", table_header_style),
            Paragraph("Source / Camera", table_header_style),
            Paragraph("Time Range", table_header_style),
            Paragraph("Observations", table_header_style),
            Paragraph("Status", table_header_style)
        ]]
        for _, ev in events_df.head(15).iterrows():
            title = str(ev.get("event_title", "Event"))
            src = str(ev.get("source", "CAM-01"))
            cnt = ev.get("observation_count", 0)
            st_ev = str(ev.get("review_status", "Pending"))
            range_str = f"{_fmt_mmss(ev.get('start_seconds'))} - {_fmt_mmss(ev.get('end_seconds'))}"
            ev_badge = badge_approved if st_ev == "Approved" else (badge_rejected if st_ev == "Rejected" else badge_pending)

            ev_table_data.append([
                Paragraph(f"<b>{esc(title)}</b>", table_body_style),
                Paragraph(esc(src), table_body_style),
                Paragraph(range_str, table_body_style),
                Paragraph(esc(cnt), table_body_style),
                Paragraph(esc(st_ev), ev_badge)
            ])

        ev_table = Table(ev_table_data, colWidths=[50 * mm, 34 * mm, 40 * mm, 25 * mm, 25 * mm])
        ev_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ev_table)
        story.append(Spacer(1, 5 * mm))

    def _table(header, rows, widths):
        data = [[Paragraph(esc(h), table_header_style) for h in header]]
        for r in rows:
            data.append([c if isinstance(c, Paragraph) else Paragraph(esc(c), table_body_style) for c in r])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    # 6. Findings: investigator-confirmed vs AI-generated
    story.append(Paragraph("6. Investigator-Confirmed vs AI-Generated Findings", section_heading))
    if observations_df is not None and not observations_df.empty and "Review Status" in observations_df.columns:
        counts = observations_df["Review Status"].value_counts().to_dict()
        story.append(Paragraph(
            f"Investigator-confirmed (Approved): <b>{counts.get('Approved', 0)}</b>. "
            f"AI-generated, not yet reviewed (Pending): <b>{counts.get('Pending', 0)}</b>. "
            f"Flagged for further review: <b>{counts.get('Flagged', 0)}</b>. "
            f"Rejected by investigator: <b>{counts.get('Rejected', 0)}</b>. "
            "Only Approved items should be treated as confirmed observations; all other items are algorithmic "
            "outputs retained for transparency.", body_style))
        confirmed = observations_df[observations_df["Review Status"] == "Approved"]
        if not confirmed.empty:
            story.append(Spacer(1, 2 * mm))
            story.append(_table(
                ["Obs ID", "Camera / time", "Object", "Investigator notes"],
                [[str(r.get("Observation ID", ""))[:12], f"{r.get('Camera ID', '')} {r.get('Timestamp', '')}",
                  str(r.get("Object", "")), str(r.get("Review Notes") or "")[:140]]
                 for _, r in confirmed.head(OBSERVATION_REGISTER_LIMIT).iterrows()],
                [22 * mm, 40 * mm, 30 * mm, 82 * mm]))
    else:
        story.append(Paragraph("No observations recorded.", body_style))
    story.append(Spacer(1, 4 * mm))

    # 7. Audio transcription summary
    if observations_df is not None and not observations_df.empty:
        audio = observations_df[observations_df["Modality"] == "Audio"]
        if not audio.empty:
            story.append(Paragraph("7. Audio Transcription Summary", section_heading))
            story.append(Paragraph(
                f"{len(audio)} speech segment(s) transcribed with Faster-Whisper. The original-language text is "
                "preserved; translations are machine-generated.", body_style))
            story.append(Spacer(1, 2 * mm))
            story.append(_table(
                ["Channel / time", "Lang.", "Original", "English", "Status"],
                [[f"{r.get('Camera ID', '')} {r.get('Timestamp', '')}", str(r.get("Detected Language") or "").upper(),
                  str(r.get("Original Transcript") or "")[:160], str(r.get("English Translation") or "")[:160],
                  str(r.get("Review Status", ""))] for _, r in audio.head(40).iterrows()],
                [32 * mm, 12 * mm, 55 * mm, 55 * mm, 20 * mm]))
            story.append(Spacer(1, 4 * mm))

    # 8-9. Evidence graph: relationships, support, contradictions
    if evidence_graph is not None:
        G = evidence_graph.graph
        summary = evidence_graph.summary()
        story.append(Paragraph("8. Evidence Relationships", section_heading))
        anchored = summary["anchored_sources"]
        story.append(Paragraph(
            f"The evidence graph contains {summary['nodes']} items and {summary['edges']} relationships. "
            f"{anchored} recording(s) have an investigator-verified start time; only those can be compared with "
            "other recordings in time. Every relationship below was produced by a documented rule, a stored AI "
            "proposal or an investigator decision, as stated in the Origin column.", body_style))
        evidential = [(u, v, d) for u, v, d in G.edges(data=True)
                      if d["relation"] not in ("CAPTURED_BY", "DETECTED_IN", "LOCATED_NEAR", "SAME_TRACK",
                                               "ASSOCIATED_WITH_EVENT", "APPEARED_AT")]
        evidential.sort(key=lambda e: (e[2].get("origin") != "Investigator", -(e[2].get("confidence") or 0)))
        if evidential:
            story.append(Spacer(1, 2 * mm))
            story.append(_table(
                ["Relationship", "From", "To", "Conf.", "Origin / status", "Reasoning"],
                [[d["relation"], G.nodes[u].get("label", u), G.nodes[v].get("label", v), f"{d['confidence']:.0%}",
                  f"{d['origin']} / {d['status']}", d["reasoning"][:220]] for u, v, d in evidential[:40]],
                [27 * mm, 26 * mm, 26 * mm, 11 * mm, 26 * mm, 58 * mm]))
            if len(evidential) > 40:
                story.append(Paragraph(f"Showing 40 of {len(evidential)} relationships (investigator links first, then "
                                       "by confidence). Export the full list from the Evidence Graph page.",
                                       disclaimer_style))
        else:
            story.append(Paragraph("No cross-item relationships beyond tracking and event membership.", body_style))
        story.append(Spacer(1, 4 * mm))

        story.append(Paragraph("9. Supporting and Contradicting Evidence", section_heading))
        supports = [(u, v, d) for u, v, d in G.edges(data=True) if d["relation"] in ("SUPPORTS", "HEARD_DURING")]
        if supports:
            story.append(_table(
                ["Evidence", "Supports", "Conf.", "Basis"],
                [[G.nodes[u].get("label", u), G.nodes[v].get("label", v), f"{d['confidence']:.0%}", d["reasoning"][:200]]
                 for u, v, d in supports[:30]],
                [40 * mm, 40 * mm, 12 * mm, 82 * mm]))
        else:
            story.append(Paragraph("No supporting relationships were found by the current rules.", body_style))
        story.append(Spacer(1, 2 * mm))
        if evidence_graph.contradictions:
            story.append(_table(
                ["Kind", "Severity", "Between", "Explanation"],
                [[c["kind"], c["severity"], " / ".join(G.nodes[n].get("label", n) for n in c["nodes"]), c["reasoning"]]
                 for c in evidence_graph.contradictions[:30]],
                [28 * mm, 16 * mm, 45 * mm, 85 * mm]))
        else:
            story.append(Paragraph("No contradictions were detected (location simultaneity, class mismatch, "
                                   "conflicting decisions, impossible transit order, event/observation review "
                                   "conflicts).", body_style))
        story.append(Spacer(1, 4 * mm))

    # 10. Investigator audit trail
    if audit_logs_df is not None and not audit_logs_df.empty:
        story.append(Paragraph("10. Investigator Audit Trail (most recent first)", section_heading))
        audit_rows = [[
            Paragraph("Time", table_header_style),
            Paragraph("Action", table_header_style),
            Paragraph("Entity", table_header_style),
            Paragraph("Details", table_header_style),
        ]]
        for _, log in audit_logs_df.head(AUDIT_TRAIL_LIMIT).iterrows():
            audit_rows.append([
                Paragraph(esc(log.get("timestamp", "")), table_body_style),
                Paragraph(esc(log.get("action", "")), table_body_style),
                Paragraph(f"{esc(log.get('entity_type', ''))}<br/><font face='Courier' size=6.5>{esc(log.get('entity_id', ''))}</font>", table_body_style),
                Paragraph(esc(str(log.get("details", "") or "")[:160]), table_body_style),
            ])
        audit_table = Table(audit_rows, colWidths=[34 * mm, 40 * mm, 34 * mm, 66 * mm], repeatRows=1)
        audit_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(audit_table)
        if len(audit_logs_df) > AUDIT_TRAIL_LIMIT:
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(
                f"Showing the {AUDIT_TRAIL_LIMIT} most recent of {len(audit_logs_df)} audit entries. "
                "Export the full trail as CSV from the Verification Centre.",
                disclaimer_style
            ))
        story.append(Spacer(1, 5 * mm))

    # 7. Forensic Chain of Custody & Disclaimers
    story.append(Spacer(1, 4 * mm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
    story.append(Paragraph(
        "<b>FORENSIC NOTICE & INTEGRITY STATEMENT:</b><br/>"
        "1. This document is generated by the AI Crime Scene & CCTV Investigation Assistant to assist in evidence linking and incident reconstruction.<br/>"
        "2. All AI object detections, track IDs, audio transcriptions, and candidate events are initial algorithmic proposals requiring human verification.<br/>"
        "3. Object classification confidence scores represent statistical model outputs and do not constitute legal proof of identity or presence.<br/>"
        "4. Original digital video files, hashes, and audit histories are preserved in secure custody.",
        disclaimer_style
    ))

    generated_at = datetime.now().strftime("%d-%b-%Y %H:%M")

    def _page_footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(18 * mm, 10 * mm, f"Case {str(case_id)[:40]} | Generated {generated_at} | AI findings require human verification")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    # Build document
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return str(report_path)

