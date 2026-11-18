"""Side-by-side comparison of two CCTV sightings, with an option to record an investigator link."""

import os

import cv2
import pandas as pd
import streamlit as st

from storage import db_manager
from ui.state import (
    clean_text as _clean_text,
    fmt_pct as _fmt_pct,
    fmt_secs as _fmt_secs,
)


def render_sighting_comparison():
    st.subheader("Compare two sightings")
    st.write(
        "Compare two candidate CCTV sightings side-by-side to inspect visual characteristics, "
        "spatio-temporal transit time (Δt), and establish or reject forensic relationships in the master knowledge graph."
    )

    cctv_obs = db_manager.get_all_observations_df(modality="CCTV")
    if cctv_obs.empty or len(cctv_obs) < 2:
        st.info("At least two CCTV object sightings are required to perform a side-by-side forensic comparison.")
    else:
        avail_classes = ["All Classes"] + sorted(list(cctv_obs["Object"].dropna().unique()))
        col_filter, _ = st.columns([2, 3])
        with col_filter:
            selected_class = st.selectbox(
                "Filter candidate sightings by Object Class:",
                avail_classes,
                key="sb_sighting_class_filter"
            )

        filtered_cctv = cctv_obs if selected_class == "All Classes" else cctv_obs[cctv_obs["Object"] == selected_class]

        if filtered_cctv.empty or len(filtered_cctv) < 2:
            st.warning(f"Not enough sightings found for class '{selected_class}'. Try 'All Classes'.")
        else:
            def format_sighting_label(r):
                oid = str(r["Observation ID"])[:8]
                cam = r.get("Camera ID", "CAM-01")
                ts = r.get("Timestamp", "00:00.00")
                obj = r.get("Object", "Target")
                conf = r.get("Confidence")
                conf_str = f"{conf:.0%}" if pd.notna(conf) else "N/A"
                track = r.get("Track ID", "Untracked")
                return f"[{oid}] {cam} @ {ts} | {obj} ({conf_str}, {track})"

            options_records = filtered_cctv.to_dict("records")
            labels = [format_sighting_label(r) for r in options_records]
            record_lookup = {labels[i]: options_records[i] for i in range(len(options_records))}

            col_pick_a, col_pick_b = st.columns(2)
            with col_pick_a:
                choice_a = st.selectbox(
                    "🔍 Select Sighting A (Origin Observation):",
                    labels,
                    index=0,
                    key="sb_choice_sighting_a"
                )
            with col_pick_b:
                choice_b = st.selectbox(
                    "🎯 Select Sighting B (Candidate Transit / Target Observation):",
                    labels,
                    index=1 if len(labels) > 1 else 0,
                    key="sb_choice_sighting_b"
                )

            rec_a = record_lookup[choice_a]
            rec_b = record_lookup[choice_b]
            same_sighting = str(rec_a["Observation ID"]) == str(rec_b["Observation ID"])
            if same_sighting:
                st.warning("Sighting A and Sighting B are the same observation. Pick two different sightings to compare.")

            # Helper to render bounding box and crop
            import ast

            def render_sighting_crop(evidence_path, bbox_str, label_text):
                if not evidence_path or not os.path.exists(evidence_path):
                    return None, None
                bgr = cv2.imread(evidence_path)
                if bgr is None:
                    return None, None
                crop_rgb = None
                if isinstance(bbox_str, str) and bbox_str.strip():
                    try:
                        coords = ast.literal_eval(bbox_str)
                        if isinstance(coords, (list, tuple)) and len(coords) == 4:
                            h, w = bgr.shape[:2]
                            x1 = max(0, int(round(float(coords[0]))))
                            y1 = max(0, int(round(float(coords[1]))))
                            x2 = min(w, int(round(float(coords[2]))))
                            y2 = min(h, int(round(float(coords[3]))))
                            if x2 > x1 and y2 > y1:
                                crop_rgb = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 3)
                            if label_text:
                                cv2.putText(bgr, label_text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    except Exception:
                        pass
                full_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return full_rgb, crop_rgb

            st.divider()

            # Visual Comparison Columns
            col_view_a, col_view_b = st.columns(2)

            with col_view_a:
                st.markdown(f"**Sighting A:** {rec_a['Camera ID']} ({rec_a['Timestamp']})")
                full_a, crop_a = render_sighting_crop(rec_a.get("Evidence Path", ""), rec_a.get("Bounding Box", ""), f"{rec_a['Object']} ({rec_a.get('Track ID', '')})")

                sub_col_crop, sub_col_full = st.columns([1, 2])
                with sub_col_crop:
                    if crop_a is not None:
                        st.caption("Zoomed Subject Crop")
                        st.image(crop_a, width="stretch")
                    else:
                        st.caption("Crop not available")
                with sub_col_full:
                    if full_a is not None:
                        st.caption("Full Frame Context")
                        st.image(full_a, width="stretch")
                    else:
                        st.info("Full frame not found on disk.")

                st.markdown(f"""
                - **Observation ID**: `{rec_a['Observation ID']}`
                - **Camera**: `{rec_a['Camera ID']}` | **Track**: `{rec_a.get('Track ID', 'N/A')}`
                - **Detected Object**: `{rec_a['Object']}` (Confidence: `{_fmt_pct(rec_a.get('Confidence'))}`)
                - **Timestamp**: `{rec_a['Timestamp']}` (`{_fmt_secs(rec_a.get('Timestamp (seconds)'))}`)
                - **Review Status**: `{rec_a.get('Review Status', 'Pending')}`
                """)

            with col_view_b:
                st.markdown(f"**Sighting B:** {rec_b['Camera ID']} ({rec_b['Timestamp']})")
                full_b, crop_b = render_sighting_crop(rec_b.get("Evidence Path", ""), rec_b.get("Bounding Box", ""), f"{rec_b['Object']} ({rec_b.get('Track ID', '')})")

                sub_col_crop_b, sub_col_full_b = st.columns([1, 2])
                with sub_col_crop_b:
                    if crop_b is not None:
                        st.caption("Zoomed Subject Crop")
                        st.image(crop_b, width="stretch")
                    else:
                        st.caption("Crop not available")
                with sub_col_full_b:
                    if full_b is not None:
                        st.caption("Full Frame Context")
                        st.image(full_b, width="stretch")
                    else:
                        st.info("Full frame not found on disk.")

                st.markdown(f"""
                - **Observation ID**: `{rec_b['Observation ID']}`
                - **Camera**: `{rec_b['Camera ID']}` | **Track**: `{rec_b.get('Track ID', 'N/A')}`
                - **Detected Object**: `{rec_b['Object']}` (Confidence: `{_fmt_pct(rec_b.get('Confidence'))}`)
                - **Timestamp**: `{rec_b['Timestamp']}` (`{_fmt_secs(rec_b.get('Timestamp (seconds)'))}`)
                - **Review Status**: `{rec_b.get('Review Status', 'Pending')}`
                """)

            # Spatio-Temporal Transit Analysis Card
            st.divider()
            st.subheader("⏱️ Spatio-Temporal Transit & Correlation Analysis")

            sec_a = pd.to_numeric(rec_a.get("Timestamp (seconds)"), errors="coerce")
            sec_b = pd.to_numeric(rec_b.get("Timestamp (seconds)"), errors="coerce")
            sec_a = 0.0 if pd.isna(sec_a) else float(sec_a)
            sec_b = 0.0 if pd.isna(sec_b) else float(sec_b)
            delta_t = abs(sec_b - sec_a)

            if sec_a < sec_b:
                seq_text = f"Sighting A ({rec_a['Camera ID']} @ {rec_a['Timestamp']}) preceded Sighting B ({rec_b['Camera ID']} @ {rec_b['Timestamp']}) by **+{delta_t:.2f} seconds**."
            elif sec_b < sec_a:
                seq_text = f"Sighting B ({rec_b['Camera ID']} @ {rec_b['Timestamp']}) preceded Sighting A ({rec_a['Camera ID']} @ {rec_a['Timestamp']}) by **+{delta_t:.2f} seconds**."
            else:
                seq_text = f"Both sightings occurred simultaneously at **{rec_a['Timestamp']}** (Δt = 0.00s)."

            # Transit plausibility assessment
            if delta_t < 4.0:
                transit_eval = "🟢 **Simultaneous / Direct Overlap**: Sightings are within 4 seconds. Highly consistent with overlapping camera fields of view or adjacent angles."
            elif delta_t <= 90.0:
                transit_eval = f"🟢 **Plausible Pedestrian Transit Window**: Transit time of {delta_t:.1f}s is anatomically and geographically consistent with walking or running between adjacent camera zones."
            elif delta_t <= 300.0:
                transit_eval = f"🟡 **Extended Transit Gap**: Transit time of {delta_t:.1f}s indicates possible dwell time, stops, or indirect path between camera zones."
            else:
                transit_eval = f"⚪ **Long Temporal Separation**: Δt = {delta_t:.1f}s (> 5 minutes). High probability of independent incidents unless stationary dwell is verified."

            match_eval = "✅ **Semantic Match**: Both detections share identical object classification." if _clean_text(rec_a["Object"]).lower() == _clean_text(rec_b["Object"]).lower() else f"⚠️ **Class Divergence**: Sightings classified as '{rec_a['Object']}' vs '{rec_b['Object']}'."

            card_col1, card_col2 = st.columns([3, 2])
            with card_col1:
                st.info(f"**Chronological Sequence**: {seq_text}\n\n**Transit Plausibility**: {transit_eval}\n\n**Classification**: {match_eval}")
            with card_col2:
                st.warning(
                    "⚠️ **Forensic Notice (Investigator Responsibility)**:\n\n"
                    "Computer vision similarity and transit proximity do **NOT** constitute definitive biometric proof of identity. "
                    "Forensic investigators must verify facial features, unique attire, luggage markings, and continuous route custody."
                )

            # Action Bar to Record Link into Knowledge Graph
            st.subheader("🔗 Record Forensic Relationship in Knowledge Graph")
            col_rel_type, col_rel_stat = st.columns([2, 1.5])

            with col_rel_type:
                rel_type = st.selectbox(
                    "Relationship Type:",
                    [
                        "TRANSIT_PATH (+Δt)",
                        "SAME_ENTITY_CANDIDATE",
                        "CO_OCCURRENCE",
                        "SUSPICIOUS_PROXIMITY",
                        "DISTINCT_ENTITIES (Rule Out)"
                    ],
                    key="sb_link_rel_type"
                )

            with col_rel_stat:
                rel_status = st.selectbox(
                    "Decision Status:",
                    ["Approved", "Flagged", "Pending", "Rejected"],
                    key="sb_link_rel_status"
                )

            link_notes = st.text_input(
                "Investigator Notes / Rationale:",
                value=f"Correlated {rec_a['Camera ID']} and {rec_b['Camera ID']} sightings with Δt = {delta_t:.1f}s",
                key="txt_link_notes"
            )

            if st.button("💾 Record Forensic Link in Knowledge Graph", type="primary", key="btn_save_sighting_link", disabled=same_sighting):
                rel_id = db_manager.record_graph_relationship(
                    source_observation=str(rec_a["Observation ID"]),
                    target_observation=str(rec_b["Observation ID"]),
                    relationship_type=rel_type,
                    explanation=f"{seq_text}. {link_notes}",
                    confidence=0.85 if rel_status == "Approved" else 0.5,
                    status=rel_status,
                    origin="Investigator Sighting Comparison",
                    reviewer_notes=link_notes
                )
                st.toast(f"Forensic link {rel_id} recorded. It now appears in the Evidence Graph tab.")
                st.rerun()
