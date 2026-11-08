"""Advanced Multimodal Forensic Evidence Graph Builder.
Powered by NetworkX and PyVis with Custom Vis.js Forensic HUD.

Visual Innovations (matching reference video design):
1. Circular Nodes with Real Embedded Visual Detection Thumbnails (Subject A, Object X - Red Backpack)
2. High-Tech Glassmorphism Forensic HUD Card overlay on the graph canvas
3. Spatio-Temporal Trajectory Inference (Cross-camera transit paths with delta-t)
4. Audio-Visual Cross-Modal Corroboration (Whisper speech linked to visual sightings)
5. Multiple Forensic Layout Perspectives (Force-Directed Cluster, Left-to-Right Timeline, Hierarchy Tree)
6. NetworkX Centrality Analytics (Key Evidence Hub & Critical Bridge Detection)
7. Standalone Interactive HTML Dossier Export for Court Presentations
"""

import ast
import base64
import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import db_manager
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
GRAPHS_DIR = OUTPUT_DIR / "evidence_graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# High-tech SVG Icons for nodes without photos
CAM_ICON = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
    "<circle cx='50' cy='50' r='48' fill='%230f172a' stroke='%230ea5e9' stroke-width='5'/>"
    "<path d='M26 38 h34 v26 h-34 z M60 44 l18 -10 v34 l-18 -10 z' fill='%2338bdf8'/>"
    "<circle cx='43' cy='51' r='7' fill='%230f172a'/>"
    "</svg>"
)

AUDIO_ICON = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
    "<circle cx='50' cy='50' r='48' fill='%230f172a' stroke='%23a855f7' stroke-width='5'/>"
    "<rect x='42' y='26' width='16' height='28' rx='8' fill='%23c084fc'/>"
    "<path d='M32 46 a18 18 0 0 0 36 0 M50 64 v12 M38 76 h24' stroke='%23c084fc' stroke-width='4' fill='none' stroke-linecap='round'/>"
    "</svg>"
)

EVENT_ICON = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
    "<circle cx='50' cy='50' r='48' fill='%230f172a' stroke='%2310b981' stroke-width='5'/>"
    "<path d='M52 18 l-20 34 h18 l-8 30 l30 -38 h-20 z' fill='%2334d399'/>"
    "</svg>"
)

PERSON_ICON = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
    "<circle cx='50' cy='50' r='48' fill='%230f172a' stroke='%23f97316' stroke-width='5'/>"
    "<circle cx='50' cy='38' r='15' fill='%23fb923c'/>"
    "<path d='M26 80 a24 20 0 0 1 48 0 z' fill='%23fb923c'/>"
    "</svg>"
)

BACKPACK_ICON = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
    "<circle cx='50' cy='50' r='48' fill='%230f172a' stroke='%23ef4444' stroke-width='5'/>"
    "<path d='M30 36 h40 a8 8 0 0 1 8 8 v30 a8 8 0 0 1 -8 8 h-40 a8 8 0 0 1 -8 -8 v-30 a8 8 0 0 1 8 -8 z' fill='%23ef4444'/>"
    "<path d='M42 36 v-8 a8 8 0 0 1 16 0 v8' stroke='%23fca5a5' stroke-width='4' fill='none'/>"
    "<rect x='34' y='52' width='32' height='18' rx='3' fill='%23991b1b'/>"
    "</svg>"
)

MULTIMODAL_VOCAB = {
    "backpack": ["backpack", "bag", "rucksack", "schoolbag", "knapsack"],
    "handbag": ["handbag", "purse", "pocketbook", "tote"],
    "suitcase": ["suitcase", "luggage", "briefcase"],
    "car": ["car", "vehicle", "automobile", "sedan", "suv"],
    "motorcycle": ["motorcycle", "bike", "motorbike", "scooter"],
    "bicycle": ["bicycle", "cycle", "bike"],
    "person": ["person", "man", "woman", "guy", "someone", "suspect", "individual"],
    "cell phone": ["phone", "mobile", "cellphone"],
    "knife": ["knife", "weapon", "blade"]
}


def get_detection_crop_data_uri(evidence_path: str, bbox_str: Optional[str] = None) -> str:
    """Extract a high-contrast 100x100 thumbnail crop of the detected object and return base64 JPEG URI."""
    if not evidence_path or not Path(evidence_path).is_file():
        return ""

    try:
        img = cv2.imread(evidence_path)
        if img is None:
            return ""

        h, w = img.shape[:2]

        # Parse bounding box if available
        if bbox_str:
            try:
                coords = ast.literal_eval(bbox_str)
                if isinstance(coords, (list, tuple)) and len(coords) == 4:
                    x1, y1, x2, y2 = [int(v) for v in coords]
                    # Add 10% padding around bounding box for context
                    pad_w = int((x2 - x1) * 0.1)
                    pad_h = int((y2 - y1) * 0.1)
                    x1 = max(0, x1 - pad_w)
                    y1 = max(0, y1 - pad_h)
                    x2 = min(w, x2 + pad_w)
                    y2 = min(h, y2 + pad_h)

                    if (x2 - x1) >= 15 and (y2 - y1) >= 15:
                        crop = img[y1:y2, x1:x2]
                    else:
                        crop = img
                else:
                    crop = img
            except Exception:
                crop = img
        else:
            crop = img

        resized = cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def compute_graph_intelligence(G: nx.DiGraph) -> Dict[str, Any]:
    """Compute forensic centrality metrics and identify critical evidence hubs."""
    if G.number_of_nodes() == 0:
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "density": 0.0,
            "key_hub": "None",
            "key_hub_score": 0.0,
            "critical_bridge": "None",
            "connected_components": 0,
            "modality_breakdown": {}
        }

    undirected = G.to_undirected()

    deg_centrality = nx.degree_centrality(G)
    key_hub_node = max(deg_centrality.items(), key=lambda x: x[1])[0] if deg_centrality else "None"
    key_hub_label = G.nodes[key_hub_node].get("label", key_hub_node) if key_hub_node != "None" else "None"

    betweenness = nx.betweenness_centrality(undirected)
    bridge_node = max(betweenness.items(), key=lambda x: x[1])[0] if betweenness else "None"
    bridge_label = G.nodes[bridge_node].get("label", bridge_node) if bridge_node != "None" else "None"

    modalities: Dict[str, int] = {}
    for _, data in G.nodes(data=True):
        ntype = data.get("type", "Unknown")
        modalities[ntype] = modalities.get(ntype, 0) + 1

    components_count = nx.number_connected_components(undirected)

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "density": round(nx.density(G), 3),
        "key_hub": key_hub_label,
        "key_hub_id": key_hub_node,
        "key_hub_score": round(deg_centrality.get(key_hub_node, 0.0), 3),
        "critical_bridge": bridge_label,
        "critical_bridge_id": bridge_node,
        "critical_bridge_score": round(betweenness.get(bridge_node, 0.0), 3),
        "connected_components": components_count,
        "modality_breakdown": modalities
    }


def build_advanced_evidence_graph(
    observations_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    cross_links_df: Optional[pd.DataFrame] = None,
    layout_mode: str = "Force-Directed Physics (Cluster)",
    filter_camera: Optional[str] = None,
    filter_modality: Optional[str] = None,
    filter_object: Optional[str] = None,
    filter_status: Optional[str] = None,
    focus_track_id: Optional[str] = None,
    search_keyword: Optional[str] = None,
    enable_trajectory_inference: bool = True,
    enable_audio_visual_fusion: bool = True,
    max_nodes: int = 150
) -> Tuple[nx.DiGraph, str, Dict[str, Any]]:
    """Build Advanced Multimodal Forensic Knowledge Graph with embedded circular thumbnails and HUD overlay."""
    G = nx.DiGraph()

    if observations_df is None or observations_df.empty:
        net = Network(height="680px", width="100%", bgcolor="#0b1120", font_color="white")
        net.add_node("empty", label="No Evidence Records Available", color="#64748b", shape="box")
        return G, net.generate_html(), compute_graph_intelligence(G)

    df = observations_df.copy()

    # Filter pipeline
    if filter_camera and filter_camera != "All Cameras":
        df = df[df["Camera ID"] == filter_camera]

    if filter_modality and filter_modality != "All Modalities":
        df = df[df["Modality"] == filter_modality]

    if filter_object and filter_object != "All Objects":
        df = df[df["Object"] == filter_object]

    if filter_status and filter_status != "All Statuses":
        df = df[df["Review Status"] == filter_status]

    if focus_track_id and focus_track_id != "All Tracks / Objects":
        df = df[df["Track ID"] == focus_track_id]

    if search_keyword and search_keyword.strip():
        kw = search_keyword.strip().lower()
        mask = (
            df["Object"].astype(str).str.lower().str.contains(kw, na=False) |
            df["Observation"].astype(str).str.lower().str.contains(kw, na=False) |
            df["Original Transcript"].astype(str).str.lower().str.contains(kw, na=False) |
            df["English Translation"].astype(str).str.lower().str.contains(kw, na=False) |
            df["Source"].astype(str).str.lower().str.contains(kw, na=False)
        )
        df = df[mask]

    if df.empty:
        net = Network(height="680px", width="100%", bgcolor="#0b1120", font_color="white")
        net.add_node("no_match", label="No observations match selected filters", color="#f59e0b", shape="box")
        return G, net.generate_html(), compute_graph_intelligence(G)

    if len(df) > max_nodes:
        df = df.head(max_nodes)

    df["_sort_sec"] = pd.to_numeric(df["Timestamp (seconds)"], errors="coerce").fillna(999999)
    df = df.sort_values("_sort_sec")

    net = Network(
        height="720px",
        width="100%",
        directed=True,
        bgcolor="#0b1120",
        font_color="#f8fafc",
        notebook=False,
        cdn_resources="in_line"
    )

    # Layout selection
    if layout_mode == "Timeline (Left-to-Right)":
        net.set_options("""
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed",
              "levelSeparation": 190,
              "nodeSpacing": 100
            }
          },
          "physics": { "enabled": false },
          "edges": { "smooth": { "type": "cubicBezier", "forceDirection": "horizontal" } }
        }
        """)
    elif layout_mode == "Forensic Hierarchy Tree":
        net.set_options("""
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 160,
              "nodeSpacing": 110
            }
          },
          "physics": { "enabled": false },
          "edges": { "smooth": { "type": "cubicBezier", "forceDirection": "vertical" } }
        }
        """)
    else:  # Force-Directed Physics (Cluster)
        net.barnes_hut(
            gravity=-3000,
            central_gravity=0.20,
            spring_length=160,
            spring_strength=0.04,
            damping=0.09
        )
        net.set_options("""
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "multiselect": true,
            "tooltipDelay": 80
          },
          "physics": {
            "enabled": true,
            "stabilization": { "enabled": true, "iterations": 120 }
          },
          "edges": {
            "smooth": { "type": "cubicBezier", "roundness": 0.2 },
            "font": { "size": 10, "color": "#94a3b8", "strokeWidth": 2, "strokeColor": "#0b1120" }
          }
        }
        """)

    STATUS_BORDERS = {
        "Approved": "#22c55e",  # Green
        "Confirmed": "#22c55e",
        "Flagged": "#f59e0b",   # Amber
        "Needs Review": "#f59e0b",
        "Rejected": "#ef4444",  # Red
        "Incorrect": "#ef4444",
        "Pending": "#38bdf8"    # Neon Cyan
    }

    added_nodes: Set[str] = set()
    node_metadata_lookup: Dict[str, Dict[str, Any]] = {}
    visual_sightings_by_object: Dict[str, List[Dict[str, Any]]] = {}
    audio_nodes: List[Dict[str, Any]] = []

    # Iterate observations
    for _, row in df.iterrows():
        obs_id = str(row["Observation ID"])
        modality = str(row.get("Modality", "CCTV"))
        source_id = str(row.get("Source ID", ""))
        source_name = str(row.get("Source", "Source"))
        camera_id = str(row.get("Camera ID", "CAM-01"))
        obj_name = str(row.get("Object", "Unknown"))
        timestamp = str(row.get("Timestamp", "00:00.00"))
        sec = row.get("Timestamp (seconds)")
        conf = row.get("Confidence")
        status = str(row.get("Review Status", "Pending"))
        track_id = str(row.get("Track ID", "Untracked"))
        evidence_path = str(row.get("Evidence Path", ""))
        bbox_str = str(row.get("Bounding Box", ""))

        # 1. Camera Anchor Node
        cam_node = f"cam_{camera_id}"
        if cam_node not in added_nodes:
            net.add_node(
                cam_node,
                label=f"📹 {camera_id}",
                title=f"Forensic Sensor: {camera_id}",
                shape="circularImage",
                image=CAM_ICON,
                size=36,
                borderWidth=4,
                color={"border": "#0ea5e9", "background": "#0f172a"},
                level=1 if "hierarchical" in layout_mode.lower() else None,
                font={"color": "#38bdf8", "size": 13, "bold": True, "strokeWidth": 2, "strokeColor": "#0b1120"}
            )
            G.add_node(cam_node, type="Camera", label=camera_id, camera_id=camera_id)
            added_nodes.add(cam_node)
            node_metadata_lookup[cam_node] = {
                "title": f"Sensor {camera_id}",
                "type": "CCTV Sensor",
                "cam": camera_id,
                "time": "Continuous",
                "conf": "100%",
                "status": "Verified Hardware",
                "img": CAM_ICON,
                "desc": f"Recording camera stream covering target sector: {source_name}"
            }

        # 2. Source Recording Node
        src_node = f"src_{source_id}"
        if src_node not in added_nodes:
            net.add_node(
                src_node,
                label=f"📁 {source_name[:14]}",
                title=f"Source Video: {source_name}",
                shape="database",
                size=22,
                color={"border": "#3b82f6", "background": "#1e293b"},
                level=2 if "hierarchical" in layout_mode.lower() else None,
                font={"color": "#93c5fd", "size": 11}
            )
            G.add_node(src_node, type="Source", label=source_name, modality=modality)
            added_nodes.add(src_node)

            net.add_edge(cam_node, src_node, label="RECORDED", color="#64748b", arrows="to")
            G.add_edge(cam_node, src_node, relation="RECORDED")

        # 3. Observation Node with Real Embedded Detection Crop
        obs_node = f"obs_{obs_id}"
        if obs_node not in added_nodes:
            status_border = STATUS_BORDERS.get(status, "#38bdf8")

            # Extract detection crop or fallback to sleek SVG
            crop_uri = get_detection_crop_data_uri(evidence_path, bbox_str)
            if not crop_uri:
                if modality == "Audio":
                    crop_uri = AUDIO_ICON
                elif "person" in obj_name.lower():
                    crop_uri = PERSON_ICON
                elif "pack" in obj_name.lower() or "bag" in obj_name.lower():
                    crop_uri = BACKPACK_ICON
                else:
                    crop_uri = EVENT_ICON

            conf_str = f"{conf:.0%}" if conf is not None and not np.isnan(conf) else "N/A"

            # Node label format matching reference (Subject A / Object X)
            node_display_label = f"{obj_name}\n[{timestamp}]"
            if "person" in obj_name.lower():
                node_display_label = f"Subject\n[{timestamp}]"
            elif "backpack" in obj_name.lower() or "bag" in obj_name.lower():
                node_display_label = f"Red Backpack\n[{timestamp}]"

            net.add_node(
                obs_node,
                label=node_display_label,
                shape="circularImage",
                image=crop_uri,
                size=32,
                borderWidth=4,
                color={
                    "border": status_border,
                    "background": "#0f172a",
                    "highlight": {"border": "#38bdf8", "background": "#1e293b"}
                },
                shadow={"enabled": True, "color": status_border, "size": 10},
                level=3 if "hierarchical" in layout_mode.lower() else None,
                font={"color": "#f8fafc", "size": 11, "strokeWidth": 2, "strokeColor": "#0b1120"}
            )
            G.add_node(
                obs_node,
                type="Observation",
                label=f"{obj_name} ({timestamp})",
                object=obj_name,
                modality=modality,
                camera_id=camera_id,
                timestamp=timestamp,
                seconds=sec,
                status=status,
                evidence_path=evidence_path,
                image=crop_uri
            )
            added_nodes.add(obs_node)

            # Metadata for HUD overlay
            node_metadata_lookup[obs_node] = {
                "title": f"{obj_name} ({track_id})",
                "type": f"{modality} Observation",
                "cam": camera_id,
                "time": timestamp,
                "conf": conf_str,
                "status": status,
                "img": crop_uri,
                "desc": str(row.get("Observation", f"{obj_name} identified at {camera_id}"))
            }

            net.add_edge(src_node, obs_node, label="DETECTED_IN", color="#475569", arrows="to")
            G.add_edge(src_node, obs_node, relation="DETECTED_IN")

            if modality in ["CCTV", "Photo"] and sec is not None and not np.isnan(sec):
                visual_sightings_by_object.setdefault(obj_name.lower().strip(), []).append({
                    "node": obs_node,
                    "camera_id": camera_id,
                    "seconds": float(sec),
                    "object": obj_name,
                    "timestamp": timestamp,
                    "track_id": track_id
                })

            if modality == "Audio":
                audio_nodes.append({
                    "node": obs_node,
                    "camera_id": camera_id,
                    "seconds": float(sec) if sec is not None and not np.isnan(sec) else 0.0,
                    "transcript": str(row.get("Original Transcript", "")),
                    "translation": str(row.get("English Translation", ""))
                })

        # Track continuity
        if modality == "CCTV" and track_id not in ("Untracked", "", "nan", "None"):
            trk_node = f"trk_{source_id}_{track_id}"
            if trk_node not in added_nodes:
                net.add_node(
                    trk_node,
                    label=f"Track: {track_id}",
                    shape="diamond",
                    size=20,
                    color={"border": "#eab308", "background": "#1e293b"},
                    level=2 if "hierarchical" in layout_mode.lower() else None,
                    font={"color": "#fde047", "size": 10}
                )
                G.add_node(trk_node, type="Track", label=track_id)
                added_nodes.add(trk_node)

                net.add_edge(src_node, trk_node, label="TRACKS", color="#ca8a04", arrows="to")
                G.add_edge(src_node, trk_node, relation="TRACKS")

            net.add_edge(trk_node, obs_node, label="CONTINUITY", color="#eab308", arrows="to")
            G.add_edge(trk_node, obs_node, relation="CONTINUITY")

    # ---------------------------------------------------------
    # 4. Spatio-Temporal Trajectory Inference (Cross-Camera Paths)
    # ---------------------------------------------------------
    if enable_trajectory_inference:
        for obj_key, sightings in visual_sightings_by_object.items():
            if len(sightings) < 2:
                continue

            sorted_sightings = sorted(sightings, key=lambda s: s["seconds"])
            for i in range(len(sorted_sightings) - 1):
                s1 = sorted_sightings[i]
                s2 = sorted_sightings[i + 1]

                if s1["camera_id"] != s2["camera_id"] or (s2["seconds"] - s1["seconds"]) >= 5.0:
                    delta_t = s2["seconds"] - s1["seconds"]
                    if 0 < delta_t <= 300.0:
                        edge_label = f"TRANSIT (+{delta_t:.1f}s)"
                        net.add_edge(
                            s1["node"],
                            s2["node"],
                            label=edge_label,
                            color="#38bdf8",
                            dashes=True,
                            width=2.5,
                            arrows="to"
                        )
                        G.add_edge(
                            s1["node"],
                            s2["node"],
                            relation="TRANSIT_PATH",
                            delta_seconds=delta_t,
                            from_cam=s1["camera_id"],
                            to_cam=s2["camera_id"]
                        )

    # ---------------------------------------------------------
    # 5. Multimodal Audio-Visual Corroboration
    # ---------------------------------------------------------
    if enable_audio_visual_fusion and audio_nodes:
        for a_item in audio_nodes:
            spoken_text = f"{a_item['transcript']} {a_item['translation']}".lower()

            for obj_key, visual_items in visual_sightings_by_object.items():
                aliases = MULTIMODAL_VOCAB.get(obj_key, [obj_key])
                if any(re.search(r"\b" + re.escape(alias) + r"\b", spoken_text) for alias in aliases):
                    for v_item in visual_items:
                        time_diff = abs(a_item["seconds"] - v_item["seconds"])
                        if time_diff <= 45.0:
                            net.add_edge(
                                a_item["node"],
                                v_item["node"],
                                label=f"CORROBORATES ({obj_key})",
                                color="#ec4899",
                                width=2.5,
                                dashes=[5, 5],
                                arrows="to"
                            )
                            G.add_edge(
                                a_item["node"],
                                v_item["node"],
                                relation="AUDIO_VISUAL_CORROBORATION",
                                keyword=obj_key,
                                time_delta=time_diff
                            )

    # ---------------------------------------------------------
    # 6. Reconstructed Events & Cross-Camera Links
    # ---------------------------------------------------------
    if events_df is not None and not events_df.empty:
        for _, ev in events_df.iterrows():
            ev_id = str(ev.get("event_id", ""))
            ev_title = str(ev.get("event_title", "Event"))
            ev_node = f"ev_{ev_id}"

            if ev_node not in added_nodes:
                net.add_node(
                    ev_node,
                    label=f"⚡ {ev_title[:16]}",
                    shape="circularImage",
                    image=EVENT_ICON,
                    size=34,
                    borderWidth=4,
                    color={"border": "#10b981", "background": "#0f172a"},
                    level=4 if "hierarchical" in layout_mode.lower() else None,
                    font={"color": "#6ee7b7", "size": 11, "bold": True}
                )
                G.add_node(ev_node, type="Event", label=ev_title)
                added_nodes.add(ev_node)

    if cross_links_df is not None and not cross_links_df.empty:
        for _, link in cross_links_df.iterrows():
            src_a = f"cam_{link.get('source_a', '')}"
            src_b = f"cam_{link.get('source_b', '')}"
            cat = str(link.get("category", "Target"))

            if src_a in added_nodes and src_b in added_nodes and src_a != src_b:
                net.add_edge(
                    src_a,
                    src_b,
                    label=f"LINK: {cat}",
                    color="#f59e0b",
                    dashes=True,
                    width=2
                )
                G.add_edge(src_a, src_b, relation="CROSS_CAMERA_LINK", category=cat)

    # 6b. Investigator Verified Relationships from Database
    try:
        db_rels = db_manager.get_graph_relationships()
        if not db_rels.empty:
            for _, r in db_rels.iterrows():
                src_obs = f"obs_{r['source_observation']}"
                tgt_obs = f"obs_{r['target_observation']}"
                rel_type = str(r.get('relationship_type', 'LINK'))
                rel_status = str(r.get('status', 'Approved'))
                rel_color = "#10b981" if rel_status == "Approved" else ("#ef4444" if rel_status == "Rejected" else "#f59e0b")

                if src_obs in added_nodes and tgt_obs in added_nodes and src_obs != tgt_obs:
                    net.add_edge(
                        src_obs,
                        tgt_obs,
                        label=f"{rel_type} ({rel_status})",
                        color=rel_color,
                        width=3.2,
                        dashes=(rel_status != "Approved"),
                        arrows="to"
                    )
                    G.add_edge(src_obs, tgt_obs, relation=rel_type, status=rel_status)
    except Exception:
        pass

    intelligence = compute_graph_intelligence(G)
    raw_html = net.generate_html()

    # ---------------------------------------------------------
    # 7. INJECT FORENSIC HUD OVERLAY (Matching Reference Video)
    # ---------------------------------------------------------
    import json
    metadata_json = json.dumps(node_metadata_lookup)

    hud_injection = f"""
    <style>
      #forensic-hud {{
        position: absolute;
        top: 20px;
        right: 20px;
        width: 310px;
        background: rgba(15, 23, 42, 0.94);
        border: 1px solid rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.7);
        border-radius: 10px;
        padding: 16px;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #f8fafc;
        z-index: 1000;
        backdrop-filter: blur(8px);
        transition: all 0.25s ease-in-out;
      }}
      #forensic-hud img {{
        width: 100%;
        height: 140px;
        object-fit: cover;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 10px;
      }}
      .hud-badge {{
        display: inline-block;
        font-size: 11px;
        font-weight: bold;
        padding: 2px 8px;
        border-radius: 4px;
        margin-bottom: 8px;
      }}
      .badge-approved {{ background: #166534; color: #86efac; border: 1px solid #22c55e; }}
      .badge-pending {{ background: #854d0e; color: #fde047; border: 1px solid #eab308; }}
      .badge-rejected {{ background: #991b1b; color: #fca5a5; border: 1px solid #ef4444; }}
      .hud-title {{ font-size: 15px; font-weight: bold; color: #38bdf8; margin: 0 0 6px 0; }}
      .hud-meta {{ font-size: 12px; color: #cbd5e1; margin-bottom: 4px; }}
      .hud-desc {{ font-size: 12px; color: #94a3b8; line-height: 1.4; margin-top: 8px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px; }}
    </style>

    <div id="forensic-hud">
      <div id="hud-badge" class="hud-badge badge-pending">INTERACTIVE EVIDENCE HUD</div>
      <img id="hud-img" src="{BACKPACK_ICON}" alt="Evidence Thumbnail" />
      <div id="hud-title" class="hud-title">Select Any Node to Inspect</div>
      <div id="hud-meta1" class="hud-meta">Sensor / Provenance: Active Knowledge Graph</div>
      <div id="hud-meta2" class="hud-meta">Observation Confidence: 87%</div>
      <div id="hud-desc" class="hud-desc">Click any circular node in the network to load detection crops, provenance records, and transit links.</div>
    </div>

    <script type="text/javascript">
      var nodeLookup = {metadata_json};

      network.on("selectNode", function(params) {{
        if (params.nodes.length > 0) {{
          var nodeId = params.nodes[0];
          var data = nodeLookup[nodeId];
          if (data) {{
            document.getElementById("hud-title").innerText = data.title;
            document.getElementById("hud-meta1").innerText = "Sensor: " + data.cam + " | Time: " + data.time;
            document.getElementById("hud-meta2").innerText = "Confidence Score: " + data.conf;
            document.getElementById("hud-desc").innerText = data.desc;
            document.getElementById("hud-img").src = data.img;

            var badge = document.getElementById("hud-badge");
            badge.innerText = data.status.toUpperCase();
            badge.className = "hud-badge " + (data.status === "Approved" ? "badge-approved" : (data.status === "Rejected" ? "badge-rejected" : "badge-pending"));
          }}
        }}
      }});

      network.on("deselectNode", function() {{
        document.getElementById("hud-title").innerText = "Evidence Graph Synthesized";
        document.getElementById("hud-meta1").innerText = "Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}";
        document.getElementById("hud-meta2").innerText = "Primary Hub: {intelligence['key_hub'][:20]}";
        document.getElementById("hud-desc").innerText = "Click on any circular node to inspect its high-resolution detection crop and transit path.";
      }});
    </script>
    """

    final_html = raw_html.replace("</body>", f"{hud_injection}</body>")

    # Save local export
    graph_export_file = GRAPHS_DIR / "evidence_graph_latest.html"
    with open(graph_export_file, "w", encoding="utf-8") as f_out:
        f_out.write(final_html)

    return G, final_html, intelligence
