"""Interactive rendering of the evidence graph (PyVis / vis-network).

Draws a chosen subset of `EvidenceGraph.graph`. Node shape and colour encode the
evidence type; entity and observation nodes show the actual detection crop with a
border coloured by review status. Clicking a node or an edge opens an inspector
panel with its source, timing, confidence, provenance and reasoning.
"""

from __future__ import annotations

import base64
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import cv2
from pyvis.network import Network

from core.config import GRAPHS_DIR
from services.evidence_graph import appearance
from services.evidence_graph import schema as S

STATUS_BORDER = {"Approved": "#22c55e", "Rejected": "#ef4444", "Flagged": "#f59e0b", "Pending": "#94a3b8"}
COMMUNITY_PALETTE = ["#60a5fa", "#f59e0b", "#34d399", "#f472b6", "#a78bfa", "#fb923c", "#22d3ee",
                     "#facc15", "#4ade80", "#e879f9", "#38bdf8", "#fda4af"]

LAYOUTS = ("Force-directed", "Timeline (left to right)", "Hierarchy (top down)")


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------
def get_detection_crop_data_uri(evidence_path: str, bbox_str: Optional[str] = None) -> str:
    """100x100 JPEG data URI of the detection crop (cached by path, bbox and file mtime)."""
    if not evidence_path or not Path(evidence_path).is_file():
        return ""
    try:
        mtime = Path(evidence_path).stat().st_mtime
    except OSError:
        return ""
    return _crop_data_uri_cached(evidence_path, bbox_str or "", mtime)


@lru_cache(maxsize=2048)
def _crop_data_uri_cached(evidence_path: str, bbox_str: str, _mtime: float) -> str:
    img = cv2.imread(evidence_path)
    if img is None:
        return ""
    h, w = img.shape[:2]
    crop = img
    box = appearance.parse_bbox(bbox_str)
    if box:
        x1, y1, x2, y2 = box
        pad_w, pad_h = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
        x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
        x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
        if (x2 - x1) >= 15 and (y2 - y1) >= 15:
            crop = img[y1:y2, x1:x2]
    ok, buffer = cv2.imencode(".jpg", cv2.resize(crop, (100, 100), interpolation=cv2.INTER_AREA),
                              [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _js_json(value: Any) -> str:
    """JSON safe to embed inside <script> (no '</script>' breakout)."""
    return json.dumps(value, default=str).replace("</", "<\\/").replace("<!--", "<\\!--")


def _fmt_time(d: Dict[str, Any]) -> str:
    if d.get("abs_start"):
        return d["abs_start"] + (f" to {d['abs_end'][11:]}" if d.get("abs_end") and d["abs_end"] != d["abs_start"] else "")
    t0, t1 = d.get("t_start"), d.get("t_end")
    if t0 is None:
        return "No timestamp"
    span = f"{t0:.1f}s" if t1 is None or abs(t1 - t0) < 0.05 else f"{t0:.1f}s to {t1:.1f}s"
    return f"{span} (source-relative)"


def _node_details(G, n: str, d: Dict[str, Any]) -> Dict[str, Any]:
    conf = d.get("confidence")
    related = G.in_degree(n) + G.out_degree(n)
    rows = [
        ("Type", d["type"]),
        ("Time", _fmt_time(d)),
        ("Camera", d.get("camera_id") or "-"),
        ("Source", d.get("source_id") or "-"),
        ("Confidence", f"{conf:.0%}" if isinstance(conf, (int, float)) else "-"),
        ("Review status", d.get("status") or "-"),
        ("Linked items", str(related)),
    ]
    if d.get("dominant_colour"):
        rows.append(("Approx. colour", f"{d['dominant_colour']} ({d.get('dominant_colour_share', 0):.0%} of crop, lighting-dependent)"))
    if d.get("location"):
        rows.append(("Location", d["location"]))
    if d.get("sha256"):
        rows.append(("SHA-256", d["sha256"][:16] + "..."))
    text = d.get("description") or d.get("text") or ""
    return {"title": d.get("label", n), "rows": rows, "text": text[:400], "status": d.get("status", "")}


def render_graph_html(
    eg,
    nodes: Iterable[str],
    layout: str = LAYOUTS[0],
    highlight_nodes: Optional[Set[str]] = None,
    highlight_edges: Optional[Set[Tuple[str, str]]] = None,
    communities: Optional[List[Set[str]]] = None,
    show_thumbnails: bool = True,
    show_edge_labels: bool = False,
    relations: Optional[Set[str]] = None,
    min_confidence: float = 0.0,
    height_px: int = 700,
) -> str:
    """Return standalone HTML for the subgraph induced by `nodes`."""
    G = eg.graph
    keep = [n for n in nodes if n in G]
    highlight_nodes = highlight_nodes or set()
    highlight_edges = highlight_edges or set()
    community_of = {}
    for i, comm in enumerate(communities or []):
        for n in comm:
            community_of[n] = COMMUNITY_PALETTE[i % len(COMMUNITY_PALETTE)]
    dim = bool(highlight_nodes)

    net = Network(height=f"{height_px}px", width="100%", directed=True, bgcolor="#0b1220",
                  font_color="#e2e8f0", cdn_resources="in_line")
    hierarchical = layout != LAYOUTS[0]
    if hierarchical:
        direction = "LR" if layout == LAYOUTS[1] else "UD"
        options = {"layout": {"hierarchical": {"enabled": True, "direction": direction, "sortMethod": "directed",
                                               "levelSeparation": 170, "nodeSpacing": 90}},
                   "physics": {"enabled": False}}
    else:
        options = {"physics": {"enabled": True, "solver": "barnesHut",
                               "barnesHut": {"gravitationalConstant": -4000, "springLength": 140, "damping": 0.12},
                               "stabilization": {"enabled": True, "iterations": 150}}}
    options.update({
        "interaction": {"hover": True, "navigationButtons": True, "keyboard": True, "tooltipDelay": 120},
        "edges": {"smooth": {"type": "continuous"}, "font": {"size": 9, "color": "#94a3b8", "strokeWidth": 0},
                  "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}}},
        "nodes": {"font": {"size": 11, "face": "Inter, Segoe UI, sans-serif"}},
    })
    net.set_options(json.dumps(options))

    # Hierarchical levels follow the provenance chain: location/camera -> file -> entity -> observation -> event.
    level = {S.LOCATION: 0, S.CAMERA: 1, S.VIDEO: 2, S.AUDIO: 2, S.IMAGE: 2, S.PERSON: 3, S.VEHICLE: 3,
             S.BELONGING: 3, S.OBJECT: 3, S.OBSERVATION: 4, S.TRANSCRIPT: 4, S.EVENT: 5}
    details: Dict[str, Any] = {}

    for n in keep:
        d = G.nodes[n]
        style = S.NODE_STYLE.get(d["type"], {"shape": "dot", "color": "#94a3b8"})
        color = community_of.get(n, style["color"])
        faded = dim and n not in highlight_nodes
        border = STATUS_BORDER.get(d.get("status", ""), color)
        kwargs: Dict[str, Any] = dict(
            label=d.get("label", n), title=f"{d['type']}: {d.get('label', n)}\n{_fmt_time(d)}",
            shape=style["shape"], size=18 if d["type"] in S.STRUCTURAL_TYPES else 14,
            color={"background": color, "border": border,
                   "highlight": {"background": color, "border": "#f8fafc"}},
            borderWidth=3 if n in highlight_nodes else 2,
            opacity=0.25 if faded else 1.0,
            font={"color": "#64748b" if faded else "#e2e8f0"},
        )
        if hierarchical:
            kwargs["level"] = level.get(d["type"], 4)
        if show_thumbnails and d["type"] in S.ENTITY_TYPES + (S.OBSERVATION,) and d.get("evidence_path"):
            uri = get_detection_crop_data_uri(d["evidence_path"], d.get("bbox"))
            if uri:
                kwargs.update(shape="circularImage", image=uri, size=22)
        net.add_node(n, **kwargs)
        details[n] = _node_details(G, n, d)

    edge_details: Dict[str, Any] = {}
    keep_set = set(keep)
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        if u not in keep_set or v not in keep_set:
            continue
        if relations is not None and d["relation"] not in relations:
            continue
        if (d.get("confidence") or 0) < min_confidence:
            continue
        style = S.EDGE_STYLE.get(d["relation"], {"color": "#64748b", "dashes": False})
        on_path = (u, v) in highlight_edges or (v, u) in highlight_edges
        faded = dim and not (u in highlight_nodes and v in highlight_nodes)
        eid = f"e{i}"
        net.add_edge(
            u, v, id=eid, title=f"{d['relation']} ({d['confidence']:.0%}, {d['origin']})",
            label=d["relation"] if show_edge_labels or on_path else "",
            color={"color": "#f8fafc" if on_path else style["color"], "opacity": 0.15 if faded else 0.9},
            dashes=style["dashes"], width=4 if on_path else (1 + 2 * float(d.get("confidence") or 0)),
        )
        edge_details[eid] = {
            "title": d["relation"],
            "rows": [("From", G.nodes[u].get("label", u)), ("To", G.nodes[v].get("label", v)),
                     ("Confidence", f"{d['confidence']:.0%}"), ("Origin", d["origin"]), ("Status", d["status"]),
                     ("Clock basis", d.get("basis") or "-"),
                     ("Provenance", ", ".join(str(p) for p in d.get("provenance", []) if p)[:120] or "-")],
            "text": d["reasoning"], "status": d["status"],
        }

    html = net.generate_html()
    inspector = f"""
    <style>
      body {{ margin: 0; }}
      #inspector {{ position: absolute; top: 12px; right: 12px; width: 300px; max-height: 88%; overflow: auto;
        background: rgba(15, 23, 42, 0.96); border: 1px solid #1e293b; border-radius: 8px; padding: 12px 14px;
        font: 12px/1.45 Inter, 'Segoe UI', sans-serif; color: #cbd5e1; z-index: 10; }}
      #inspector h4 {{ margin: 0 0 6px; font-size: 13px; color: #f8fafc; }}
      #inspector table {{ width: 100%; border-collapse: collapse; }}
      #inspector td {{ padding: 2px 0; vertical-align: top; }}
      #inspector td:first-child {{ color: #64748b; width: 38%; padding-right: 6px; }}
      #inspector p {{ margin: 8px 0 0; color: #94a3b8; border-top: 1px solid #1e293b; padding-top: 6px; }}
      #legend {{ position: absolute; left: 12px; bottom: 12px; background: rgba(15,23,42,.9); border: 1px solid #1e293b;
        border-radius: 8px; padding: 8px 10px; font: 11px Inter, 'Segoe UI', sans-serif; color: #94a3b8; z-index: 10; }}
      #legend span {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin: 0 4px 0 8px; }}
    </style>
    <div id="inspector"><h4>Inspector</h4><p style="border:0;margin:0;padding:0">
      Click a node or a relationship to see its source, timing, confidence and reasoning.</p></div>
    <div id="legend">Review:
      <span style="background:#94a3b8"></span>Pending<span style="background:#22c55e"></span>Approved
      <span style="background:#f59e0b"></span>Flagged<span style="background:#ef4444"></span>Rejected</div>
    <script type="text/javascript">
      var NODE_INFO = {_js_json(details)};
      var EDGE_INFO = {_js_json(edge_details)};
      function esc(s) {{ var d = document.createElement('div'); d.textContent = String(s); return d.innerHTML; }}
      function show(info) {{
        if (!info) return;
        var rows = info.rows.map(function (r) {{ return '<tr><td>' + esc(r[0]) + '</td><td>' + esc(r[1]) + '</td></tr>'; }}).join('');
        document.getElementById('inspector').innerHTML = '<h4>' + esc(info.title) + '</h4><table>' + rows + '</table>' +
          (info.text ? '<p>' + esc(info.text) + '</p>' : '');
      }}
      network.on('click', function (p) {{
        if (p.nodes.length) {{ show(NODE_INFO[p.nodes[0]]); }}
        else if (p.edges.length) {{ show(EDGE_INFO[p.edges[0]]); }}
      }});
    </script>
    """
    html = html.replace("</body>", inspector + "</body>")
    try:
        (GRAPHS_DIR / "evidence_graph_latest.html").write_text(html, encoding="utf-8")
    except OSError:
        pass
    return html
