"""Evidence graph explorer: filter, search, expand, trace paths and inspect relationships."""

import io
from datetime import datetime
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st

from services.evidence_graph import analysis as A
from services.evidence_graph import schema as S
from services.evidence_graph.visualization import LAYOUTS, render_graph_html
from storage import db_manager
from ui.components import page_header
from ui.state import REVIEW_STATUSES, clean_text, fmt_pct, get_evidence_graph, sync_session_from_database

DEFAULT_TYPES = [t for t in S.NODE_TYPES if t != S.OBSERVATION]
MAX_RENDERED_NODES = 400


def _node_option(G, n: str) -> str:
    d = G.nodes[n]
    return f"{d.get('label', n)}  ·  {d['type']}  ·  {n.split(':', 1)[-1][:24]}"


def _iframe(html: str, height: int) -> None:
    import base64

    st.iframe("data:text/html;base64," + base64.b64encode(html.encode("utf-8")).decode("ascii"), height=height)


def _details_panel(eg, n: str) -> None:
    G = eg.graph
    d = G.nodes[n]
    st.markdown(f"#### {d.get('label', n)}")
    c1, c2 = st.columns([1, 2])
    with c1:
        path = d.get("evidence_path")
        if path and Path(path).is_file() and Path(path).suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            st.image(path, width="stretch")
        st.caption(f"Node ID: `{n}`")
    with c2:
        when = d.get("abs_start") or (f"{d['t_start']:.2f}s from start of recording" if d.get("t_start") is not None else "No timestamp")
        facts = {
            "Type": d["type"], "When": when, "Camera": d.get("camera_id") or "-",
            "Source file": G.nodes.get(f"src:{d.get('source_id', '')}", {}).get("label", d.get("source_id") or "-"),
            "Confidence": fmt_pct(d.get("confidence")), "Review status": d.get("status") or "-",
        }
        if d.get("object_class"):
            facts["Class"] = d["object_class"]
        if d.get("dominant_colour"):
            facts["Approx. colour"] = f"{d['dominant_colour']} (estimated from crop pixels)"
        if d.get("sha256"):
            facts["SHA-256"] = d["sha256"]
        if d.get("location"):
            facts["Location"] = d["location"]
        st.dataframe(pd.DataFrame(facts.items(), columns=["Field", "Value"]), hide_index=True, width="stretch")
        text = d.get("text") or d.get("description")
        if text:
            st.caption(text)

    related = A.connected_evidence(G, n, depth=1)
    st.markdown("**Directly related evidence**")
    if related:
        rows = []
        for r in related:
            rels = sorted({e["relation"] for e in (G.get_edge_data(n, r["node"]) or {}).values()} |
                          {e["relation"] for e in (G.get_edge_data(r["node"], n) or {}).values()})
            rows.append({"Item": r["label"], "Type": r["type"], "Relationship": ", ".join(rels),
                         "When": r["abs_start"] or (f"{r['t_start']:.1f}s" if r["t_start"] is not None else "-"),
                         "Review": r["status"] or "-"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("No related evidence.")

    if d["type"] in (S.OBSERVATION, S.TRANSCRIPT) and d.get("observation_id"):
        with st.form(key=f"graph_review_{n}"):
            cur = d.get("status") or "Pending"
            status = st.selectbox("Review decision", REVIEW_STATUSES,
                                  index=REVIEW_STATUSES.index(cur) if cur in REVIEW_STATUSES else 0)
            note = st.text_input("Investigator note", value=clean_text(d.get("review_notes")))
            if st.form_submit_button("Save decision"):
                db_manager.update_observation_review(d["observation_id"], status, note)
                sync_session_from_database()
                st.toast(f"Saved: {status}")
                st.rerun()


def render():
    page_header(
        "Evidence graph",
        "Typed, time-aware graph of cameras, files, tracked entities, observations, speech and events. "
        "Every relationship records its rule, confidence, clock basis and source records.",
        eyebrow="Analysis",
    )
    eg = get_evidence_graph()
    G = eg.graph
    if G.number_of_nodes() == 0:
        st.info("No evidence yet. Upload and analyse evidence first.")
        st.page_link("ui/pages/evidence_upload.py", label="Upload evidence", icon=":material/upload:")
        return

    summary = eg.summary()
    m = st.columns(5)
    m[0].metric("Nodes", summary["nodes"])
    m[1].metric("Relationships", summary["edges"])
    m[2].metric("Anchored sources", summary["anchored_sources"],
                help="Recordings with a verified start time. Only these can be compared across cameras.")
    m[3].metric("Investigator links", summary["origins"].get(S.ORIGIN_INVESTIGATOR, 0))
    m[4].metric("Contradictions", summary["contradictions"])

    # ------------------------------------------------------------------ filters
    with st.container(border=True):
        f1, f2, f3 = st.columns([2, 2, 1.4])
        present_types = [t for t in S.NODE_TYPES if summary["node_types"].get(t)]
        with f1:
            types = st.multiselect("Evidence types", present_types,
                                   default=[t for t in present_types if t in DEFAULT_TYPES], key="g_types")
        present_rel = sorted(summary["relations"])
        with f2:
            rels = st.multiselect("Relationship types", present_rel, default=present_rel, key="g_rels",
                                  help="; ".join(f"{r}: {S.RELATIONS.get(r, '')}" for r in present_rel)[:900])
        with f3:
            min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05, key="g_minconf")

        cams = sorted(d["camera_id"] for _, d in G.nodes(data=True) if d["type"] == S.CAMERA)
        g1, g2, g3 = st.columns([2, 2, 1.4])
        with g1:
            cam_filter = st.multiselect("Cameras", cams, default=[], key="g_cams", placeholder="All cameras")
        with g2:
            time_mode = st.radio("Time filter", ["None", "Seconds from recording start", "Wall clock (anchored only)"],
                                 horizontal=True, key="g_time_mode")
        with g3:
            layout = st.selectbox("Layout", LAYOUTS, key="g_layout")

        window_nodes = None
        if time_mode == "Seconds from recording start":
            tmax = max([d.get("t_end") or d.get("t_start") or 0 for _, d in G.nodes(data=True)] + [1.0])
            lo, hi = st.slider("Window (s)", 0.0, float(tmax), (0.0, float(tmax)), key="g_rel_window")
            window_nodes = A.nodes_in_window(G, start=lo, end=hi)
        elif time_mode == "Wall clock (anchored only)":
            stamps = sorted(d["abs_start"] for _, d in G.nodes(data=True) if d.get("abs_start"))
            if not stamps:
                st.caption("No recording has a verified start time. Set one in Settings > Evidence sources.")
            else:
                t0, t1 = datetime.fromisoformat(stamps[0]), datetime.fromisoformat(stamps[-1])
                if t0 == t1:
                    window_nodes = A.nodes_in_window(G, abs_start=t0, abs_end=t1)
                else:
                    lo, hi = st.slider("Window", t0, t1, (t0, t1), format="YYYY-MM-DD HH:mm:ss", key="g_abs_window")
                    window_nodes = A.nodes_in_window(G, abs_start=lo, abs_end=hi)

    # ----------------------------------------------------------- search & focus
    with st.container(border=True):
        s1, s2, s3 = st.columns([3, 1, 1.4])
        candidates = [n for n, d in G.nodes(data=True) if d["type"] != S.OBSERVATION or d.get("modality") != "CCTV"
                      or d.get("track_id") in ("", "Untracked")]
        labels = {_node_option(G, n): n for n in sorted(candidates, key=lambda n: G.nodes[n].get("label", n))}
        with s1:
            focus_label = st.selectbox("Search and focus on an item", ["(whole case)"] + list(labels),
                                       key="g_focus", help="Type to search by label, class, camera or ID.")
        focus = labels.get(focus_label)
        with s2:
            depth = st.number_input("Expand depth", 0, 4, 1, key="g_depth", disabled=focus is None,
                                    help="0 = only the item; each step adds directly connected evidence.")
        with s3:
            colour_by = st.radio("Colour by", ["Evidence type", "Community"], horizontal=True, key="g_colour")
        o1, o2, o3 = st.columns(3)
        thumbs = o1.toggle("Detection thumbnails", value=True, key="g_thumbs")
        edge_labels = o2.toggle("Relationship labels", value=False, key="g_edge_labels")
        path_mode = o3.toggle("Find evidence path", value=False, key="g_path")
        path_hops = []
        if path_mode:
            p1, p2 = st.columns(2)
            a = p1.selectbox("From", list(labels), key="g_path_a")
            b = p2.selectbox("To", list(labels), index=min(1, len(labels) - 1), key="g_path_b")
            evidential_only = st.checkbox("Ignore registry links (file/camera)", value=True, key="g_path_ev",
                                          help="Find connections through evidence, not just 'same file'.")
            path_hops = A.evidence_path(G, labels[a], labels[b], include_structural=not evidential_only)
            if not path_hops:
                st.caption("No connecting chain of evidence between these items.")

    # ------------------------------------------------------------ node selection
    if focus:
        nodes = A.neighbourhood(G, focus, depth=int(depth))
    else:
        nodes = set(G.nodes)
    nodes = {n for n in nodes if G.nodes[n]["type"] in types or n == focus}
    if cam_filter:
        nodes = {n for n in nodes if G.nodes[n].get("camera_id") in cam_filter
                 or G.nodes[n]["type"] in (S.LOCATION,) or n == focus}
    if window_nodes is not None:
        nodes = {n for n in nodes if n in window_nodes or G.nodes[n]["type"] in S.STRUCTURAL_TYPES or n == focus}
    highlight_nodes, highlight_edges = set(), set()
    if path_hops:
        for h in path_hops:
            highlight_nodes |= {h["from"], h["to"]}
            highlight_edges.add((h["from"], h["to"]))
        nodes |= highlight_nodes
    elif focus:
        highlight_nodes = {focus} | A.neighbourhood(G, focus, depth=1)
    truncated = len(nodes) > MAX_RENDERED_NODES
    if truncated:
        keep = A.chronological(G, nodes)[:MAX_RENDERED_NODES]
        nodes = set(keep) | highlight_nodes

    communities = A.evidence_communities(G) if colour_by == "Community" else None
    html = render_graph_html(eg, nodes, layout=layout, highlight_nodes=highlight_nodes, highlight_edges=highlight_edges,
                             communities=communities, show_thumbnails=thumbs, show_edge_labels=edge_labels,
                             relations=set(rels), min_confidence=min_conf)
    if truncated:
        st.caption(f"Showing the first {MAX_RENDERED_NODES} items in time order. Narrow the filters or focus on an item.")
    _iframe(html, 720)
    st.caption("Shape and colour show the evidence type; the border shows the review status. Dashed lines are "
               "candidate or temporal relationships. Click any node or line for details.")

    if path_hops:
        st.markdown("**Evidence path**")
        st.dataframe(pd.DataFrame([{
            "Step": i + 1, "From": G.nodes[h["from"]].get("label"), "Relationship": h["relation"],
            "To": G.nodes[h["to"]].get("label"), "Confidence": f"{h['confidence']:.0%}", "Origin": h["origin"],
            "Reasoning": h["reasoning"]} for i, h in enumerate(path_hops)]), hide_index=True, width="stretch")

    # ---------------------------------------------------------------- tabs
    t_item, t_rel, t_con, t_comm, t_key, t_export = st.tabs(
        ["Selected item", "Relationships", f"Contradictions ({len(eg.contradictions)})", "Communities",
         "Key evidence", "Export"])

    with t_item:
        if focus:
            _details_panel(eg, focus)
        else:
            st.caption("Choose an item in 'Search and focus' to see its source, timing, confidence and related evidence.")

    with t_rel:
        edges = eg.edges_table()
        if edges.empty:
            st.caption("No relationships.")
        else:
            visible = edges[edges["from_id"].isin(nodes) & edges["to_id"].isin(nodes) & edges["Relation"].isin(rels)
                            & (edges["Confidence"] >= min_conf)]
            r1, r2 = st.columns([2, 2])
            rel_pick = r1.multiselect("Relationship", sorted(visible["Relation"].unique()), key="g_tab_rel",
                                      placeholder="All")
            origin_pick = r2.multiselect("Origin", sorted(visible["Origin"].unique()), key="g_tab_origin",
                                         placeholder="All")
            if rel_pick:
                visible = visible[visible["Relation"].isin(rel_pick)]
            if origin_pick:
                visible = visible[visible["Origin"].isin(origin_pick)]
            st.dataframe(visible.drop(columns=["from_id", "to_id"]), hide_index=True, width="stretch",
                         column_config={"Confidence": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0,
                                                                                      format="%.2f")})
            st.caption(f"{len(visible)} of {len(edges)} relationships shown.")

    with t_con:
        if not eg.contradictions:
            st.success("No contradictions detected by the current rules (location simultaneity, class mismatch, "
                       "conflicting link decisions, impossible transit order, event vs observation reviews).")
        else:
            st.dataframe(pd.DataFrame([{
                "Kind": c["kind"], "Severity": c["severity"],
                "Between": " / ".join(G.nodes[n].get("label", n) for n in c["nodes"]),
                "Explanation": c["reasoning"], "Records": ", ".join(p for p in c["provenance"] if p)}
                for c in eg.contradictions]), hide_index=True, width="stretch")

    with t_comm:
        comms = A.evidence_communities(G)
        st.caption("Louvain community detection over evidential relationships (registry links excluded). "
                   "A community is a cluster of evidence that is more connected internally than externally.")
        if not comms:
            st.caption("Not enough evidential relationships to form clusters.")
        for i, c in enumerate(comms[:12], 1):
            types_in = pd.Series([G.nodes[n]["type"] for n in c]).value_counts().to_dict()
            cams_in = sorted({G.nodes[n].get("camera_id") for n in c if G.nodes[n].get("camera_id")})
            with st.expander(f"Cluster {i}: {len(c)} items, cameras {', '.join(cams_in) or '-'}"):
                st.write(", ".join(f"{v} {k}" for k, v in types_in.items()))
                st.dataframe(pd.DataFrame([{"Item": G.nodes[n].get("label", n), "Type": G.nodes[n]["type"],
                                            "Review": G.nodes[n].get("status", "")}
                                           for n in A.chronological(G, c)]), hide_index=True, width="stretch")

    with t_key:
        key = A.key_evidence(G, top_n=15)
        st.caption("Betweenness: how often an item lies on the strongest paths between other evidence. "
                   "Degree: share of items it connects to. Cameras and files are excluded.")
        if key:
            st.dataframe(pd.DataFrame(key).drop(columns=["node"]), hide_index=True, width="stretch")

    with t_export:
        st.download_button("Interactive graph (HTML)", html.encode("utf-8"), "evidence_graph.html", "text/html",
                           key="g_dl_html")
        H = nx.MultiDiGraph()
        for n, d in G.nodes(data=True):
            H.add_node(n, **{k: ("" if v is None else str(v)) for k, v in d.items() if k not in ("status_summary",)})
        for u, v, k, d in G.edges(keys=True, data=True):
            H.add_edge(u, v, key=k, **{kk: ("" if vv is None else str(vv)) for kk, vv in d.items()})
        buf = io.BytesIO()
        try:
            nx.write_graphml(H, buf)
            st.download_button("Graph for Gephi / Cytoscape (GraphML)", buf.getvalue(), "evidence_graph.graphml",
                               "application/xml", key="g_dl_graphml")
        except Exception as e:  # GraphML is strict about attribute types
            st.caption(f"GraphML export unavailable: {e}")
        st.download_button("Relationships (CSV)", eg.edges_table().to_csv(index=False).encode("utf-8"),
                           "evidence_relationships.csv", "text/csv", key="g_dl_edges")


if __name__ == "__main__":
    render()
