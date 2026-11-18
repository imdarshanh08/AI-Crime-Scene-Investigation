"""Graph analysis over the evidence graph.

All functions are read-only and operate on `EvidenceGraph.graph` (a MultiDiGraph).
Structural registry edges (file -> camera, observation -> file) are useful for
navigation but say nothing about the incident, so the investigative analyses
(paths, communities, centrality) can exclude them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set

import networkx as nx

from services.evidence_graph import schema as S

STRUCTURAL_RELATIONS = {"CAPTURED_BY", "DETECTED_IN", "LOCATED_NEAR"}


def _simple_undirected(G: nx.MultiDiGraph, relations: Optional[Set[str]] = None,
                       exclude: Optional[Set[str]] = None) -> nx.Graph:
    """Collapse parallel edges; weight = strongest confidence between the pair."""
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        rel = d["relation"]
        if relations is not None and rel not in relations:
            continue
        if exclude and rel in exclude:
            continue
        conf = float(d.get("confidence") or 0.0)
        if H.has_edge(u, v):
            if conf > H[u][v]["confidence"]:
                H[u][v].update(confidence=conf, relation=rel)
        else:
            H.add_edge(u, v, confidence=conf, relation=rel)
    for u, v, d in H.edges(data=True):
        # Strong evidence = short distance.
        d["distance"] = 1.0 / (d["confidence"] + 0.05)
    return H


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------
def neighbourhood(G: nx.MultiDiGraph, center: str, depth: int = 1,
                  relations: Optional[Iterable[str]] = None) -> Set[str]:
    """Nodes reachable from `center` within `depth` hops (edges in either direction)."""
    if center not in G:
        return set()
    allowed = set(relations) if relations else None
    seen = {center}
    frontier = {center}
    for _ in range(max(0, depth)):
        nxt = set()
        for n in frontier:
            for _, m, d in G.out_edges(n, data=True):
                if allowed is None or d["relation"] in allowed:
                    nxt.add(m)
            for m, _, d in G.in_edges(n, data=True):
                if allowed is None or d["relation"] in allowed:
                    nxt.add(m)
        nxt -= seen
        seen |= nxt
        frontier = nxt
    return seen


def connected_evidence(G: nx.MultiDiGraph, center: str, depth: int = 2) -> List[Dict[str, Any]]:
    """Everything linked to `center`, with the hop distance, for an evidence listing."""
    if center not in G:
        return []
    H = _simple_undirected(G)
    lengths = nx.single_source_shortest_path_length(H, center, cutoff=depth)
    out = []
    for n, hops in sorted(lengths.items(), key=lambda kv: kv[1]):
        if n == center:
            continue
        d = G.nodes[n]
        out.append({"node": n, "hops": hops, "type": d["type"], "label": d.get("label", n),
                    "t_start": d.get("t_start"), "abs_start": d.get("abs_start", ""), "status": d.get("status", "")})
    return out


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def evidence_path(G: nx.MultiDiGraph, a: str, b: str, include_structural: bool = True) -> List[Dict[str, Any]]:
    """Strongest chain of evidence between two nodes (Dijkstra on 1/confidence).

    Returns one dict per hop with the edge used, or [] if the nodes are not connected.
    """
    if a not in G or b not in G:
        return []
    H = _simple_undirected(G, exclude=None if include_structural else STRUCTURAL_RELATIONS)
    try:
        nodes = nx.dijkstra_path(H, a, b, weight="distance")
    except nx.NetworkXNoPath:
        return []
    hops = []
    for u, v in zip(nodes, nodes[1:]):
        # Report the actual directed edge (either direction) with the highest confidence.
        candidates = [(u, v, d) for d in (G.get_edge_data(u, v) or {}).values()] + \
                     [(v, u, d) for d in (G.get_edge_data(v, u) or {}).values()]
        src, dst, data = max(candidates, key=lambda c: c[2].get("confidence") or 0)
        hops.append({"from": src, "to": dst, "relation": data["relation"], "confidence": data["confidence"],
                     "reasoning": data["reasoning"], "origin": data["origin"], "status": data["status"]})
    return hops


# ---------------------------------------------------------------------------
# Communities and centrality
# ---------------------------------------------------------------------------
def evidence_communities(G: nx.MultiDiGraph, min_size: int = 3, seed: int = 42) -> List[Set[str]]:
    """Louvain communities over evidential (non-registry) relationships.

    Registry edges are excluded so clusters reflect incident structure rather than
    simply "everything from the same file". Deterministic for a fixed seed.
    """
    H = _simple_undirected(G, exclude=STRUCTURAL_RELATIONS)
    H.remove_nodes_from([n for n in list(H.nodes) if H.degree(n) == 0])
    if H.number_of_edges() == 0:
        return []
    comms = nx.community.louvain_communities(H, weight="confidence", seed=seed)
    return sorted((set(c) for c in comms if len(c) >= min_size), key=len, reverse=True)


def key_evidence(G: nx.MultiDiGraph, top_n: int = 8) -> List[Dict[str, Any]]:
    """Rank incident-level nodes by betweenness (bridging) and degree (hub) centrality.

    Cameras, files and locations are excluded from the ranking: they are always
    highly connected by construction and would crowd out actual evidence.
    """
    H = _simple_undirected(G, exclude=STRUCTURAL_RELATIONS)
    candidates = [n for n, d in G.nodes(data=True) if d["type"] not in S.STRUCTURAL_TYPES and H.degree(n) > 0]
    if not candidates:
        return []
    betweenness = nx.betweenness_centrality(H, weight="distance", seed=42, k=min(200, H.number_of_nodes()))
    degree = nx.degree_centrality(H)
    rows = []
    for n in candidates:
        d = G.nodes[n]
        rows.append({"node": n, "label": d.get("label", n), "type": d["type"],
                     "betweenness": round(betweenness.get(n, 0.0), 4), "degree": round(degree.get(n, 0.0), 4),
                     "status": d.get("status", "")})
    rows.sort(key=lambda r: (r["betweenness"], r["degree"]), reverse=True)
    return rows[:top_n]


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
def nodes_in_window(G: nx.MultiDiGraph, start: Optional[float] = None, end: Optional[float] = None,
                    abs_start: Optional[datetime] = None, abs_end: Optional[datetime] = None) -> Set[str]:
    """Nodes whose time span intersects the window.

    Use (start, end) for source-relative seconds, or (abs_start, abs_end) for
    wall-clock time; the latter only matches nodes from anchored sources.
    """
    out = set()
    for n, d in G.nodes(data=True):
        if abs_start is not None or abs_end is not None:
            if not d.get("abs_start"):
                continue
            n0 = datetime.fromisoformat(d["abs_start"])
            n1 = datetime.fromisoformat(d.get("abs_end") or d["abs_start"])
            if (abs_end is None or n0 <= abs_end) and (abs_start is None or n1 >= abs_start):
                out.add(n)
        else:
            t0, t1 = d.get("t_start"), d.get("t_end")
            if t0 is None:
                continue
            t1 = t0 if t1 is None else t1
            if (end is None or t0 <= end) and (start is None or t1 >= start):
                out.add(n)
    return out


def chronological(G: nx.MultiDiGraph, nodes: Iterable[str]) -> List[str]:
    """Order nodes by wall-clock time when anchored, else by source then offset; untimed last."""
    def key(n):
        d = G.nodes[n]
        if d.get("abs_start"):
            return (0, d["abs_start"], "", 0.0)
        if d.get("t_start") is not None:
            return (1, "", d.get("source_id", ""), float(d["t_start"]))
        return (2, "", d.get("source_id", ""), 0.0)
    return sorted(nodes, key=key)
