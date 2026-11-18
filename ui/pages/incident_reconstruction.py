"""Incident reconstruction: ask questions of the evidence and build referenced timelines."""

import pandas as pd
import streamlit as st

from services.evidence_graph import schema as S
from services.reconstruction.query_engine import answer_question, entity_sequence
from storage import db_manager
from ui.components import page_header
from ui.state import get_evidence_graph
from ui.views.cross_camera_view import render_cross_camera_linking
from ui.views.cross_modal_view import render_cross_modal_correlation
from ui.views.sighting_comparison_view import render_sighting_comparison

EXAMPLES = [
    "Show me everything related to the red backpack",
    "What happened between 00:10 and 00:40?",
    "What happened between 2:30 PM and 2:45 PM?",
    "Which cameras recorded a person?",
    "Show all related audio and video observations",
    "Are there any conflicting observations?",
]

STATUS_ORDER = ["Investigator-verified", "AI finding (unverified)", "Flagged for review", "Rejected by investigator"]


def _timeline_table(rows):
    df = pd.DataFrame(rows).drop(columns=["node", "_sort"], errors="ignore")
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"What": st.column_config.TextColumn(width="large"),
                                "References": st.column_config.TextColumn(width="medium")})


def _ask_tab(eg):
    st.markdown("Ask in plain language. The answer cites stored evidence only; nothing is inferred beyond it.")
    with st.form("ask_form"):
        q = st.text_input("Question", key="ir_question",
                          placeholder="e.g. Show me everything related to the red backpack")
        submitted = st.form_submit_button("Reconstruct", type="primary")
    st.caption("Examples: " + " · ".join(f"_{e}_" for e in EXAMPLES))
    if submitted and q.strip():
        st.session_state["ir_last_question"] = q.strip()
        db_manager.log_investigator_action("Incident query", "Query", "-", q.strip()[:200])
    question = st.session_state.get("ir_last_question")
    if not question:
        return

    res = answer_question(question, eg)
    with st.container(border=True):
        st.markdown("**How the question was interpreted**")
        for line in res.plan.describe():
            st.markdown(f"- {line}")
        for note in res.plan.notes:
            st.caption(note)

    if not res.timeline:
        st.info("No stored evidence matches this question.")
    else:
        counts = pd.Series([r["Evidence status"] for r in res.timeline]).value_counts()
        m = st.columns(4)
        for col, label in zip(m, STATUS_ORDER):
            col.metric(label, int(counts.get(label, 0)))
        st.markdown(f"**Chronological reconstruction** ({len(res.matched)} direct matches, "
                    f"{len(res.related)} connected through the graph)")
        st.caption("Wall-clock times appear only for recordings with a verified start time; other times are seconds "
                   "from the start of that recording and are not comparable across recordings.")
        _timeline_table(res.timeline)
        st.download_button("Download timeline (CSV)",
                           pd.DataFrame(res.timeline).drop(columns=["node", "_sort"]).to_csv(index=False).encode("utf-8"),
                           "incident_timeline.csv", "text/csv", key="ir_dl")

    if res.cameras:
        st.markdown("**Cameras involved**")
        st.dataframe(pd.DataFrame(res.cameras), hide_index=True, width="stretch")

    st.markdown("**Supporting relationships**")
    if res.supporting_edges:
        st.dataframe(pd.DataFrame(res.supporting_edges), hide_index=True, width="stretch")
    else:
        st.caption("No supporting or temporal relationships between the retrieved items.")

    st.markdown("**Contradictions**")
    if res.contradictions:
        G = eg.graph
        st.dataframe(pd.DataFrame([{
            "Kind": c["kind"], "Severity": c["severity"],
            "Between": " / ".join(G.nodes[n].get("label", n) for n in c["nodes"]),
            "Explanation": c["reasoning"]} for c in res.contradictions]), hide_index=True, width="stretch")
    else:
        st.caption("No contradictions involve the retrieved evidence.")


def _sequence_tab(eg):
    G = eg.graph
    choices = {f"{d.get('label', n)}  ·  {d['type']}  ·  {d.get('camera_id', '')}": n
               for n, d in G.nodes(data=True) if d["type"] in S.ENTITY_TYPES + (S.EVENT,)}
    if not choices:
        st.info("No tracked entities or events yet. Run CCTV analysis and reconstruct events first.")
        return
    pick = st.selectbox("Entity or event", sorted(choices), key="ir_seq_pick")
    rows = entity_sequence(eg, choices[pick])
    st.caption("Everything directly evidenced about this item, in time order.")
    _timeline_table(rows)


def render():
    page_header(
        "Incident reconstruction",
        "Question the evidence, review sequences of events, and manage cross-camera and cross-modal proposals.",
        eyebrow="Analysis",
    )
    eg = get_evidence_graph()
    t_ask, t_seq, t_cam, t_modal, t_cmp = st.tabs(
        ["Ask the evidence", "Entity sequence", "Cross-camera links", "Cross-modal links", "Compare sightings"])
    with t_ask:
        _ask_tab(eg)
    with t_seq:
        _sequence_tab(eg)
    with t_cam:
        render_cross_camera_linking()
        st.caption("Cross-camera associations are candidate links. They do not prove physical identity.")
    with t_modal:
        render_cross_modal_correlation(db_manager.get_all_observations_df())
    with t_cmp:
        render_sighting_comparison()


if __name__ == "__main__":
    render()
