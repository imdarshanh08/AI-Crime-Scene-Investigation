LEGACY NOTES: these describe the early evidence_graph_core.py prototype, which the
application no longer uses. See README.md for the current project.

EVIDENCE GRAPH STARTER - DAY 1
================================

Files
-----
evidence_graph_core.py
    SQLite data store, source/observation models, explainable candidate-link rules,
    investigator review audit trail, candidate event storage and graph payload.

demo_evidence_graph.py
    Creates a clearly labelled synthetic example database. It does not touch app.py.

Requirements
------------
Python 3.11+; standard library only for this core module.
Your existing app already uses NetworkX. PyVis can be added for interactive rendering.

Run the synthetic test
----------------------
Open PowerShell in C:\CrimeInvestigationAI, activate your existing .venv if desired,
and copy these files into that folder. Then run:

    .\.venv\Scripts\python.exe demo_evidence_graph.py

It creates demo_evidence.db in the current directory. It does not overwrite app.py.

Important behaviour / limitations
---------------------------------
1. Original source-relative timestamps are retained.
2. Cross-source temporal links are only proposed when BOTH source clocks have
   clock_alignment_verified=True. The stored mapping is:
       aligned_seconds = source_relative_seconds + clock_offset_seconds
   The offset must map that source's clock onto a common case timeline.
3. Matching a class label such as "backpack" alone does not create a possible
   visual match. At least one matching recorded appearance attribute is required.
   The current YOLO class label alone is not enough to establish physical identity.
4. Transcript keyword matches are simple keyword rules, not semantic AI reasoning.
5. The heuristic score is NOT a probability that a relationship is true.
6. All AI-created relationships start as pending. Investigator decisions and
   notes are recorded separately in review_actions.
7. Events created with create_event() start as pending candidate events.
8. This is a college-project foundation, not a validated forensic product.

Next integration step
---------------------
Do not replace your existing app.py. First run the synthetic test. Then share your
current app.py (or upload it) so the store can be connected to the existing upload,
detection, photo and audio sections without removing working functionality.
