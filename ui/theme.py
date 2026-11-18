"""Visual theme: a restrained dark forensic interface.

Colour is reserved for meaning (review status, evidence type, alerts). Surfaces are
flat, borders are hairline, and there are no glows or decorative animation.
"""

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #0b1220;
  --surface: #111a2b;
  --surface-2: #162236;
  --border: #1f2c42;
  --text: #e2e8f0;
  --muted: #8a99b0;
  --accent: #3b82f6;
  --ok: #22c55e;
  --warn: #f59e0b;
  --bad: #ef4444;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  color: var(--text);
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
.main .block-container, [data-testid="stMainBlockContainer"] {
  padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;
}
h1, h2, h3, h4 { letter-spacing: -0.01em; font-weight: 600; }
h1 { font-size: 1.55rem !important; }
h2 { font-size: 1.2rem !important; }
h3 { font-size: 1.02rem !important; }

[data-testid="stSidebar"] { background: #0a101c !important; border-right: 1px solid var(--border); }
[data-testid="stSidebarNav"] a span { font-size: 0.88rem; }

[data-testid="stMetric"] {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.78rem !important; font-weight: 500 !important; }
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 600 !important; color: var(--text) !important; }

.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { padding: 8px 14px; color: var(--muted); font-size: 0.86rem; font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--text) !important; }

.stButton > button, .stDownloadButton > button { border-radius: 6px; font-weight: 500; }
[data-testid="stExpander"] details { border: 1px solid var(--border) !important; border-radius: 8px !important; background: var(--surface); }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--border) !important; border-radius: 8px !important; }

code, pre { font-family: 'JetBrains Mono', Consolas, monospace !important; font-size: 0.82em; }
button:focus-visible, [role="tab"]:focus-visible, input:focus-visible, textarea:focus-visible {
  outline: 2px solid var(--accent) !important; outline-offset: 2px !important;
}

/* Page header */
.page-head { margin: 0 0 18px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
.page-head .eyebrow { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
.page-head h1 { margin: 2px 0 4px !important; padding: 0 !important; }
.page-head p { color: var(--muted); margin: 0; font-size: 0.9rem; max-width: 820px; }

/* Status pills */
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 0.74rem; font-weight: 600; border: 1px solid; }
.pill-Approved { color: #86efac; border-color: #166534; background: #052e16; }
.pill-Pending  { color: #cbd5e1; border-color: #334155; background: #0f172a; }
.pill-Flagged  { color: #fcd34d; border-color: #92400e; background: #1c1305; }
.pill-Rejected { color: #fca5a5; border-color: #991b1b; background: #1f0707; }

/* Workflow strip */
.flow { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 8px; margin: 4px 0 18px; }
.flow .step { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.flow .step .k { font-size: 0.72rem; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }
.flow .step .v { font-size: 1.15rem; font-weight: 600; margin-top: 2px; }
.flow .step .h { font-size: 0.74rem; color: var(--muted); margin-top: 2px; }
.flow .step.done { border-left: 3px solid var(--ok); }
@media (max-width: 900px) { .flow { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

.note { color: var(--muted); font-size: 0.84rem; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""


def apply_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
