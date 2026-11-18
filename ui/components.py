"""Reusable UI building blocks shared by all pages."""

from __future__ import annotations

import html
from typing import Iterable, Optional, Tuple

import streamlit as st


def page_header(title: str, description: str = "", eyebrow: str = "") -> None:
    st.markdown(
        '<div class="page-head">'
        + (f'<div class="eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else "")
        + f"<h1>{html.escape(title)}</h1>"
        + (f"<p>{html.escape(description)}</p>" if description else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def status_pill(status: str) -> str:
    """HTML pill for a review status (use inside st.markdown with unsafe_allow_html)."""
    s = status if status in ("Approved", "Pending", "Flagged", "Rejected") else "Pending"
    return f'<span class="pill pill-{s}">{html.escape(status or "Pending")}</span>'


def workflow_strip(steps: Iterable[Tuple[str, str, str, bool]]) -> None:
    """steps: (label, value, hint, done)."""
    cells = []
    for label, value, hint, done in steps:
        cells.append(
            f'<div class="step{" done" if done else ""}"><div class="k">{html.escape(label)}</div>'
            f'<div class="v">{html.escape(str(value))}</div>'
            f'<div class="h">{"Done" if done else html.escape(hint)}</div></div>'
        )
    st.markdown(f'<div class="flow">{"".join(cells)}</div>', unsafe_allow_html=True)


def empty_state(message: str, page: Optional[str] = None, label: str = "Go there") -> None:
    """Explain why a page is empty and link to the page that fills it."""
    st.info(message)
    if page:
        st.page_link(page, label=label, icon=":material/arrow_forward:")


def ai_disclaimer(text: str = "AI findings are candidates until an investigator approves them.") -> None:
    st.caption(text)
