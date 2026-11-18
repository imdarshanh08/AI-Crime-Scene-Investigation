"""Multi-camera workstation: synchronized feeds beside the evidence graph, with one-click review."""

from ui.components import page_header
from ui.views.workstation_view import render_trace_ai_workstation


def render():
    page_header(
        "Multi-camera workstation",
        "Recorded feeds side by side with the evidence graph for those cameras, the earliest observations, "
        "and quick review actions.",
        eyebrow="Review",
    )
    render_trace_ai_workstation()


if __name__ == "__main__":
    render()
