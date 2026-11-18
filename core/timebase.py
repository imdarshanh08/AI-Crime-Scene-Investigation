"""Shared timebase for comparing timestamps from different evidence sources.

Every observation timestamp is *source-relative* (seconds from the start of its own
recording). Two timestamps are only comparable when:

* they come from the same source (e.g. CCTV video and the audio extracted from it
  share one clock), or
* both sources have an investigator-verified recording start time ("anchor"), so
  both can be converted to wall-clock time.

Anything else is not comparable, and the evidence graph never invents an ordering
or overlap between such items.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Mapping, Optional, Tuple

SHARED_CLOCK = "shared-clock"
ANCHORED = "anchored"


@dataclass(frozen=True)
class Timebase:
    """Maps source IDs to verified recording start times."""

    anchors: Mapping[str, datetime]

    @classmethod
    def from_sources(cls, sources) -> "Timebase":
        anchors: Dict[str, datetime] = {}
        for src in sources:
            raw = (src.get("recorded_start") or "").strip() if isinstance(src, dict) else ""
            if not raw:
                continue
            try:
                anchors[str(src["source_id"])] = datetime.fromisoformat(raw)
            except ValueError:
                continue  # malformed values are ignored rather than guessed
        return cls(anchors)

    def is_anchored(self, source_id: str) -> bool:
        return str(source_id) in self.anchors

    def to_absolute(self, source_id: str, seconds: Optional[float]) -> Optional[datetime]:
        if seconds is None:
            return None
        start = self.anchors.get(str(source_id))
        if start is None:
            return None
        return start + timedelta(seconds=float(seconds))

    def comparable_delta(
        self,
        source_a: str,
        seconds_a: Optional[float],
        source_b: str,
        seconds_b: Optional[float],
    ) -> Optional[Tuple[float, str]]:
        """Return (b - a in seconds, basis) when the two instants can be compared, else None."""
        if seconds_a is None or seconds_b is None:
            return None
        if str(source_a) == str(source_b):
            return float(seconds_b) - float(seconds_a), SHARED_CLOCK
        abs_a = self.to_absolute(source_a, seconds_a)
        abs_b = self.to_absolute(source_b, seconds_b)
        if abs_a is None or abs_b is None:
            return None
        return (abs_b - abs_a).total_seconds(), ANCHORED

    def compare_intervals(
        self,
        source_a: str, start_a: float, end_a: float,
        source_b: str, start_b: float, end_b: float,
    ) -> Optional[Tuple[str, float, str]]:
        """Compare interval A with interval B.

        Returns (order, seconds, basis) or None when the clocks are not comparable:
          order "overlaps"   -> seconds = length of the overlap (>= 0)
          order "a_before_b" -> seconds = gap from end of A to start of B (> 0)
          order "b_before_a" -> seconds = gap from end of B to start of A (> 0)
        """
        shift = self.comparable_delta(source_a, 0.0, source_b, 0.0)
        if shift is None:
            return None
        offset, basis = shift  # B's clock zero expressed on A's clock
        b_start, b_end = float(start_b) + offset, float(end_b) + offset
        a_start, a_end = float(start_a), float(end_a)
        if b_start > a_end:
            return "a_before_b", b_start - a_end, basis
        if b_end < a_start:
            return "b_before_a", a_start - b_end, basis
        return "overlaps", min(a_end, b_end) - max(a_start, b_start), basis
