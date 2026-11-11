"""Hornet module: turn vespid tracks into *hawking* events.

A hornet flying past is not an attack. *Vespa velutina* hunts by hovering in
front of the entrance ("hawking") and catching returning foragers. An event is
therefore opened only when a vespid track stays inside the hawking zone for at
least ``min_seconds`` without leaving a small area, and it is closed when the
track ends. See docs/design.md, section 6.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .geometry import Point, point_in_polygon
from .tracks import VESPIDS, Track


@dataclass
class HornetEvent:
    event_id: int
    track_id: int
    start_t: float
    species: str
    confidence: float
    end_t: float | None = None
    strikes: int = 0  # close approaches to a bee
    snapshot_jpeg: bytes | None = None
    box: tuple[float, float, float, float] | None = None
    _struck: set = field(default_factory=set, repr=False)

    @property
    def duration(self) -> float:
        return (self.end_t if self.end_t is not None else self.start_t) - self.start_t

    @property
    def is_velutina(self) -> bool:
        return self.species == "vespa_velutina"


@dataclass
class HornetMonitor:
    hawk_zone: Sequence[Point] | None  # pixels; None = whole frame
    frame_diag: float
    min_seconds: float = 3.0
    max_spread: float = 0.30  # fraction of the frame diagonal
    min_vespid_score: float = 0.5
    strike_body_lengths: float = 1.5
    _next_id: int = 1
    _open: dict[int, HornetEvent] = field(default_factory=dict)

    def update(
        self,
        t: float,
        tracks: Mapping[int, Track],
        visible: set[int],
        bee_points: Sequence[tuple[int, Point]] = (),
    ) -> tuple[list[HornetEvent], list[HornetEvent]]:
        """Process one frame. Returns (opened, closed) events.

        ``tracks`` holds every live track, ``visible`` the ids seen this frame,
        ``bee_points`` the (track_id, centre) of non-vespid tracks this frame.
        """
        opened: list[HornetEvent] = []
        for tid in visible:
            tr = tracks[tid]
            ev = self._open.get(tid)
            if ev is None:
                if self.is_hawking(tr):
                    ev = HornetEvent(self._next_id, tid, t, tr.species, tr.species_confidence)
                    self._next_id += 1
                    self._open[tid] = ev
                    opened.append(ev)
            else:
                # Species can firm up as more crops are voted.
                ev.species, ev.confidence = tr.species, tr.species_confidence
            if ev is not None and tr.points:
                hx, hy = tr.points[-1]
                reach = self.strike_body_lengths * tr.body_length
                for bid, (bx, by) in bee_points:
                    if bid not in ev._struck and math.hypot(bx - hx, by - hy) <= reach:
                        ev._struck.add(bid)
                        ev.strikes += 1
        closed = []
        for tid in [tid for tid in self._open if tid not in tracks]:
            ev = self._open.pop(tid)
            ev.end_t = max(ev.start_t, t)
            closed.append(ev)
        return opened, closed

    def close_all(self, t: float) -> list[HornetEvent]:
        out = []
        for ev in self._open.values():
            ev.end_t = max(ev.start_t, t)
            out.append(ev)
        self._open.clear()
        return out

    def open_event(self, track_id: int) -> HornetEvent | None:
        return self._open.get(track_id)

    def is_hawking(self, tr: Track) -> bool:
        if tr.vespid_score < self.min_vespid_score:
            return False
        species = tr.species
        if species not in VESPIDS and species != "vespid":
            return False
        if tr.duration < self.min_seconds:
            return False
        pts = tr.window(self.min_seconds)
        if len(pts) < 3:
            return False
        if self.hawk_zone is not None and not all(point_in_polygon(p, self.hawk_zone) for p in pts):
            return False
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        spread = max(math.hypot(p[0] - cx, p[1] - cy) for p in pts)
        return spread <= self.max_spread * self.frame_diag


def pressure_level(velutina_events_per_hour: float, check_per_hour: float = 3.0) -> str:
    """Fixed levels for an invasive predator: none / present / check."""
    if velutina_events_per_hour <= 0:
        return "none"
    if velutina_events_per_hour < check_per_hour:
        return "present"
    return "check"
