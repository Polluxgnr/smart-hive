"""Hornet module: turn vespid sightings into *hawking* events.

A hornet flying past is not an attack. *Vespa velutina* hunts by hovering in
front of the entrance ("hawking") and catching returning foragers. Fast, erratic
flight breaks tracks into short pieces, so the rule works on **presence over a
sliding window** rather than on one track:

* a *sighting* is a visible track judged to be a vespid: the crop classifier's
  track vote says hornet/wasp (>= 0.6) and either the detector agrees on part of
  the track or the classifier is very sure (>= 0.8);
* an event opens when sightings cover at least ``min_seconds`` of the last
  ``min_seconds + 1`` seconds, on at least ``min_presence`` of the analysed
  frames, inside the hawking zone, within a small area (not flying through) and
  with some movement (a bee cluster sitting at the slot is not a hornet);
* it closes after ``gap_seconds`` without a sighting.

See docs/design.md, section 6.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .geometry import Point, point_in_polygon
from .tracks import SPECIES, VESPIDS, Track


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
    last_seen: float = 0.0
    _struck: set = field(default_factory=set, repr=False)
    _votes: list = field(default_factory=lambda: [0.0] * len(SPECIES), repr=False)

    @property
    def duration(self) -> float:
        return (self.end_t if self.end_t is not None else self.last_seen) - self.start_t

    @property
    def is_velutina(self) -> bool:
        return self.species == "vespa_velutina"


@dataclass
class HornetMonitor:
    hawk_zone: Sequence[Point] | None  # pixels; None = whole frame
    frame_diag: float
    min_seconds: float = 3.0
    max_spread: float = 0.30  # fraction of the frame diagonal
    min_vespid_score: float = 0.6  # classifier track vote (sum of hornet / wasp probabilities)
    sure_vespid_score: float = 0.8  # classifier alone is enough above this
    min_detector_share: float = 0.3  # otherwise the detector must agree on part of the track
    min_presence: float = 0.4  # share of analysed frames with a sighting in the window
    min_motion_body_lengths: float = 1.5  # hovering means moving on the spot, not sitting
    gap_seconds: float = 2.0
    strike_body_lengths: float = 1.5
    _next_id: int = 1
    _current: HornetEvent | None = None
    _frames: deque = field(default_factory=deque)  # (t, [(track_id, point, body_length)])

    # -- per-track judgement --------------------------------------------------------------
    def is_sighting(self, tr: Track) -> bool:
        if tr.prob_n == 0:
            return False  # never decide on the detector alone: it confuses close, blurred bees
        score = tr.vespid_score
        if score < self.min_vespid_score:
            return False
        if score < self.sure_vespid_score and tr.detector_vespid_share < self.min_detector_share:
            return False
        return not (self.hawk_zone is not None and tr.points and not point_in_polygon(tr.points[-1], self.hawk_zone))

    # -- frame update --------------------------------------------------------------------
    def update(
        self,
        t: float,
        tracks: Mapping[int, Track],
        visible: set[int],
        bee_points: Sequence[tuple[int, Point]] = (),
    ) -> tuple[list[HornetEvent], list[HornetEvent]]:
        """Process one analysed frame. Returns (opened, closed) events."""
        sightings = [
            (i, tracks[i].points[-1], tracks[i].body_length)
            for i in visible
            if tracks[i].points and self.is_sighting(tracks[i])
        ]
        window = self.min_seconds + 1.0
        self._frames.append((t, sightings))
        while self._frames and t - self._frames[0][0] > window:
            self._frames.popleft()

        opened: list[HornetEvent] = []
        closed: list[HornetEvent] = []
        ev = self._current
        if ev is None and self._presence_ok():
            i, _, _ = sightings[-1] if sightings else self._last_sighting()
            ev = self._current = HornetEvent(self._next_id, i, self._first_sighting_t(), "vespid", 0.0)
            ev.last_seen = t
            self._next_id += 1
            for _, s in self._frames:
                for tid, _, _ in s:
                    self._vote(ev, tracks.get(tid))
            opened.append(ev)
        elif ev is not None:
            if sightings:
                ev.last_seen = t
                ev.track_id = sightings[-1][0]
                for tid, _, _ in sightings:
                    self._vote(ev, tracks.get(tid))
            elif t - ev.last_seen > self.gap_seconds:
                ev.end_t = ev.last_seen
                closed.append(ev)
                self._current = None
                self._frames.clear()
                ev = None
        if ev is not None:
            for _, (hx, hy), body in sightings:
                reach = self.strike_body_lengths * body
                for bid, (bx, by) in bee_points:
                    if bid not in ev._struck and math.hypot(bx - hx, by - hy) <= reach:
                        ev._struck.add(bid)
                        ev.strikes += 1
        return opened, closed

    def close_all(self, t: float) -> list[HornetEvent]:
        if self._current is None:
            return []
        ev, self._current = self._current, None
        ev.end_t = ev.last_seen
        self._frames.clear()
        return [ev]

    def open_event(self, track_id: int | None = None) -> HornetEvent | None:
        """The event in progress (optionally only if ``track_id`` is one of its recent sightings)."""
        if self._current is None or track_id is None:
            return self._current
        recent = {i for _, s in list(self._frames)[-3:] for i, _, _ in s}
        return self._current if track_id in recent else None

    # -- helpers ---------------------------------------------------------------------------
    def _presence_ok(self) -> bool:
        if not self._frames:
            return False
        with_s = [(t, s) for t, s in self._frames if s]
        if len(with_s) < 3 or len(with_s) / len(self._frames) < self.min_presence:
            return False
        if with_s[-1][0] - with_s[0][0] < self.min_seconds:
            return False
        pts = [p for _, s in with_s for _, p, _ in s]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        if max(math.hypot(p[0] - cx, p[1] - cy) for p in pts) > self.max_spread * self.frame_diag:
            return False  # flying through, not hovering
        body = sorted(b for _, s in with_s for _, _, b in s)[len(pts) // 2]
        path = sum(math.dist(a[1][0][1], b[1][0][1]) for a, b in itertools.pairwise(with_s))
        return path >= self.min_motion_body_lengths * body

    def _first_sighting_t(self) -> float:
        return next(t for t, s in self._frames if s)

    def _last_sighting(self) -> tuple[int, Point, float]:
        return next(s[-1] for _, s in reversed(self._frames) if s)

    def _vote(self, ev: HornetEvent, tr: Track | None) -> None:
        if tr is None or not tr.probs:
            return
        for k, name in enumerate(SPECIES):
            ev._votes[k] += tr.probs[name]
        vesp = {name: ev._votes[SPECIES.index(name)] for name in VESPIDS}
        ev.species = max(vesp, key=vesp.get)  # type: ignore[arg-type]
        total = sum(ev._votes)
        ev.confidence = vesp[ev.species] / total if total else 0.0


def pressure_level(velutina_events_per_hour: float, check_per_hour: float = 3.0) -> str:
    """Fixed levels for an invasive predator: none / present / check."""
    if velutina_events_per_hour <= 0:
        return "none"
    if velutina_events_per_hour < check_per_hour:
        return "present"
    return "check"
