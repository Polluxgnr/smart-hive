"""Count bees going IN and OUT of the hive from tracked positions.

Two rules (see docs/design.md, section 4):

* ``ZoneRule``: a polygon drawn around the entrance slot. Bees entering the hive
  vanish into the slot, so a track that *ends* inside the zone after being seen
  outside is IN; a track that *starts* inside the zone and ends outside is OUT.
* ``LineRule``: a line across the board with hysteresis. A track is counted when
  its centre goes from more than ``margin`` on one side to more than ``margin``
  on the other side.

The counter is independent of the detector and tracker: it consumes
``{track_id: (x, y)}`` per frame, which keeps it easy to test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from .geometry import Point, point_in_polygon, signed_distance_to_line

Direction = Literal["in", "out"]


@dataclass(frozen=True)
class Crossing:
    track_id: int
    direction: Direction
    t: float


@dataclass
class _Track:
    first_t: float
    last_t: float
    first_inside: bool
    last_inside: bool
    seen_outside: bool
    n: int = 1
    side: int = 0  # line rule: -1 outside, +1 inside, 0 unknown yet
    last_point: Point = (0.0, 0.0)
    max_travel: float = 0.0
    first_point: Point = (0.0, 0.0)


class ZoneRule:
    """Entrance-zone rule. ``zone`` is a polygon in pixels."""

    def __init__(self, zone: Sequence[Point]):
        if len(zone) < 3:
            raise ValueError("zone needs at least 3 points")
        self.zone: list[Point] = [(float(p[0]), float(p[1])) for p in zone]

    def inside(self, p: Point) -> bool:
        return point_in_polygon(p, self.zone)


class LineRule:
    """Directed-line rule.

    ``inside_point`` is any pixel on the hive side of the line, so the user does
    not have to think about line orientation.
    """

    def __init__(self, a: Point, b: Point, inside_point: Point, margin: float):
        self.a, self.b = a, b
        self.margin = float(margin)
        d = signed_distance_to_line(inside_point, a, b)
        if d == 0:
            raise ValueError("inside_point must not lie on the line")
        self._sign = 1.0 if d > 0 else -1.0

    def distance(self, p: Point) -> float:
        """Signed distance, positive on the hive side."""
        return self._sign * signed_distance_to_line(p, self.a, self.b)


@dataclass
class EntranceCounter:
    """Turns per-frame track positions into IN / OUT crossings.

    ``max_age`` (seconds) is how long a track may be unseen before it is
    considered finished. ``min_points`` ignores very short tracks (noise).
    """

    rule: ZoneRule | LineRule
    max_age: float = 1.5
    min_points: int = 3
    min_travel: float = 0.0
    _tracks: dict[int, _Track] = field(default_factory=dict)
    total_in: int = 0
    total_out: int = 0

    def update(self, t: float, observations: Mapping[int, Point]) -> list[Crossing]:
        events: list[Crossing] = []
        for tid, p in observations.items():
            events.extend(self._observe(int(tid), (float(p[0]), float(p[1])), t))
        stale = [tid for tid, tr in self._tracks.items() if t - tr.last_t > self.max_age]
        for tid in stale:
            ev = self._finish(tid)
            if ev:
                events.append(ev)
        self._tally(events)
        return events

    def flush(self) -> list[Crossing]:
        """Finish every open track (end of video / shutdown)."""
        events = [ev for tid in list(self._tracks) if (ev := self._finish(tid))]
        self._tally(events)
        return events

    @property
    def open_tracks(self) -> int:
        return len(self._tracks)

    # -- internals ---------------------------------------------------------
    def _tally(self, events: list[Crossing]) -> None:
        for ev in events:
            if ev.direction == "in":
                self.total_in += 1
            else:
                self.total_out += 1

    def _observe(self, tid: int, p: Point, t: float) -> list[Crossing]:
        tr = self._tracks.get(tid)
        if isinstance(self.rule, ZoneRule):
            inside = self.rule.inside(p)
            if tr is None:
                self._tracks[tid] = _Track(t, t, inside, inside, not inside, last_point=p, first_point=p)
                return []
            tr.last_t, tr.last_inside, tr.n, tr.last_point = t, inside, tr.n + 1, p
            tr.seen_outside |= not inside
            tr.max_travel = max(tr.max_travel, _dist(p, tr.first_point))
            return []

        # Line rule: emit as soon as the hysteresis band is crossed.
        d = self.rule.distance(p)
        side = 1 if d > self.rule.margin else -1 if d < -self.rule.margin else 0
        if tr is None:
            tr = self._tracks[tid] = _Track(t, t, False, False, False, side=side, last_point=p, first_point=p)
            return []
        tr.last_t, tr.n, tr.last_point = t, tr.n + 1, p
        tr.max_travel = max(tr.max_travel, _dist(p, tr.first_point))
        if side == 0 or side == tr.side:
            return []
        previous, tr.side = tr.side, side
        if previous == 0:
            return []
        return [Crossing(tid, "in" if side > 0 else "out", t)]

    def _finish(self, tid: int) -> Crossing | None:
        tr = self._tracks.pop(tid)
        if not isinstance(self.rule, ZoneRule):
            return None
        if tr.n < self.min_points or tr.max_travel < self.min_travel:
            return None
        if tr.last_inside and not tr.first_inside and tr.seen_outside:
            return Crossing(tid, "in", tr.last_t)
        if tr.first_inside and not tr.last_inside:
            return Crossing(tid, "out", tr.last_t)
        return None


def _dist(a: Point, b: Point) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
