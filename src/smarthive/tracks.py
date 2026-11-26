"""Per-track history: positions, sizes and the track-level class vote."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .geometry import Point, path_length, turning_index

# Classes of the crop classifier, in a fixed order.
SPECIES = ("bee", "bee_pollen", "vespa_velutina", "vespa_crabro", "wasp")
VESPIDS = ("vespa_velutina", "vespa_crabro", "wasp")


@dataclass
class Track:
    track_id: int
    first_t: float
    points: deque = field(default_factory=lambda: deque(maxlen=600))
    times: deque = field(default_factory=lambda: deque(maxlen=600))
    sizes: deque = field(default_factory=lambda: deque(maxlen=600))
    det_vespid: float = 0.0  # running sum of detector "vespid" confidence
    det_n: int = 0
    det_vespid_n: int = 0  # frames where the detector said "vespid"
    prob_sum: list[float] = field(default_factory=lambda: [0.0] * len(SPECIES))
    prob_n: int = 0
    last_classified_frame: int = -(10**9)
    last_t: float = 0.0

    def add(self, t: float, p: Point, size: float, det_is_vespid: bool, det_conf: float) -> None:
        self.points.append(p)
        self.times.append(t)
        self.sizes.append(size)
        self.last_t = t
        self.det_n += 1
        if det_is_vespid:
            self.det_vespid += det_conf
            self.det_vespid_n += 1

    def add_probs(self, probs: list[float], frame_idx: int) -> None:
        for i, v in enumerate(probs):
            self.prob_sum[i] += float(v)
        self.prob_n += 1
        self.last_classified_frame = frame_idx

    # -- votes -------------------------------------------------------------
    @property
    def probs(self) -> dict[str, float]:
        if self.prob_n == 0:
            return {}
        return {name: s / self.prob_n for name, s in zip(SPECIES, self.prob_sum, strict=True)}

    @property
    def vespid_score(self) -> float:
        """Probability that this track is a hornet or wasp.

        Uses the classifier vote when available, else the detector's class.
        """
        p = self.probs
        if p:
            return sum(p[s] for s in VESPIDS)
        return self.det_vespid / self.det_n if self.det_n else 0.0

    @property
    def detector_vespid_share(self) -> float:
        """Share of this track's detections that the detector labelled 'vespid'."""
        return self.det_vespid_n / self.det_n if self.det_n else 0.0

    @property
    def species(self) -> str:
        """Best label: one of SPECIES, or 'vespid' when only the detector spoke."""
        p = self.probs
        if not p:
            return "vespid" if self.vespid_score >= 0.5 else "bee"
        if self.vespid_score >= 0.5:
            return max(VESPIDS, key=lambda s: p[s])
        return "bee_pollen" if p["bee_pollen"] > p["bee"] else "bee"

    @property
    def species_confidence(self) -> float:
        p = self.probs
        if not p:
            return self.vespid_score if self.species == "vespid" else 1 - self.vespid_score
        if self.species in VESPIDS:
            return p[self.species]
        return p["bee"] + p["bee_pollen"]

    @property
    def carries_pollen(self) -> bool:
        return self.species == "bee_pollen"

    # -- motion features ---------------------------------------------------
    @property
    def duration(self) -> float:
        return self.last_t - self.first_t

    @property
    def body_length(self) -> float:
        if not self.sizes:
            return 1.0
        s = sorted(self.sizes)
        return max(s[len(s) // 2], 1.0)

    def window(self, seconds: float) -> list[Point]:
        if not self.times:
            return []
        t_end = self.times[-1]
        return [p for p, t in zip(self.points, self.times, strict=True) if t_end - t <= seconds]

    def net_displacement(self, seconds: float | None = None) -> float:
        pts = list(self.points) if seconds is None else self.window(seconds)
        return math.dist(pts[0], pts[-1]) if len(pts) > 1 else 0.0

    def path(self, seconds: float | None = None) -> float:
        pts = list(self.points) if seconds is None else self.window(seconds)
        return path_length(pts)

    def turning(self) -> float:
        return turning_index(list(self.points), min_step=0.15 * self.body_length)

    def is_stationary(self, seconds: float = 20.0) -> bool:
        """Has stayed within one body length for at least ``seconds``."""
        if self.duration < seconds:
            return False
        return self.net_displacement(seconds) < self.body_length

    def is_erratic(self, min_points: int = 15, threshold: float = 1.2) -> bool:
        """Zig-zag motion: high mean heading change while actually moving."""
        if len(self.points) < min_points:
            return False
        if self.path() < 4 * self.body_length:
            return False
        return self.turning() >= threshold
