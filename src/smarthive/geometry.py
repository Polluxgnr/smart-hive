"""Small 2-D geometry helpers (pure Python, no dependencies)."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

Point = tuple[float, float]


def point_in_polygon(p: Point, polygon: Sequence[Point]) -> bool:
    """Ray-casting test. Points exactly on an edge may fall either side."""
    x, y = p
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def signed_distance_to_line(p: Point, a: Point, b: Point) -> float:
    """Signed distance from ``p`` to the infinite line a->b.

    Positive on the left of the direction a->b in image coordinates (y down),
    i.e. ``cross(b - a, p - a) < 0`` maps to positive. With a line drawn from
    left to right across the image, positive means *above* the line.
    """
    ax, ay = a
    bx, by = b
    length = math.hypot(bx - ax, by - ay)
    if length == 0:
        raise ValueError("line endpoints must differ")
    cross = (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
    return -cross / length


def projection_t(p: Point, a: Point, b: Point) -> float:
    """Position of the projection of ``p`` on segment a->b (0 at a, 1 at b)."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0:
        return 0.0
    return ((p[0] - ax) * dx + (p[1] - ay) * dy) / denom


def scale_points(points: Sequence[Sequence[float]], width: float, height: float) -> list[Point]:
    """Convert normalised (0-1) coordinates to pixels."""
    return [(float(x) * width, float(y) * height) for x, y in points]


def path_length(points: Sequence[Point]) -> float:
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def turning_index(points: Sequence[Point], min_step: float = 1e-6) -> float:
    """Mean absolute heading change (radians) between successive moves.

    Steps shorter than ``min_step`` are ignored (a stationary bee has no heading).
    0 for a straight path, about pi/2 for a random walk.
    """
    headings = []
    for i in range(len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        if math.hypot(dx, dy) >= min_step:
            headings.append(math.atan2(dy, dx))
    if len(headings) < 2:
        return 0.0
    total = 0.0
    for h1, h2 in itertools.pairwise(headings):
        d = (h2 - h1 + math.pi) % (2 * math.pi) - math.pi
        total += abs(d)
    return total / (len(headings) - 1)
