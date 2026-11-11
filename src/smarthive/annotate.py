"""Drawing helpers for the annotated video and hornet snapshots."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .geometry import Point
from .i18n import t

# BGR colours
COLOURS = {
    "bee": (60, 200, 255),
    "bee_pollen": (0, 140, 255),
    "vespa_velutina": (40, 40, 230),
    "vespa_crabro": (200, 90, 200),
    "wasp": (220, 160, 40),
    "vespid": (40, 40, 230),
}
ZONE = (90, 220, 90)
HAWK = (60, 60, 230)


def draw_polygon(img: np.ndarray, pts: Sequence[Point], colour, thickness: int = 2, alpha: float = 0.12) -> None:
    arr = np.array([[int(x), int(y)] for x, y in pts], np.int32)
    if alpha > 0:
        overlay = img.copy()
        cv2.fillPoly(overlay, [arr], colour)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [arr], True, colour, thickness, cv2.LINE_AA)


def draw_line(img: np.ndarray, a: Point, b: Point, colour=ZONE, thickness: int = 2) -> None:
    cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colour, thickness, cv2.LINE_AA)


def draw_box(img: np.ndarray, box, label: str, colour, thick: int = 2) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, thick, cv2.LINE_AA)
    if label:
        scale = max(0.4, img.shape[0] / 1400)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_panel(img: np.ndarray, lines: list[str], warn: str | None = None) -> None:
    scale = max(0.5, img.shape[0] / 1000)
    pad = int(10 * scale)
    lh = int(30 * scale)
    width = int(max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] for s in lines) + 2 * pad)
    height = lh * len(lines) + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (pad, lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    if warn:
        h, w = img.shape[:2]
        bar = int(46 * scale)
        cv2.rectangle(img, (0, h - bar), (w, h), (30, 30, 200), -1)
        cv2.putText(
            img,
            warn,
            (pad, h - int(14 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale * 1.1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def species_label(species: str, lang: str) -> str:
    # OpenCV's Hershey fonts have no accents: strip them for on-video text.
    text = t(f"species.{species}", lang)
    return text.translate(str.maketrans("éèêàçôîïù", "eeeacoiiu"))
