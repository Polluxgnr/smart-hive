"""Detector + tracker + crop classifier (Ultralytics YOLO11n, ByteTrack)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import cv2
import numpy as np

from .tracks import SPECIES

log = logging.getLogger(__name__)

DETECTOR_CLASSES = ("bee", "vespid")


@dataclass
class Detection:
    track_id: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 pixels
    label: str  # "bee" or "vespid"
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def size(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(x2 - x1, y2 - y1)


def tracker_config() -> str:
    return str(resources.files("smarthive") / "assets" / "bytetrack_hive.yaml")


class Perception:
    """Runs the detector with ByteTrack, and the crop classifier on demand."""

    def __init__(
        self,
        detector: str | Path,
        classifier: str | Path | None = None,
        imgsz: int = 640,
        conf: float = 0.25,
        cls_imgsz: int = 96,
        device: str = "cpu",
    ):
        from ultralytics import YOLO

        if not Path(detector).exists():
            raise FileNotFoundError(
                f"detector weights not found: {detector}. Run `python tools/fetch_weights.py` "
                "or train them (see docs/report.md)."
            )
        self.detector = YOLO(str(detector), task="detect")
        self.det_names = {int(k): v for k, v in self.detector.names.items()}
        self.classifier = None
        if classifier and Path(classifier).exists():
            self.classifier = YOLO(str(classifier), task="classify")
            names = {int(k): v for k, v in self.classifier.names.items()}
            # Map classifier outputs onto SPECIES order (missing classes get 0).
            self._cls_index = [next((i for i, n in names.items() if n == s), None) for s in SPECIES]
        elif classifier:
            log.warning("classifier weights not found (%s): species from the detector only", classifier)
        self.imgsz, self.conf, self.cls_imgsz, self.device = imgsz, conf, cls_imgsz, device

    def reset(self) -> None:
        """Forget tracker state (call between unrelated videos)."""
        predictor = getattr(self.detector, "predictor", None)
        if predictor is not None and getattr(predictor, "trackers", None):
            for tr in predictor.trackers:
                tr.reset()

    def track(self, frame: np.ndarray) -> list[Detection]:
        res = self.detector.track(
            frame,
            persist=True,
            tracker=tracker_config(),
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        out = []
        for b, i, c, s in zip(xyxy, ids, cls, confs, strict=True):
            label = self.det_names.get(int(c), "bee")
            out.append(Detection(int(i), (float(b[0]), float(b[1]), float(b[2]), float(b[3])), label, float(s)))
        return out

    def classify(self, frame: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> list[list[float]]:
        """Species probabilities (SPECIES order) for each box."""
        if self.classifier is None or not boxes:
            return []
        crops = [crop_box(frame, b) for b in boxes]
        results = self.classifier.predict(crops, imgsz=self.cls_imgsz, device=self.device, verbose=False)
        out = []
        for r in results:
            p = r.probs.data.cpu().numpy()
            out.append([float(p[i]) if i is not None else 0.0 for i in self._cls_index])
        return out


def crop_box(frame: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.25) -> np.ndarray:
    """Square crop around a box with some context, clipped to the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * pad)
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    c, d = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = frame[b:d, a:c]
    if crop.size == 0:
        crop = np.zeros((8, 8, 3), np.uint8)
    return crop


def letterbox_square(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8)
    y, x = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    out[y : y + img.shape[0], x : x + img.shape[1]] = img
    return out
