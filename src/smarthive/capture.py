"""Frame sources: OpenCV (USB webcams, files, RTSP) and the Raspberry Pi camera.

The Pi Camera Module is driven by libcamera, which OpenCV cannot open directly
on Raspberry Pi OS Bookworm; ``source: picamera`` uses Picamera2 instead
(install with ``sudo apt install python3-picamera2`` and create the virtual
environment with ``--system-site-packages``, which install_pi.sh does).
"""

from __future__ import annotations

import cv2
import numpy as np


class PicameraCapture:
    """Minimal cv2.VideoCapture look-alike on top of Picamera2."""

    def __init__(self, width: int = 1280, height: int = 720, fps: float = 15.0):
        from picamera2 import Picamera2  # only on Raspberry Pi OS

        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}, controls={"FrameRate": fps}
        )
        self.cam.configure(config)
        self.cam.start()
        self.fps = fps

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self.cam.capture_array()
        return frame is not None, frame  # RGB888 in Picamera2 is BGR byte order, as OpenCV expects

    def get(self, prop: int) -> float:
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def set(self, prop: int, value: float) -> bool:
        return False

    def release(self) -> None:
        self.cam.stop()
        self.cam.close()


def open_source(source, width: int = 1280, height: int = 720, fps: float = 15.0):
    if isinstance(source, str) and source.lower() in ("picamera", "picamera2", "pi"):
        return PicameraCapture(width, height, fps)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    return cv2.VideoCapture(source)


def is_live(source) -> bool:
    if isinstance(source, int):
        return True
    s = str(source).lower()
    return s.isdigit() or s in ("picamera", "picamera2", "pi") or s.startswith(("rtsp://", "http://", "https://"))
