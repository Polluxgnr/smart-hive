"""Video helpers: browser-friendly H.264 re-encoding and GIF export (optional ffmpeg)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_h264(src: str | Path, dst: str | Path | None = None, crf: int = 28) -> Path:
    """Re-encode an OpenCV mp4v file to H.264 so browsers can play it.

    Returns the new path, or ``src`` unchanged if ffmpeg is not available.
    """
    src = Path(src)
    exe = ffmpeg_exe()
    if exe is None:
        log.info("imageio-ffmpeg not installed: keeping %s as mp4v", src.name)
        return src
    dst = Path(dst) if dst else src.with_name(src.stem + "_h264.mp4")
    cmd = [
        exe,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)
    return dst


def to_gif(
    src: str | Path, dst: str | Path, width: int = 640, fps: int = 8, start: float = 0.0, duration: float = 8.0
) -> Path | None:
    exe = ffmpeg_exe()
    if exe is None:
        return None
    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];"
        f"[b][p]paletteuse=dither=bayer:bayer_scale=5"
    )
    cmd = [exe, "-y", "-loglevel", "error", "-ss", str(start), "-t", str(duration), "-i", str(src), "-vf", vf, str(dst)]
    subprocess.run(cmd, check=True)
    return Path(dst)
