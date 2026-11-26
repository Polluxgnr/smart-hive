"""Build the small, redistributable sample set in samples/.

Real footage
    * Three 12 s clips cut from the landing-board tracking videos
      (Sledevič et al., Mendeley Data, CC BY 4.0), downscaled to 960x540.
    * "Honey bees fanning (1)" from Wikimedia Commons (BorekT, CC BY-SA 4.0).
Derived (synthetic, same licence as their source, marked as modified)
    * night, rain and empty-entrance variants of a real clip;
    * hornet / wasp composites: a real entrance clip with an insect cut out of
      an openly licensed iNaturalist photo (GrabCut mask) flying over it,
      hovering (V. velutina, V. crabro) or passing through (wasp).

Every output file is written to samples/SOURCES.md with origin and licence.

Usage:
    python tools/make_samples.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import urllib.request
from pathlib import Path

import cv2
import numpy as np

LB = Path("data/raw/landing_board/tracking")
COMMONS_FANNING = (
    "https://upload.wikimedia.org/wikipedia/commons/a/ab/Honey_bees_fanning_%281%29.webm",
    "https://commons.wikimedia.org/wiki/File:Honey_bees_fanning_(1).webm",
)
LB_CREDIT = (
    "T. Sledevič et al., *Labeled dataset for bee detection and direction estimation on beehive landing boards*, "
    "Mendeley Data v6, doi:10.17632/8gb9r2yhfc.6"
)
SIZE = (960, 540)


def read_clip(path: Path, start_s: float, seconds: float, fps_out: float = 25.0) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps / fps_out))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps))
    frames = []
    for i in range(int(seconds * fps)):
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def write_clip(frames: list[np.ndarray], dst: Path, fps: float = 25.0, crf: int = 30) -> None:
    """H.264 via imageio-ffmpeg when available (small, plays in browsers), else mp4v."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    try:
        import subprocess

        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                exe,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(tmp),
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(dst),
            ],
            check=True,
        )
        tmp.unlink()
    except Exception:
        tmp.replace(dst)
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


# -- derived variants -----------------------------------------------------------------
def night(frames: list[np.ndarray], seed: int = 0) -> list[np.ndarray]:
    """Dusk / IR-like footage: dark, grey, noisy."""
    rng = np.random.default_rng(seed)
    out = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        g = np.power(g, 1.8) * 0.35
        g = g + rng.normal(0, 0.035, g.shape)
        g = np.clip(g * 255, 0, 255).astype(np.uint8)
        out.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
    return out


def rain(frames: list[np.ndarray], seed: int = 0) -> list[np.ndarray]:
    """Rain streaks over a darker, lower-contrast scene."""
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    out = []
    for f in frames:
        base = cv2.addWeighted(f, 0.65, np.full_like(f, 90), 0.35, 0)
        streaks = np.zeros((h, w), np.float32)
        for _ in range(350):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            length = int(rng.integers(12, 30))
            cv2.line(streaks, (x, y), (x + length // 5, y + length), 1.0, 1, cv2.LINE_AA)
        streaks = cv2.GaussianBlur(streaks, (3, 3), 0)[..., None]
        rain_f = base.astype(np.float32) * (1 - 0.5 * streaks) + 220 * 0.5 * streaks
        out.append(np.clip(rain_f, 0, 255).astype(np.uint8))
    return out


def empty(frames: list[np.ndarray], n: int, seed: int = 0) -> list[np.ndarray]:
    """Temporal median = the board without bees; add sensor noise and slight light drift."""
    rng = np.random.default_rng(seed)
    bg = np.median(np.stack(frames[:: max(1, len(frames) // 60)]), axis=0).astype(np.float32)
    out = []
    for i in range(n):
        gain = 1 + 0.04 * math.sin(i / 30)
        f = bg * gain + rng.normal(0, 2.0, bg.shape)
        out.append(np.clip(f, 0, 255).astype(np.uint8))
    return out


# -- insect sprites ----------------------------------------------------------------------
def cutout(img: np.ndarray, box: list[float]) -> tuple[np.ndarray, np.ndarray] | None:
    """GrabCut the insect inside ``box``. Returns (crop BGR, alpha 0-1) or None if the mask looks wrong."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    pad = int(0.08 * max(x2 - x1, y2 - y1))
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    mask = np.zeros((H, W), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, (x1, y1, x2 - x1, y2 - y1), bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)[y1:y2, x1:x2]
    frac = float(m.mean())
    if not 0.12 <= frac <= 0.7:
        return None
    m = cv2.GaussianBlur(m, (5, 5), 0)
    return img[y1:y2, x1:x2].copy(), m


def credit(sp: dict) -> str:
    """'iNaturalist photo N by NAME' (CC BY) or 'iNaturalist photo N, no rights reserved' (CC0)."""
    who = sp["attribution"]
    who = " by " + who[4:].split(", some rights reserved")[0] if who.startswith("(c) ") else ", " + who
    return f"iNaturalist photo {sp['photo_id']}{who} ([observation]({sp['observation_url']}))"


def pick_sprites(inat: Path, boxes: dict, attribution: dict, folder: str, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    keys = [k for k, v in boxes.items() if k.startswith(folder + "/") and len(v) == 1]
    keys = [k for k in keys if attribution.get(k, {}).get("licence") in ("cc0", "cc-by")]
    rng.shuffle(keys)
    out = []
    for k in keys:
        img = cv2.imread(str(inat / k))
        if img is None:
            continue
        b = boxes[k][0]
        if (b[2] - b[0]) < 80 or (b[3] - b[1]) < 80:  # need enough pixels for a clean mask
            continue
        res = cutout(img, b)
        if res is None:
            continue
        out.append({"key": k, "crop": res[0], "alpha": res[1], **attribution[k]})
        if len(out) >= n:
            break
    return out


def composite(frames: list[np.ndarray], sprite: dict, size_px: int, path_fn, seed: int = 0) -> list[np.ndarray]:
    rng = random.Random(seed)
    crop, alpha = sprite["crop"], sprite["alpha"]
    # Scale on the insect itself (the mask), not on the photo box with its margins:
    # ``size_px`` is the insect's length in the frame.
    ys, xs = np.nonzero(alpha > 0.5)
    extent = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1 if len(xs) else max(crop.shape[:2])
    s = size_px / extent
    crop = cv2.resize(
        crop, (max(4, int(crop.shape[1] * s)), max(4, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA
    )
    alpha = cv2.resize(alpha, (crop.shape[1], crop.shape[0]))
    out = []
    H, W = frames[0].shape[:2]
    for i, f in enumerate(frames):
        pos = path_fn(i / len(frames), i)
        f = f.copy()
        if pos is not None:
            cx, cy, angle = pos
            M = cv2.getRotationMatrix2D((crop.shape[1] / 2, crop.shape[0] / 2), angle, 1.0)
            c = cv2.warpAffine(crop, M, (crop.shape[1], crop.shape[0]), borderMode=cv2.BORDER_REFLECT)
            a = cv2.warpAffine(alpha, M, (crop.shape[1], crop.shape[0]))
            if rng.random() < 0.6:  # wing beat / motion blur
                k = rng.choice([3, 5])
                c = cv2.blur(c, (k, 1))
            x1, y1 = int(cx - c.shape[1] / 2), int(cy - c.shape[0] / 2)
            a1, b1, a2, b2 = max(0, x1), max(0, y1), min(W, x1 + c.shape[1]), min(H, y1 + c.shape[0])
            if a2 > a1 and b2 > b1:
                m = a[b1 - y1 : b2 - y1, a1 - x1 : a2 - x1, None]
                region = f[b1:b2, a1:a2].astype(np.float32)
                patch = c[b1 - y1 : b2 - y1, a1 - x1 : a2 - x1].astype(np.float32)
                f[b1:b2, a1:a2] = (m * patch + (1 - m) * region).astype(np.uint8)
        out.append(f)
    return out


def hover_path(cx: float, cy: float, rx: float, ry: float, start: float = 0.15, end: float = 0.9):
    """Enter from the side, hover facing the entrance with small loops, leave."""

    def fn(u: float, i: int):
        if u < start or u > end:
            return None
        v = (u - start) / (end - start)
        if v < 0.12:  # arrival
            k = v / 0.12
            return (cx - 500 * (1 - k), cy + 120 * (1 - k), 10)
        if v > 0.9:  # departure
            k = (v - 0.9) / 0.1
            return (cx + 600 * k, cy - 80 * k, -10)
        return (cx + rx * math.sin(i / 7.0), cy + ry * math.sin(i / 4.3), 8 * math.sin(i / 5.0))

    return fn


def pass_path(y: float, width: int, start: float = 0.3, duration: float = 0.12):
    def fn(u: float, i: int):
        if not start <= u <= start + duration:
            return None
        k = (u - start) / duration
        return (-60 + (width + 120) * k, y + 40 * math.sin(k * 6), 90)

    return fn


# -- manifest -----------------------------------------------------------------------------
CLIPS = {
    # file: (background video, title, what to expect)
    "entrance_guards.mp4": (
        "20230609b-def",
        "Guards at the entrance / Gardiennes",
        "Real footage. Few foragers, guard bees near the slot.",
    ),
    "entrance_busy.mp4": ("20230711a-fan", "Busy board / Planche animée", "Real footage. Heavy traffic."),
    "entrance_morning.mp4": ("20230711b-fan", "Normal traffic / Trafic normal", "Real footage."),
    "night.mp4": (
        "20230711b-fan",
        "Dusk (synthetic) / Crépuscule (synthétique)",
        "Low light: expect fewer detections.",
    ),
    "rain.mp4": ("20230711a-fan", "Rain (synthetic) / Pluie (synthétique)", "Rain streaks over a busy board."),
    "empty_entrance.mp4": (
        "20230711a-fan",
        "Empty entrance (synthetic) / Entrée vide (synthétique)",
        "No bees: nothing should be counted.",
    ),
    "fanning_commons.mp4": (
        None,
        "Fanning bees / Abeilles ventileuses",
        "Real footage from another hive and camera angle (Wikimedia Commons).",
    ),
    "hornet_velutina_hawking.mp4": (
        "20230711b-fan",
        "Asian hornet hawking (composite) / Frelon asiatique",
        "Expected: one Vespa velutina event and an alert.",
    ),
    "hornet_velutina_busy.mp4": (
        "20230711a-fan",
        "Asian hornet over a busy board (composite)",
        "Expected: one Vespa velutina event and an alert.",
    ),
    "hornet_crabro_hawking.mp4": (
        "20230609b-def",
        "European hornet (composite) / Frelon européen",
        "Expected: a Vespa crabro event, logged without alert.",
    ),
    "wasp_passing.mp4": (
        "20230711b-fan",
        "Wasp passing (composite) / Guêpe de passage",
        "Expected: no event (it does not hover).",
    ),
}
# Entrance zone for the Commons clip (other hive), read off its first frame.
COMMONS_ZONE = [[0.05, 0.30], [0.95, 0.30], [0.95, 0.45], [0.05, 0.45]]


def slot_zone_norm(video: str) -> list[list[float]]:
    """Band along the slot edge of the annotated board polygon, normalised to 0-1."""
    import re

    text = (LB / f"entrance_zone_{video}.txt").read_text()
    pts = np.array([[float(x), float(y)] for x, y in re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)])
    cap = cv2.VideoCapture(str(LB / f"{video}.mp4"))
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    p0, p1, p2, p3 = pts[:4]
    zone = [p0 - 0.10 * (p3 - p0), p1 - 0.10 * (p2 - p1), p1 + 0.25 * (p2 - p1), p0 + 0.25 * (p3 - p0)]
    return [[round(float(x / w), 4), round(float(y / h), 4)] for x, y in zone]


def write_manifest(out: Path, files: list[str]) -> None:
    import yaml

    clips = []
    for f in files:
        name = Path(f).name
        video, title, note = CLIPS[name]
        zone = slot_zone_norm(video) if video else COMMONS_ZONE
        clips.append({"file": f, "title": title, "note": note, "entrance": {"mode": "zone", "zone": zone}})
    header = "# Generated by tools/make_samples.py: one entry per sample clip, with its entrance zone.\n"
    (out / "manifest.yaml").write_text(
        header + yaml.safe_dump({"clips": clips}, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )


# -- main --------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("samples"))
    ap.add_argument("--inat", type=Path, default=Path("data/raw/inat"))
    ap.add_argument("--boxes", type=Path, default=Path("data/interim/inat_boxes_filtered.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    vids = args.out / "videos"
    imgs = args.out / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    sources: list[tuple[str, str, str, str]] = []  # file, origin, licence, notes

    defence = read_clip(LB / "20230609b-def.mp4", 20, 12)
    busy = read_clip(LB / "20230711a-fan.mp4", 30, 12)
    quiet = read_clip(LB / "20230711b-fan.mp4", 40, 12)
    for name, frames, src, note in (
        ("entrance_guards.mp4", defence, "20230609b-def.mp4", "guard bees at the entrance (defence behaviour)"),
        ("entrance_busy.mp4", busy, "20230711a-fan.mp4", "busy landing board, fanning bees"),
        ("entrance_morning.mp4", quiet, "20230711b-fan.mp4", "normal traffic"),
    ):
        write_clip(frames, vids / name)
        sources.append(
            (
                f"videos/{name}",
                f"{LB_CREDIT}, file `{src}` (12 s excerpt)",
                "CC BY 4.0",
                f"{note}; cut and downscaled to 960x540",
            )
        )

    write_clip(night(quiet), vids / "night.mp4")
    sources.append(
        (
            "videos/night.mp4",
            f"{LB_CREDIT}, file `20230711b-fan.mp4`",
            "CC BY 4.0",
            "**synthetic**: darkened, grey and noisy to imitate dusk / IR footage",
        )
    )
    write_clip(rain(busy), vids / "rain.mp4")
    sources.append(
        (
            "videos/rain.mp4",
            f"{LB_CREDIT}, file `20230711a-fan.mp4`",
            "CC BY 4.0",
            "**synthetic**: rain streaks and lower contrast added",
        )
    )
    write_clip(empty(busy, 150), vids / "empty_entrance.mp4")
    sources.append(
        (
            "videos/empty_entrance.mp4",
            f"{LB_CREDIT}, file `20230711a-fan.mp4`",
            "CC BY 4.0",
            "**synthetic**: temporal median of the clip (bees removed) plus sensor noise",
        )
    )

    # Wikimedia Commons fanning clip.
    raw = Path("data/raw/commons")
    raw.mkdir(parents=True, exist_ok=True)
    webm = raw / "honey_bees_fanning_1.webm"
    if not webm.exists():
        req = urllib.request.Request(COMMONS_FANNING[0], headers={"User-Agent": "smart-hive-samples/0.1"})
        with urllib.request.urlopen(req, timeout=300) as r:
            webm.write_bytes(r.read())
    write_clip(read_clip(webm, 0, 12), vids / "fanning_commons.mp4")
    sources.append(
        (
            "videos/fanning_commons.mp4",
            f"BorekT, Wikimedia Commons, [{COMMONS_FANNING[1]}]({COMMONS_FANNING[1]})",
            "CC BY-SA 4.0",
            "worker bees fanning at the entrance; first 12 s, downscaled to 960x540",
        )
    )

    # Hornet / wasp composites.
    boxes = json.loads(args.boxes.read_text(encoding="utf-8"))
    attribution = {}
    with (args.inat / "attribution.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            attribution[row["file"]] = row
    sprites = {
        f: pick_sprites(args.inat, boxes, attribution, f, 3, args.seed)
        for f in ("vespa_velutina", "vespa_crabro", "vespula")
    }
    bee_px = 40  # median bee length (px) at 960x540 in these clips; V. velutina is ~2x, a wasp ~1.3x
    plans = [
        (
            "hornet_velutina_hawking.mp4",
            "vespa_velutina",
            0,
            quiet,
            hover_path(480, 300, 60, 25),
            2.0,
            "Asian hornet hovering in front of the entrance (should raise an alert)",
        ),
        (
            "hornet_velutina_busy.mp4",
            "vespa_velutina",
            1,
            busy,
            hover_path(420, 330, 80, 30),
            2.0,
            "Asian hornet hawking over a busy board (should raise an alert)",
        ),
        (
            "hornet_crabro_hawking.mp4",
            "vespa_crabro",
            0,
            defence,
            hover_path(500, 360, 60, 20),
            2.3,
            "European hornet hovering (logged, no alert: native species)",
        ),
        (
            "wasp_passing.mp4",
            "vespula",
            0,
            quiet,
            pass_path(260, SIZE[0]),
            1.3,
            "a wasp flying through without hovering (no event expected)",
        ),
    ]
    for name, folder, idx, frames, path_fn, scale, note in plans:
        if len(sprites[folder]) <= idx:
            print(f"no clean sprite for {folder}, skipping {name}")
            continue
        sp = sprites[folder][idx]
        write_clip(composite(frames, sp, int(bee_px * scale), path_fn, seed=args.seed), vids / name)
        sources.append(
            (
                f"videos/{name}",
                f"background: {LB_CREDIT}; insect: {credit(sp)}",
                f"CC BY 4.0 (background) + {sp['licence'].upper().replace('CC-', 'CC ')} (insect)",
                f"**synthetic composite**: {note}",
            )
        )

    # Still images: one entrance frame and one photo per species.
    cv2.imwrite(str(imgs / "landing_board.jpg"), busy[0], [cv2.IMWRITE_JPEG_QUALITY, 88])
    sources.append(
        (
            "images/landing_board.jpg",
            f"{LB_CREDIT}, file `20230711a-fan.mp4`",
            "CC BY 4.0",
            "first frame of the busy clip",
        )
    )
    for folder, sps in sprites.items():
        for j, sp in enumerate(sps[:2]):
            img = cv2.imread(str(args.inat / sp["key"]))
            name = f"{folder}_{j + 1}.jpg"
            cv2.imwrite(str(imgs / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            sources.append(
                (
                    f"images/{name}",
                    credit(sp),
                    sp["licence"].upper().replace("CC-", "CC "),
                    "unchanged",
                )
            )

    write_manifest(args.out, [f for f, *_ in sources if f.startswith("videos/")])

    lines = [
        "# Sample sources and licences",
        "",
        "Every file in `samples/` is listed here. Files marked **synthetic** were generated by",
        "`tools/make_samples.py` from the listed source; they are modified versions and keep the",
        "source licence. Nothing here comes from a source whose licence is unclear: VnPollenBee",
        "images, for example, are only downloaded by script for local training.",
        "",
        "| File | Origin | Licence | Notes |",
        "|---|---|---|---|",
    ]
    lines += [f"| `{f}` | {o} | {lic} | {n} |" for f, o, lic, n in sources]
    lines += [
        "",
        "Licence texts: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), "
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), "
        "[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).",
        "",
    ]
    (args.out / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out / 'SOURCES.md'} ({len(sources)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
