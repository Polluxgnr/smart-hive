"""Counting evaluation: IN/OUT from our detector + ByteTrack vs the dataset's manual tracks.

For each annotated video of the landing-board dataset, the *same* zone rule is
applied to (a) the manually annotated tracks and (b) the tracks produced by our
pipeline. The difference measures detection + tracking error, not the rule.

The entrance zone is a band along the top edge of the annotated board polygon
(the slot side), from 10 % above the edge to 25 % of the board depth below it.
Only the annotated board area is evaluated, so predictions outside it are dropped.

Outputs docs/results/counting.json and a figure in docs/img/.

Usage:
    python tools/eval_counting.py --fps 12.5 25
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from smarthive.counting import EntranceCounter, ZoneRule
from smarthive.geometry import point_in_polygon

VIDEOS = ["20230609b-def", "20230711a-fan", "20230711b-fan", "yt5", "yt8"]
WINDOW_S = 30.0
MIN_TRAVEL = 0.03  # same default as EntranceCfg.min_travel


def board_polygon(path: Path) -> list[tuple[float, float]]:
    """Parse ``polygon = np.array([[x, y], ...])`` as data (never executed)."""
    pts = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", path.read_text())
    return [(float(x), float(y)) for x, y in pts]


def slot_zone(poly: list[tuple[float, float]], below: float = 0.25, above: float = 0.10) -> list[tuple[float, float]]:
    p0, p1, p2, p3 = (np.array(p) for p in poly[:4])
    d1, d0 = p2 - p1, p3 - p0
    return [tuple(p0 - above * d0), tuple(p1 - above * d1), tuple(p1 + below * d1), tuple(p0 + below * d0)]


def gt_observations(tracks_file: Path, w: int, h: int) -> dict[int, dict[int, tuple[float, float]]]:
    a = np.loadtxt(tracks_file, delimiter=",")
    frames: dict[int, dict[int, tuple[float, float]]] = {}
    for f, tid, x, y, bw, bh in a[:, :6]:
        frames.setdefault(int(f), {})[int(tid)] = ((x + bw / 2) * w, (y + bh / 2) * h)
    return frames


def count(
    frames: dict[int, dict[int, tuple[float, float]]],
    fps: float,
    zone,
    width: int,
    stride: int = 1,
    max_age: float = 1.5,
):
    """Run the zone counter; returns list of (t, direction)."""
    c = EntranceCounter(ZoneRule(zone), max_age=max_age, min_travel=MIN_TRAVEL * width)
    events = []
    for f in sorted(frames):
        if (f - 1) % stride:
            continue
        events += [(e.t, e.direction) for e in c.update(f / fps, frames[f])]
    events += [(e.t, e.direction) for e in c.flush()]
    return events


def predict_tracks(video: Path, poly, detector: str, proc_fps: float, imgsz: int, conf: float):
    from smarthive.vision import Perception

    perc = Perception(detector, None, imgsz=imgsz, conf=conf)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    stride = max(1, round(fps / proc_fps))
    frames: dict[int, dict[int, tuple[float, float]]] = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % stride:
            continue
        obs = {}
        for d in perc.track(frame):
            if d.label == "bee" and point_in_polygon(d.center, poly):
                obs[d.track_id] = d.center
        frames[i] = obs
    cap.release()
    return frames, fps, stride


def windows(events, duration: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(np.ceil(duration / WINDOW_S))
    ins, outs = np.zeros(n), np.zeros(n)
    for t, d in events:
        k = min(n - 1, int(t // WINDOW_S))
        (ins if d == "in" else outs)[k] += 1
    return ins, outs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/raw/landing_board/tracking"))
    ap.add_argument("--detector", default="models/detector.pt")
    ap.add_argument("--fps", type=float, nargs="+", default=[12.5, 25.0])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cache", type=Path, default=Path("data/interim/pred_tracks"))
    ap.add_argument("--out", type=Path, default=Path("docs/results/counting.json"))
    args = ap.parse_args(argv)
    results = []
    for name in VIDEOS:
        video = args.data / f"{name}.mp4"
        cap = cv2.VideoCapture(str(video))
        w, h, fps, n = (int(cap.get(3)), int(cap.get(4)), cap.get(5), int(cap.get(7)))
        cap.release()
        duration = n / fps
        poly = board_polygon(args.data / f"entrance_zone_{name}.txt")
        zone = slot_zone(poly)
        gt_frames = gt_observations(args.data / f"tracks_{name}.txt", w, h)
        gt = count(gt_frames, fps, zone, w)
        gt_in, gt_out = windows(gt, duration)
        row = {
            "video": name,
            "duration_s": round(duration, 1),
            "gt_in": int(gt_in.sum()),
            "gt_out": int(gt_out.sum()),
            "unseen_hive": name.startswith("yt"),
            "runs": [],
        }
        for proc_fps in args.fps:
            stride = max(1, round(fps / proc_fps))
            # Reference: manual tracks seen at the same frame rate (isolates the frame-rate effect).
            gt_sub = count(gt_frames, fps, zone, w, stride=stride)
            cache = args.cache / f"{name}_{proc_fps:g}fps.json"
            if cache.exists():
                pred_frames = {
                    int(k): {int(i): tuple(p) for i, p in v.items()} for k, v in json.loads(cache.read_text()).items()
                }
            else:
                pred_frames, _, _ = predict_tracks(video, poly, args.detector, proc_fps, args.imgsz, args.conf)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(pred_frames))
            pred = count(pred_frames, fps, zone, w, max_age=max(1.5, 30 * stride / fps))
            p_in, p_out = windows(pred, duration)
            s_in, s_out = windows(gt_sub, duration)
            run = {
                "fps": proc_fps,
                "pred_in": int(p_in.sum()),
                "pred_out": int(p_out.sum()),
                "gt_subsampled_in": int(s_in.sum()),
                "gt_subsampled_out": int(s_out.sum()),
                "mae_per_30s_in": float(np.mean(np.abs(p_in - gt_in))),
                "mae_per_30s_out": float(np.mean(np.abs(p_out - gt_out))),
                "gt_mean_per_30s_in": float(gt_in.mean()),
                "gt_mean_per_30s_out": float(gt_out.mean()),
            }
            row["runs"].append(run)
            print(name, json.dumps(run), flush=True)
        results.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
