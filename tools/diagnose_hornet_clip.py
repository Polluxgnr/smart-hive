"""Frame-by-frame diagnosis of a hornet composite clip.

The composite clips are generated with a known flight path (tools/make_samples.py), so for
every analysed frame we know where the hornet is. This prints, stage by stage, where it is
lost: detector box, detector label, classifier vote, track-level sighting, presence rule.

Usage:
    python tools/diagnose_hornet_clip.py samples/videos/hornet_velutina_hawking.mp4 480 300 60 25
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_samples import hover_path

from smarthive.hornet import HornetMonitor
from smarthive.tracks import SPECIES, VESPIDS, Track
from smarthive.vision import Perception


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("cx", type=float)
    ap.add_argument("cy", type=float)
    ap.add_argument("rx", type=float)
    ap.add_argument("ry", type=float)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--hornet-px", type=float, default=80.0)
    args = ap.parse_args(argv)

    cap = cv2.VideoCapture(args.clip)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    path = hover_path(args.cx, args.cy, args.rx, args.ry)
    perc = Perception("models/detector.pt", "models/classifier.pt")
    mon = HornetMonitor(None, math.hypot(960, 540))
    tracks: dict[int, Track] = {}
    stats = {"present": 0, "boxed": 0, "boxed_vespid": 0, "cls_vespid": 0, "sighting": 0}
    cls_sum = [0.0] * len(SPECIES)
    events = 0
    k = 0
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if i % args.stride:
            continue
        k += 1
        t = i / fps
        dets = perc.track(frame)
        for d in dets:
            tr = tracks.setdefault(d.track_id, Track(d.track_id, first_t=t))
            tr.add(t, d.center, d.size, d.label == "vespid", d.conf)
        probs = perc.classify(frame, [d.box for d in dets])
        for d, p in zip(dets, probs, strict=False):
            tracks[d.track_id].add_probs(p, k)
        opened, _ = mon.update(t, tracks, {d.track_id for d in dets})
        events += len(opened)

        pos = path(i / n_frames, i)
        if pos is None:
            continue
        stats["present"] += 1
        hx, hy, _ = pos
        near = [
            (d, p)
            for d, p in zip(dets, probs, strict=False)
            if math.hypot(d.center[0] - hx, d.center[1] - hy) < 0.6 * args.hornet_px
        ]
        if not near:
            continue
        d, p = min(near, key=lambda dp: math.hypot(dp[0].center[0] - hx, dp[0].center[1] - hy))
        stats["boxed"] += 1
        stats["boxed_vespid"] += d.label == "vespid"
        vesp = sum(p[SPECIES.index(s)] for s in VESPIDS)
        stats["cls_vespid"] += vesp >= 0.6
        cls_sum = [a + b for a, b in zip(cls_sum, p, strict=True)]
        tr = tracks[d.track_id]
        stats["sighting"] += mon.is_sighting(tr)
    present = max(1, stats["present"])
    boxed = max(1, stats["boxed"])
    print(f"clip: {args.clip}")
    print(f"  hornet on screen in {stats['present']} analysed frames")
    print(
        f"  boxed by the detector:           {stats['boxed']:4d} ({stats['boxed'] / present:.0%} of frames on screen)"
    )
    print(
        f"  ... labelled 'vespid' by it:     {stats['boxed_vespid']:4d} ({stats['boxed_vespid'] / boxed:.0%} of boxed)"
    )
    print(f"  ... classifier vespid >= 0.6:    {stats['cls_vespid']:4d} ({stats['cls_vespid'] / boxed:.0%} of boxed)")
    print(f"  ... its track counts as sighting:{stats['sighting']:4d} ({stats['sighting'] / boxed:.0%} of boxed)")
    print(
        "  mean classifier output on the hornet crops:",
        {s: round(v / boxed, 2) for s, v in zip(SPECIES, cls_sum, strict=True)},
    )
    print(f"  events opened: {events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
