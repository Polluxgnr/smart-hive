"""Hornet evaluation: hawking events on held-out composites, false alerts on real footage.

1. Composite test clips: 10 s windows of the *test-hive* videos with an insect cut
   out of a *test-split* iNaturalist photo (GrabCut mask, unlike the elliptical
   paste used for training) that either hovers (V. velutina, V. crabro) or flies
   straight through (wasp), plus clips with no insect. Expected: one event per
   hovering hornet, an alert only for V. velutina, nothing otherwise.
2. False alerts: the full pipeline on the real, hornet-free annotated videos.

Writes docs/results/hornet.json.

Usage:
    python tools/eval_hornet.py --per-kind 12
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_samples import composite, hover_path, pass_path, pick_sprites, read_clip

from smarthive.alerts import Notifier
from smarthive.config import Config
from smarthive.pipeline import HivePipeline
from smarthive.storage import Storage
from smarthive.vision import Perception

TEST_VIDEOS = ["20230609b-def", "20230711a-fan", "20230711b-fan"]
REAL_VIDEOS = ["20230609b-def", "20230711a-fan", "20230711b-fan", "yt5", "yt8"]
KINDS = {
    "velutina_hover": ("vespa_velutina", "hover"),
    "crabro_hover": ("vespa_crabro", "hover"),
    "wasp_pass": ("vespula", "pass"),
    "none": (None, None),
}


def write_tmp(frames, path: Path, fps: float = 25.0) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def run_pipeline(perception, video: Path, fps: float, tmp: Path) -> tuple[list[dict], int, float]:
    cfg = Config()
    cfg.storage.data_dir = str(tmp / "db")
    cfg.camera.process_fps = fps
    storage = Storage(cfg.storage.data_dir)
    storage.clear_hive(cfg.hive.id)
    notifier = Notifier(cfg.alerts, dry_run=True)
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        s = HivePipeline(cfg, storage, notifier, perception).run(str(video), start_time=datetime(2026, 7, 1, 12))
    storage.close()
    return s.hornet_events, s.alerts, s.seconds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-kind", type=int, default=12)
    ap.add_argument("--fps", type=float, default=12.5)
    ap.add_argument("--inat", type=Path, default=Path("data/raw/inat"))
    ap.add_argument("--boxes", type=Path, default=Path("data/interim/inat_boxes.json"))
    ap.add_argument("--split", type=Path, default=Path("data/splits/inat_split.json"))
    ap.add_argument("--tracking", type=Path, default=Path("data/raw/landing_board/tracking"))
    ap.add_argument("--out", type=Path, default=Path("docs/results/hornet.json"))
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    boxes = json.loads(args.boxes.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_boxes = {k: v for k, v in boxes.items() if split.get(k) == "test"}
    attribution = {}
    with (args.inat / "attribution.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            attribution[row["file"]] = {**row, "licence": "cc-by"}  # licence filter not needed for evaluation
    sprites = {
        f: pick_sprites(args.inat, test_boxes, attribution, f, args.per_kind, args.seed)
        for f in ("vespa_velutina", "vespa_crabro", "vespula")
    }
    print({k: len(v) for k, v in sprites.items()}, "test sprites")

    perception = Perception("models/detector.pt", "models/classifier.pt")
    results = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for kind, (folder, motion) in KINDS.items():
            pool = sprites.get(folder, []) if folder else [None] * args.per_kind
            for i, sp in enumerate(pool[: args.per_kind]):
                video = rng.choice(TEST_VIDEOS)
                frames = read_clip(args.tracking / f"{video}.mp4", rng.uniform(5, 100), 10)
                if sp is not None:
                    size = int(40 * (2.0 if folder != "vespula" else 1.3) * rng.uniform(0.85, 1.15))
                    cx, cy = rng.uniform(250, 710), rng.uniform(200, 400)
                    path = (
                        hover_path(cx, cy, rng.uniform(30, 90), rng.uniform(15, 35))
                        if motion == "hover"
                        else pass_path(rng.uniform(150, 450), 960)
                    )
                    frames = composite(frames, sp, size, path, seed=i)
                clip = tmp / f"{kind}_{i}.mp4"
                write_tmp(frames, clip)
                events, alerts, _ = run_pipeline(perception, clip, args.fps, tmp)
                rec = {
                    "kind": kind,
                    "i": i,
                    "video": video,
                    "events": [e["species"] for e in events],
                    "alerts": alerts,
                    "sprite": sp["key"] if sp else None,
                }
                results.append(rec)
                print(json.dumps(rec), flush=True)

        real = []
        if not args.skip_real:
            for name in REAL_VIDEOS:
                events, alerts, seconds = run_pipeline(perception, args.tracking / f"{name}.mp4", args.fps, tmp)
                real.append(
                    {"video": name, "seconds": seconds, "events": [e["species"] for e in events], "alerts": alerts}
                )
                print(json.dumps(real[-1]), flush=True)

    def stats(kind: str) -> dict:
        rows = [r for r in results if r["kind"] == kind]
        return {
            "clips": len(rows),
            "with_event": sum(bool(r["events"]) for r in rows),
            "with_velutina_alert": sum(r["alerts"] > 0 for r in rows),
            "species_of_events": [s for r in rows for s in r["events"]],
        }

    summary = {k: stats(k) for k in KINDS}
    v = summary["velutina_hover"]
    false_alert_clips = sum(summary[k]["with_velutina_alert"] for k in ("crabro_hover", "wasp_pass", "none"))
    tp = v["with_velutina_alert"]
    summary["alert_recall"] = tp / v["clips"] if v["clips"] else None
    summary["alert_precision"] = tp / (tp + false_alert_clips) if tp + false_alert_clips else None
    real_seconds = sum(r["seconds"] for r in real)
    summary["real_footage"] = {
        "seconds": real_seconds,
        "velutina_alerts": sum(r["alerts"] for r in real),
        "vespid_events": sum(len(r["events"]) for r in real),
        "videos": real,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "clips": results}, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "real_footage"}, indent=1))
    print("real footage:", {k: v for k, v in summary["real_footage"].items() if k != "videos"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
