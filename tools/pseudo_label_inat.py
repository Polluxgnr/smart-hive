"""Box the insect in each iNaturalist photo with an open-vocabulary detector.

iNaturalist photos carry a species label but no box. YOLOE, Ultralytics'
open-vocabulary detector (text prompts: hornet, wasp, bee, insect), proposes
boxes. (YOLO-World v2 was tried first and found an insect in only 1-2 of 20
photos, so it was dropped.) we keep, per photo, the boxes whose
score is at least ``--min-conf`` and at least half of the best score in that
photo, then drop photos with more than ``--max-boxes`` (nests, swarms) or none.
The species comes from the iNaturalist label, never from the prompt.

Output: data/interim/inat_boxes.json  {relative_path: [[x1, y1, x2, y2, score], ...]}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inat", type=Path, default=Path("data/raw/inat"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/inat_boxes.json"))
    ap.add_argument("--weights", default="models/third_party/yoloe-11s-seg.pt")
    ap.add_argument("--min-conf", type=float, default=0.15)
    ap.add_argument("--max-boxes", type=int, default=3)
    args = ap.parse_args(argv)

    import subprocess
    import sys

    from ultralytics import YOLO

    # Encoding the prompts needs a text model (several hundred MB of RAM). Do it once in a
    # child process and save a model with the prompt embeddings baked in.
    baked = Path(args.weights).with_name("yoloe-11s-insects.pt")
    if not baked.exists():
        code = (
            "from ultralytics import YOLOE; import sys; m = YOLOE(sys.argv[1]); "
            "n = ['hornet', 'wasp', 'bee', 'insect']; m.set_classes(n, m.get_text_pe(n)); m.save(sys.argv[2])"
        )
        subprocess.run([sys.executable, "-c", code, args.weights, str(baked)], check=True)
    model = YOLO(str(baked))
    files = sorted(p for p in args.inat.rglob("*.jpg"))
    result: dict[str, list[list[float]]] = {}
    kept = 0
    for i, f in enumerate(files):
        # One image per call: a list source would be stacked into a single huge batch.
        r = model.predict(str(f), imgsz=640, conf=0.05, verbose=False)[0]
        rel = f.relative_to(args.inat).as_posix()
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        best = float(conf.max())
        if best < args.min_conf:
            continue
        sel = [
            [*map(float, b), float(s)] for b, s in zip(xyxy, conf, strict=True) if s >= max(args.min_conf, 0.5 * best)
        ]
        sel = _dedupe(sel)
        if 0 < len(sel) <= args.max_boxes:
            result[rel] = sel
            kept += 1
        if (i + 1) % 250 == 0:
            print(f"{i + 1}/{len(files)} photos, kept {kept}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result), encoding="utf-8")
    print(f"kept {kept}/{len(files)} photos -> {args.out}")
    return 0


def _iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _dedupe(boxes: list[list[float]], thr: float = 0.5) -> list[list[float]]:
    """Class-agnostic NMS: the same insect is often found by several prompts."""
    out: list[list[float]] = []
    for b in sorted(boxes, key=lambda x: -x[4]):
        if all(_iou(b, o) < thr for o in out):
            out.append(b)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
