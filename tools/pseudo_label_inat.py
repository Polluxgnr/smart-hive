"""Box the insect in each iNaturalist photo with an open-vocabulary detector.

iNaturalist photos carry a species label but no box. YOLOE, Ultralytics'
open-vocabulary detector (text prompts: hornet, wasp, bee, insect), proposes
boxes. (YOLO-World v2 was tried first and found an insect in only 1-2 of 20
photos, so it was dropped.) we keep, per photo, the boxes whose
raw boxes (score >= 0.05, at most 5 per photo after NMS) are saved; the dataset
builder then keeps boxes scoring at least ``--min-conf`` and half the photo's best,
and drops photos with none or with more than 3 (nests, swarms).
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
    ap.add_argument("--imgsz", type=int, default=448, help="iNaturalist 'medium' photos are 500 px")
    ap.add_argument("--max-bees", type=int, default=150, help="honey-bee photos to label")
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
    # Vespids first (they matter most), and only a few honey-bee photos: bee crops come
    # mainly from the landing-board frames.
    order = ["vespa_velutina", "vespa_crabro", "vespula", "polistes", "apis_mellifera"]
    files = []
    for folder in order:
        found = sorted((args.inat / folder).glob("*.jpg"))
        files += found[: args.max_bees] if folder == "apis_mellifera" else found
    # Resume: results and the list of photos already looked at are saved as we go.
    done_path = args.out.with_suffix(".done.json")
    result: dict[str, list[list[float]]] = json.loads(args.out.read_text()) if args.out.exists() else {}
    done: set[str] = set(json.loads(done_path.read_text())) if done_path.exists() else set()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        args.out.write_text(json.dumps(result), encoding="utf-8")
        done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")

    for i, f in enumerate(files):
        rel = f.relative_to(args.inat).as_posix()
        if rel in done:
            continue
        # One image per call: a list source would be stacked into a single huge batch.
        r = model.predict(str(f), imgsz=args.imgsz, conf=0.05, verbose=False)[0]
        done.add(rel)
        boxes = r.boxes
        if boxes is not None and len(boxes):
            # Keep every box >= 0.05 (top 5); the confidence cut is applied when building datasets.
            raw = [
                [*map(float, b), float(c)]
                for b, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), strict=True)
            ]
            result[rel] = _dedupe(raw)[:5]
        if (i + 1) % 100 == 0:
            save()
            print(f"{i + 1}/{len(files)} photos, kept {len(result)}", flush=True)
    save()
    kept = len(result)
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
