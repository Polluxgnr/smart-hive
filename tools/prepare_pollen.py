"""Download part of VnPollenBee and cut pollen / no-pollen bee crops.

VnPollenBee (Ecological Informatics, 2024; comvis-hust.github.io/datasets/pollenbee.html)
is shared from a Google Drive folder with no licence stated, so it is fetched
here for local training only and **never redistributed** (not in samples/, not
in the repo). Annotations are LabelMe JSON files next to the images.

The JSON is parsed as data; nothing from the download is executed.

Usage:
    python tools/prepare_pollen.py --max-images 600
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2

FOLDER = "https://drive.google.com/drive/folders/1fdEcu7CNmEkVAamu9wh_Ppw_-uW3VNY1"


def list_files(retries: int = 5) -> list[tuple[str, str]]:
    import gdown

    last = None
    for attempt in range(retries):
        try:
            files = gdown.download_folder(url=FOLDER, skip_download=True, quiet=True)
            return [(f.id, f.path.replace("\\", "/")) for f in files]
        except Exception as exc:  # Drive listing is flaky
            last = exc
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"cannot list the VnPollenBee folder: {last}")


def fetch(file_id: str, dest: Path) -> bool:
    import gdown

    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            gdown.download(id=file_id, output=str(dest), quiet=True)
            if dest.exists() and dest.stat().st_size > 0:
                return True
        except Exception:
            time.sleep(5 * (attempt + 1))
    return False


def crops_from(img_path: Path, ann_path: Path) -> list[tuple[str, object]]:
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    im = cv2.imread(str(img_path))
    if im is None:
        return []
    out = []
    for shape in ann.get("shapes", []):
        label = str(shape.get("label", "")).lower()
        pts = shape.get("points") or []
        if len(pts) < 2:
            continue
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        side = max(max(xs) - min(xs), max(ys) - min(ys)) * 1.5
        H, W = im.shape[:2]
        a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        c, d = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
        crop = im[b:d, a:c]
        if crop.size == 0:
            continue
        cls = "bee" if "non" in label or "no" in label.split("_") else "bee_pollen" if "pollen" in label else "bee"
        out.append((cls, crop))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/vnpollenbee"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/pollen_crops"))
    ap.add_argument("--max-images", type=int, default=600)
    ap.add_argument("--max-nonpollen-per-image", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    listing_path = args.raw / "listing.json"
    if listing_path.exists():
        files = [tuple(x) for x in json.loads(listing_path.read_text(encoding="utf-8"))]
    else:
        files = list_files()
        args.raw.mkdir(parents=True, exist_ok=True)
        listing_path.write_text(json.dumps(files), encoding="utf-8")
    by_stem: dict[str, dict[str, tuple[str, str]]] = {}
    for fid, path in files:
        p = Path(path)
        top = p.parts[0].lower()
        part = "test" if top.startswith("test") else "val" if top.startswith("val") else "train"
        by_stem.setdefault(f"{part}/{p.stem}", {})[p.suffix.lower()] = (fid, path)
    pairs = [(k, v) for k, v in sorted(by_stem.items()) if ".json" in v and (".jpg" in v or ".png" in v)]
    print(f"{len(pairs)} annotated images listed")
    rng.shuffle(pairs)
    # Keep the official split; take a proportional subset.
    n = {"train": 0, "val": 0, "test": 0}
    quota = {
        "train": int(args.max_images * 0.7),
        "val": int(args.max_images * 0.15),
        "test": int(args.max_images * 0.15),
    }
    for key, v in pairs:
        part = key.split("/")[0]
        if n[part] >= quota[part]:
            continue
        img_id, img_path = v.get(".jpg") or v[".png"]
        ann_id, ann_path = v[".json"]
        local_img = args.raw / img_path
        local_ann = args.raw / ann_path
        if not (fetch(ann_id, local_ann) and fetch(img_id, local_img)):
            continue
        crops = crops_from(local_img, local_ann)
        pollen = [c for c in crops if c[0] == "bee_pollen"]
        other = [c for c in crops if c[0] == "bee"]
        rng.shuffle(other)
        for j, (cls, crop) in enumerate(pollen + other[: args.max_nonpollen_per_image]):
            d = args.out / part / cls
            d.mkdir(parents=True, exist_ok=True)
            side = max(crop.shape[:2])  # type: ignore[union-attr]
            if side > 128:
                s = 128 / side
                crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))  # type: ignore[union-attr]
            cv2.imwrite(str(d / f"{Path(img_path).stem.replace(' ', '_')}_{j}.jpg"), crop)
        n[part] += 1
        if sum(n.values()) % 50 == 0:
            print(n, flush=True)
    print("done", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
